# -*- coding: utf-8 -*-
"""Qwen3 在 A10 上的 GPU 前向对齐：我们引擎 vs HuggingFace transformers。

参考值**现场用 transformers 现算**（不依赖事先导出的文件），所以对权重/代码版本自洽。

## 为什么必须关 TF32

HF 在 A100/A10 上默认允许 TF32，**参考值本身就带 ~1e-3 误差**，会把我们的 1e-6
盖掉，导致"看起来没对齐"。所以脚本里先关掉 TF32，再做比较。

## 判据

* `max|Δ| / max|ref| ≤ 1e-5`；
* 失败时**先用 float64 再算一遍**归因：
  - "我们 fp32 vs HF fp64" ≤1e-5 但 "fp32 vs fp32" >1e-5 ⇒ 差异来自累加顺序，不是结构错；
  - fp64 也超 ⇒ 结构/权重错（先查 head_dim、q_norm 位置、tie、RoPE）。
* 额外做一次**梯度冒烟**：`q_norm.weight` 必须有梯度且非全零
  （CUDA 上的反向无法在 CPU CI 里验证）。

用法（A10）：
    python tools/qwen3_gpu_parity.py --dir /mnt/workspace/qwen3 --device cuda
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

IDS = [151643, 9707, 1879, 264, 4326, 13, 151645]


def load_example():
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft"] = mod
    spec.loader.exec_module(mod)
    return mod


def cfg_from_hf(dirpath, q):
    """从 config.json 生成我们引擎的 cfg（含 Qwen3 的 DH/QK_NORM/QKV_BIAS 开关）。"""
    import json
    j = json.load(open(os.path.join(dirpath, "config.json"), encoding="utf-8"))
    D = int(j["hidden_size"])
    NH = int(j["num_attention_heads"])
    DH = int(j.get("head_dim") or D // NH)
    return dict(V=int(j["vocab_size"]), D=D, L=int(j["num_hidden_layers"]), NH=NH,
                KV=int(j.get("num_key_value_heads", NH)), FF=int(j["intermediate_size"]),
                THETA=float(j.get("rope_theta", 1e6)),
                EPS=float(j.get("rms_norm_eps", 1e-6)),
                DH=DH,
                # 下面两个开关最后会以**权重实际内容**为准（见 main），config 只作初值
                QK_NORM=False, QKV_BIAS=bool(j.get("attention_bias", False)),
                TIE=bool(j.get("tie_word_embeddings", True)))


def hf_logits(dirpath, ids, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    # ★ 关 TF32：否则参考值本身带 ~1e-3 误差，会盖掉我们的 1e-6
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:                              # noqa: BLE001
        pass
    model = AutoModelForCausalLM.from_pretrained(
        dirpath, dtype=dtype, local_files_only=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long, device=model.device))
    logits = out.logits.float().cpu().numpy()
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:                              # noqa: BLE001
        pass
    return logits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/workspace/qwen3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    q = load_example()
    cfg = cfg_from_hf(a.dir, q)
    print("结构：L=%d D=%d NH=%d KV=%d DH=%d FF=%d V=%d theta=%g eps=%g"
          % (cfg["L"], cfg["D"], cfg["NH"], cfg["KV"], cfg["DH"], cfg["FF"],
             cfg["V"], cfg["THETA"], cfg["EPS"]))
    print("config 初值：DH=%s（稍后以权重实际内容为准）" % (cfg["DH"],))

    print("== 1) transformers 现场算参考（fp32，TF32 已关）==")
    import torch
    ref = hf_logits(a.dir, IDS, torch.float32)
    print("   参考 [1,-1,:3] =", np.round(ref[0, -1, :3], 6))

    print("== 2) 我们的引擎 ==")
    import tgai_go.train as T
    T.set_default_device(a.device)
    ck = q.C.load_safetensors_dir(a.dir)
    # 开关以**权重实际内容**为准，别靠 config 猜（Qwen3 config 里没有 q_norm 字段）
    cfg = dict(cfg)
    cfg.update(ck.config.get("head_dim") and {"DH": int(ck.config["head_dim"])} or {},
               QK_NORM=any("q_norm" in k for k in ck.weights),
               QKV_BIAS=any(k.endswith("q_proj.bias") for k in ck.weights),
               TIE=bool(ck.config.get("tie_embeddings", True)))
    w = dict(ck.weights)
    cfg = q.cfg_from_checkpoint(ck)
    print("   权重 %d 个；tie=%s；有 lm_head: %s" % (len(w), cfg["TIE"], "lm_head.weight" in w))
    print("   开关（以权重为准）：QK_NORM=%s QKV_BIAS=%s DH=%s vs D/NH=%s"
          % (cfg["QK_NORM"], cfg["QKV_BIAS"], cfg["DH"], cfg["D"] // cfg["NH"]))
    model = q.Qwen25(cfg, w)
    del w
    ids = np.asarray(IDS, dtype=np.int64).reshape(1, -1)
    got = np.asarray(model.forward(ids).to_numpy(), dtype=np.float32)
    print("   我们 [1,-1,:3] =", np.round(got[0, -1, :3], 6))

    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    denom = float(np.abs(ref).max()) or 1.0
    rel = float(d.max()) / denom
    print("")
    print("max|Δ| = %.3e   相对误差 = %.3e   判据 ≤%.0e：%s"
          % (float(d.max()), rel, a.tol, "PASS" if rel <= a.tol else "FAIL"))

    if rel > a.tol:
        print("")
        print("== 诊断：用 fp64 参考再算一遍，区分'累加顺序'与'结构错' ==")
        try:
            ref64 = hf_logits(a.dir, IDS, torch.float64)
            rel64 = float(np.abs(got.astype(np.float64) - ref64).max()) / (
                float(np.abs(ref64).max()) or 1.0)
            print("   我们 fp32 vs HF fp64 相对误差 = %.3e" % rel64)
            if rel64 <= a.tol:
                print("   ⇒ 差异来自 fp32 累加顺序（SDPA/GEMM），不是结构错")
            else:
                print("   ⇒ fp64 也超差 ⇒ 结构/权重问题：先查 head_dim、q_norm 位置、tie、RoPE")
        except Exception as ex:                    # noqa: BLE001
            print("   fp64 参考不可用：%s" % (ex,))

    # 梯度冒烟：CUDA 反向无法在 CPU CI 验证，这里必须看一下
    print("")
    print("== 3) 梯度冒烟（q_norm 必须有梯度且非全零）==")
    try:
        for prm in model.params():
            prm.zero_grad()
        loss = model.forward(ids).sum()
        loss.backward()
        gname = "blocks.0.attn.q_norm.weight"
        if gname in model.p:
            g = getattr(model.p[gname], "grad", None)
            if g is None:
                print("   FAIL: %s 没有梯度（q_norm 未接入反向）" % gname)
                return 2
            ga = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)
            print("   q_norm 梯度 sum=%.4e  max|g|=%.4e  %s"
                  % (float(ga.sum()), float(np.abs(ga).max()),
                     "OK" if np.isfinite(ga).all() and float(np.abs(ga).max()) > 0 else "异常"))
        else:
            print("   该结构没有 q_norm（不是 Qwen3？）")
    except Exception as ex:                        # noqa: BLE001
        print("   梯度冒烟失败：%s" % (ex,))
        return 2

    return 0 if rel <= a.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
