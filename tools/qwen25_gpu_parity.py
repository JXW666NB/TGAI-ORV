# -*- coding: utf-8 -*-
"""Qwen2.5-0.5B 在 A10 上的 GPU 前向对齐：我们引擎 vs HuggingFace transformers。

参考值**在现场用 transformers 现算**（不依赖事先导出的 npz），
所以这条验证对权重/代码版本都自洽。

用法（A10）：
    python tools/qwen25_gpu_parity.py --dir /mnt/workspace/qwen25 --device cuda
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

# 固定输入序列（只用于对齐，内容无所谓）
IDS = [151643, 9707, 1879, 264, 4326, 13, 151645]


def load_example():
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft"] = mod
    spec.loader.exec_module(mod)
    return mod


def hf_reference(dirpath, ids, device):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        dirpath, dtype=torch.float32, local_files_only=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long, device=model.device))
    logits = out.logits.float().cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return logits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/workspace/qwen25")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    q = load_example()
    cfg = dict(q.CFG)

    print("== 1) 用 transformers 现算参考 logits ==")
    ref = hf_reference(a.dir, IDS, a.device)
    print("   参考形状", ref.shape, "最后一位前 3:", np.round(ref[0, -1, :3], 5))

    print("== 2) 我们的引擎 ==")
    import tgai_go.train as T
    T.set_default_device(a.device)
    w = q.load_safetensors_weights(a.dir, cfg)
    print("   权重 %d 个" % len(w))
    model = q.Qwen25(cfg, w)
    del w
    ids = np.asarray(IDS, dtype=np.int64).reshape(1, -1)
    logits = model.forward(ids)
    got = np.asarray(logits.to_numpy(), dtype=np.float32)
    print("   我们形状", got.shape, "最后一位前 3:", np.round(got[0, -1, :3], 5))

    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    denom = float(np.abs(ref).max()) or 1.0
    rel = float(d.max()) / denom
    print("")
    print("max|Δ| = %.3e   相对误差 = %.3e" % (float(d.max()), rel))
    ok = rel <= a.tol
    print("判据 ≤%.0e：%s" % (a.tol, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
