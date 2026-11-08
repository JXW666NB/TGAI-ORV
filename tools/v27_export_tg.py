# -*- coding: utf-8 -*-
"""把训练栈里的权重**流式导出回 `.TG`（bf16）**，用于保存检查点 / 交回产线。

设计要点：
* 名字表以**原 `.TG` 的 sidecar** 为准（982 个张量；`.pt` 里多的 `lm_head.weight` 是 tied，不写）；
* 训练栈只拥有部分张量（attn 的 q/k/v/out、MoE、ln、embed、final_norm），其余（如 `rope.cos/sin`）
  从原始 `.pt` **原样透传**，保证不丢张量；
* 用 `write_tg4_stream` 逐个写、写一个释放一个 —— 主机峰值 = 单张量（3B 全量不能进内存）；
* 头部字段直接取自原 `.TG` 的 header，确保架构字段一致。

自检（`--verify`）：导出后与给定参考 `.TG` 比较 **sha256**，用于证明"没训练时导出应逐位相同"。

用法：
    python v27_export_tg.py --out D:\\tgai_orv\\export_test.tg --verify D:\\TGAI\\models\\V2.7\\TGAI_NB_V2.7.tg
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)

TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg"
SIDECAR = TG + ".names.json"
log = RUN.log


def sha256_of(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\tgai_orv\export_test.tg")
    ap.add_argument("--verify", default=None, help="参考 .TG，导出后比较 sha256")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    from tgai_go.read_tg import read_tg, close_tg
    from tgai_go.tg4 import write_tg4_stream

    info = read_tg(TG)
    hdr = dict(info.get("header") or {})
    close_tg(info)
    for k in ("magic", "version"):
        hdr.pop(k, None)
    # _pack_header 还要求这些字段；原头部没存，按 V2.7 的实际结构补默认值
    hdr.setdefault("head_dim", int(hdr["d_model"]) // int(hdr["n_heads"]))
    hdr.setdefault("n_kv_heads", int(hdr["n_heads"]))
    hdr.setdefault("arch", 2)
    hdr.setdefault("pad_id", 0)
    hdr.setdefault("bos_id", 2)
    hdr.setdefault("eos_id", 3)
    hdr.setdefault("unk_id", 1)

    names = [t["name"] for t in json.load(open(SIDECAR, encoding="utf-8"))["tensors"]]
    log("名字表 %d 个张量；头部字段：%s" % (len(names), sorted(hdr)))

    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        if args.restore:
            RUN.restore()
        raise SystemExit(3)
    try:
        import torch

        import tgai_go.train as T
        from tgai_go.train.v27_train import V27Trainer

        T.set_default_device("cuda")
        ck = torch.load(RUN.PT, map_location="cpu", mmap=True, weights_only=False)
        mc, sd = ck["model_config"], ck["model_state_dict"]
        cfg = dict(d_model=int(mc["d_model"]), n_heads=int(mc["n_heads"]), d_ff=int(mc["d_ff"]),
                   n_experts=int(mc["n_experts"]), n_activated=int(mc["n_activated"]),
                   n_layers=int(mc["n_layers"]), rope_theta=float(mc.get("rope_theta", 1e4)))
        NL = cfg["n_layers"]
        tr = V27Trainer(cfg, n_layers=NL)
        tr.load_all(sd)
        tr.attach_head(sd)
        log("训练栈装载：%d MiB" % RUN.vram())

        def gpu_np(param):
            """把设备上的参数取回主机（fp32）—— Tensor.to_numpy() 内部会 d2h。"""
            if getattr(param, "_gpu", None) is None and getattr(param, "data", None) is None:
                return None
            return np.asarray(param.to_numpy(), np.float32)

        def cat_heads(mods, axis):
            parts = [gpu_np(m.weight) for m in mods]
            if any(p is None for p in parts):
                return None
            return np.ascontiguousarray(np.concatenate(parts, axis=axis), np.float32)

        def items():
            written = 0
            for name in names:
                arr = None
                if name == "token_embedding.weight":
                    arr = gpu_np(tr.head.weight)
                elif name == "final_norm.weight":
                    arr = gpu_np(tr.final_norm.weight)
                elif ".attn." in name and name.endswith("_proj.weight"):
                    i = int(name.split(".")[1])
                    kind = name.split(".")[-2].split("_")[0]      # q / k / v / out
                    if kind in ("q", "k", "v"):
                        arr = cat_heads(tr.layers[i][kind], 0)
                    elif kind == "out":
                        # 训练栈里是 [DH, D] 的每头切片，拼回 [D, D] 需要转置后按列拼
                        # 每头权重是 (D, DH)：原始 out_proj 是 (D, D)，按**列**拼回去
                        parts = [gpu_np(m.weight) for m in tr.layers[i]["o"]]
                        if all(p is not None for p in parts):
                            arr = np.ascontiguousarray(np.concatenate(parts, axis=1), np.float32)
                elif ".moe.router.weight" in name:
                    i = int(name.split(".")[1])
                    arr = gpu_np(tr.layers[i]["moe"].router.weight)
                elif ".moe.experts." in name and name.endswith(".weight"):
                    i = int(name.split(".")[1])
                    e = int(name.split(".")[4])
                    slot = {"w1": "gate", "w2": "up", "w3": "down"}[name.split(".")[-2]]
                    arr = gpu_np(getattr(tr.layers[i]["moe"].experts[e], slot).weight)
                elif name.endswith("moe._qb_bias"):
                    i = int(name.split(".")[1])
                    qb = tr.layers[i]["moe"].qb_bias
                    arr = np.asarray(qb.to_numpy(), np.float32) if getattr(qb, "_gpu", None) is not None else None
                if arr is None:                     # 透传原权重（如 rope.cos/sin）
                    arr = np.ascontiguousarray(sd[name].float().numpy(), np.float32)
                written += 1
                if written % 200 == 0:
                    log("  已写 %d/%d" % (written, len(names)))
                yield name, arr
                del arr

        t0 = time.time()
        write_tg4_stream(args.out, items(), hdr)
        log("导出完成：%s（%.2f GB，%.1fs）"
            % (args.out, os.path.getsize(args.out) / 2 ** 30, time.time() - t0))
        h_out = sha256_of(args.out)
        log("导出 sha256=%s" % h_out)
        if args.verify:
            h_ref = sha256_of(args.verify)
            log("参考 sha256=%s" % h_ref)
            log("逐位一致：%s" % (h_out == h_ref))
    finally:
        try:
            tr = None
            T._cuda().empty_cache()
        except Exception:
            pass
        if args.restore:
            for _ in range(12):
                time.sleep(5)
                if RUN.vram() < 2000:
                    break
            RUN.restore()


if __name__ == "__main__":
    main()
