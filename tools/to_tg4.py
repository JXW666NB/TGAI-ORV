# -*- coding: utf-8 -*-
"""把 v3 家族的 `.TG` 转成 **TG4** 容器（带名字表 + 内置分词表）。

## 用途

TG4 是目前唯一"自带名字表"的容器：`NAM` 让推理端能按名取张量、让训练端能做
优化器映射/部分加载。要把产线 v3 模型迁到 TG4，就用本工具。

## 前提

源文件旁边必须有**名字边车**（`<model>.TG.names.json`）——v3 的权重表只存 FNV 哈希、
没有名字；没有边车就无法知道每个张量叫什么，也就写不出 TG4 的 NAM。

## 流式

逐张量 `load_tensor` → `write_tg4_stream`，峰值内存 = **单张量**，
所以 6 GB 的 V2.7 不需要 12 GB 内存。分词表用 `ckpt.read_tok_blob()` 从源文件原样搬。

用法：
    python to_tg4.py <src.tg> <dst.tg4> [--dtype 6]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dtype", type=int, default=-1,
                    help="写出权重类型（0=fp32 1=fp16 2=int8 6=bf16；默认沿用源文件头）")
    ap.add_argument("--every", type=int, default=200, help="每多少个张量打一行进度")
    a = ap.parse_args()

    from tgai_go import tg4
    from tgai_go.read_tg import read_tg, close_tg, load_tensor, list_tensors
    from tgai_go.train.ckpt import read_tok_blob

    if not os.path.isfile(a.src):
        raise SystemExit("源文件不存在: %s" % a.src)
    info = read_tg(a.src)
    h = info["header"]
    rows = list_tensors(info, a.src)
    if not rows:
        raise SystemExit("源文件里没有张量。修法：确认它是权重文件而不是检查点元信息")
    unnamed = [r for r in rows if not r[0]]
    if unnamed:
        raise SystemExit(
            "有 %d 个张量没有名字（v3 权重表只存哈希）。修法：先为源文件生成名字边车 "
            "`<file>.names.json`，例如用 tools/make_tg_names.py，再重跑本工具"
            % len(unnamed))

    dt = a.dtype if a.dtype >= 0 else int(h.get("weight_dtype", tg4.DT_BF16))
    head_dim = h.get("head_dim") or (int(h["d_model"]) // max(1, int(h["n_heads"])))
    cfg = dict(n_layers=int(h["n_layers"]), d_model=int(h["d_model"]),
               n_heads=int(h["n_heads"]), head_dim=int(head_dim),
               d_ff=int(h["d_ff"]), vocab=int(h["vocab"]), max_seq=int(h["max_seq"]),
               n_kv_heads=int(h.get("num_kv_heads") or 0),
               n_experts=int(h.get("n_experts") or 0),
               n_activated=int(h.get("n_activated") or 0),
               arch=int(h.get("arch") or 0), weight_dtype=int(dt),
               rope_theta=int(h.get("rope_theta") or 10000),
               pad_id=int(h.get("pad") or 0), bos_id=int(h.get("bos") or 0),
               eos_id=int(h.get("eos") or 0), unk_id=int(h.get("unk") or 0))

    tok = read_tok_blob(a.src)
    print("源: %s（%d 张量, weight_dtype=%d, 分词段 %d 字节）"
          % (a.src, len(rows), dt, len(tok or b"")))
    print("目标: %s" % a.dst)
    if len(tok or b"") == 0:
        print("  ⚠ 源文件没有分词段 —— 转出来的 TG4 也不能 encode")

    def gen():
        for i, (name, _shape, _dt, _off, _ln) in enumerate(rows):
            yield name, load_tensor(info, name)
            if a.every and (i + 1) % a.every == 0:
                print("  已写 %d/%d" % (i + 1, len(rows)))

    tg4.write_tg4_stream(a.dst, gen(), cfg, tok=tok)
    close_tg(info)

    size = os.path.getsize(a.dst)
    print("完成: %.1f MB" % (size / 1e6))
    with tg4.TG4(a.dst) as r:
        t, _m, _s = r.tokenizer()
        print("校验: 张量 %d 个, 分词表 %d 个, 首/末名字: %s / %s"
              % (len(r.names()), len(t), r.names()[0], r.names()[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
