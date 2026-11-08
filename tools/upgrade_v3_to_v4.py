# -*- coding: utf-8 -*-
"""把 TGAI 容器（magic `TGAI`）的 v3 文件升级为 v4 —— **只改 version 字段**。

为什么可以这么简单（实测依据）：
* v4 与 v3 是**同一个容器**（128B 头 + 分词段 + 52B 权重索引 + WGT + 尾部扩展段），
  差别只有 `version` 字段（以及允许 BF16/Q4K 权重类型，生产文件本来就用 BF16）；
* 实测把 `TGAI_NB_V2.7.tg` 的 version 3→4 后：
  `TOK段`、`权重索引`、`WGT段(6.06 GB)` **全部逐位一致**，头部仅 1 字节不同。

用法：
    python upgrade_v3_to_v4.py --src <v3.tg> --dst <v4.tg> [--verify]

安全约定：
* 默认**不覆盖**已存在的 dst；
* 不修改 src（只读）；
* `--verify` 会对三段（TOK / 权重索引 / WGT）分别算 sha256 并比对。
"""
import argparse
import hashlib
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        b = f.read(128)
    magic = b[:4]
    if magic != b"TGAI":
        raise SystemExit("不是 TGAI 容器（magic=%r）。修法：本工具只处理 magic 'TGAI' 的文件" % magic)
    (version,) = struct.unpack_from("<I", b, 4)
    tok_off, tok_len, wt_off, wt_cnt, wd_off, wd_len = struct.unpack_from("<6Q", b, 64)
    return dict(magic=magic, version=version, tok_off=tok_off, tok_len=tok_len,
                wt_off=wt_off, wt_cnt=wt_cnt, wd_off=wd_off, wd_len=wd_len)


def sha_region(path, off, ln):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = ln
        while left > 0:
            b = f.read(min(1 << 22, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v3 文件（只读）")
    ap.add_argument("--dst", required=True, help="输出的 v4 文件（已存在则拒绝）")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("src 与 dst 相同，拒绝原地改写（请另存）")
    if os.path.exists(args.dst) and not args.overwrite:
        raise SystemExit("dst 已存在：%s（加 --overwrite 才覆盖）" % args.dst)

    h = read_header(args.src)
    if h["version"] == 4:
        print("源文件已是 v4，无需升级")
        return 0
    if h["version"] != 3:
        raise SystemExit("源版本为 %d，本工具只处理 v3" % h["version"])

    with open(args.src, "rb") as fi, open(args.dst, "wb") as fo:
        hdr = bytearray(fi.read(128))
        fo.write(hdr)
        while True:
            b = fi.read(1 << 22)
            if not b:
                break
            fo.write(b)
    with open(args.dst, "r+b") as f:
        f.seek(0)
        hdr = bytearray(f.read(128))
        struct.pack_into("<I", hdr, 4, 4)
        f.seek(0)
        f.write(hdr)

    print("已升级：%s → %s" % (os.path.basename(args.src), os.path.basename(args.dst)))
    print("大小 原=%d 新=%d" % (os.path.getsize(args.src), os.path.getsize(args.dst)))
    h2 = read_header(args.dst)
    print("版本 新=%d；wd_off/wd_len 一致=%s"
          % (h2["version"], (h["wd_off"], h["wd_len"]) == (h2["wd_off"], h2["wd_len"])))

    if args.verify:
        diffs = sum(1 for x, y in zip(open(args.src, "rb").read(128), open(args.dst, "rb").read(128)) if x != y)
        print("头部差异字节数=%d（应为 1）" % diffs)
        for key, name in (("tok_off", "TOK段"), ("wt_off", "权重索引"), ("wd_off", "WGT段")):
            ln = h["tok_len"] if key == "tok_off" else (h["wd_off"] - h["wt_off"] if key == "wt_off" else h["wd_len"])
            a = sha_region(args.src, h[key], ln)
            b = sha_region(args.dst, h2[key], ln)
            print("%s 逐位一致=%s (%s)" % (name, a == b, a[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
