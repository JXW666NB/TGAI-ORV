# -*- coding: utf-8 -*-
"""改写 `.TG` 头里的版本号（v3 ↔ v4），不改动任何其它字节。

## 为什么这样是安全的

2026-11-12 实测：V2.7 的 `TGAI_NB_V2.7_v4.tg` 与 ORV 训练检查点 `weights.tg` 都是
`magic=0x49414754`（TGAI）+ `version=4`，**布局与 v3 完全一致**（定长 128B 头 + FNV 权重表），
Python 侧 `read_tg()` 能直接解析出 layers/d_model 并读出张量表。
也就是说 v3→v4 在本仓库的语境里**只是头里那 4 个字节的版本号**。

⚠ 这与 `tg4.py` 里的「TG4」容器**不是一回事**：那是另一套格式
（magic `0x00344754` + section table + NAM 名字表），本工具拒绝处理。

用法：
    python set_tg_version.py in.tg out.tg --version 4
    python set_tg_version.py in.tg --version 4 --inplace    # 原地改（会先备份）
"""
import argparse
import io
import os
import shutil
import struct
import sys

TGAI_MAGIC = 0x49414754
TG4_MAGIC = 0x00344754
VERSION_OFF = 4


def read_head(path):
    with open(path, "rb") as fh:
        head = fh.read(128)
    if len(head) < 128:
        raise SystemExit("文件太短（%d 字节），不是 .TG：%s" % (len(head), path))
    magic, ver = struct.unpack_from("<II", head, 0)
    return magic, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--version", type=int, required=True, help="目标版本号（1/3/4）")
    ap.add_argument("--inplace", action="store_true", help="原地改写（先备份 .bak）")
    a = ap.parse_args()

    magic, ver = read_head(a.src)
    if magic == TG4_MAGIC:
        raise SystemExit("这是 TG4 新容器（section table），不能用本工具改版本号。"
                         "修法：TG4 与 v3 家族布局不同，需用 tg4.py 读写")
    if magic != TGAI_MAGIC:
        raise SystemExit("magic=%#x 不是 TGAI（0x%X）。修法：确认文件真的是 .TG"
                         % (magic, TGAI_MAGIC))

    dst = a.src if a.inplace else (a.dst or (a.src + ".v%d" % a.version))
    if a.inplace:
        bak = dst + ".bak"
        shutil.copy2(dst, bak)
        print("已备份:", bak)

    with open(a.src, "rb") as fh:
        data = bytearray(fh.read())
    struct.pack_into("<I", data, VERSION_OFF, a.version)
    with open(dst, "wb") as fh:
        fh.write(data)

    m2, v2 = read_head(dst)
    print("写入: %s" % dst)
    print("版本: %d -> %d（magic 不变: %#x）" % (ver, v2, m2))
    return 0 if v2 == a.version else 1


if __name__ == "__main__":
    sys.exit(main())
