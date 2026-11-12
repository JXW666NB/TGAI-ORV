# -*- coding: utf-8 -*-
"""扫描远端磁盘上的 .TG，打印每个文件的 magic / version / 大小。

目的：在 V100 上找出**已经是 V4** 的 V2.7 权重（用户说已升级到 V4），
作为"引擎读不了 v4"的第一个实证对象。

用法：`python scan_tg_versions.py D:\\ `（默认扫 D:\\，可给多个根）
"""
import os
import struct
import sys

TG_MAGIC = 0x49414754


def head_info(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError as exc:
        return None, "读取失败: %s" % exc
    if len(head) < 8:
        return None, "太短"
    magic, ver = struct.unpack_from("<II", head, 0)
    if magic != TG_MAGIC:
        return None, "magic=%#x" % magic
    return ver, "ok"


def main():
    roots = sys.argv[1:] or ["D:\\"]
    rows = []
    seen = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.lower().endswith(".tg"):
                    continue
                seen += 1
                p = os.path.join(dirpath, name)
                ver, why = head_info(p)
                if ver is None:
                    continue
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                rows.append((ver, size, p))

    rows.sort(key=lambda r: (r[0], -r[1]))
    print("扫到 %d 个 .tg，可解析头 %d 个" % (seen, len(rows)))
    print()
    for ver in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == ver]
        print("=== version=%d（%d 个，按大小倒序取前 12）===" % (ver, len(sub)))
        for v, size, p in sub[:12]:
            print("  %8.1f MB  %s" % (size / 1e6, p))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
