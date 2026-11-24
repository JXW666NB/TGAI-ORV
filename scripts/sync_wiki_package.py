#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把仓库根的 `wiki/` 同步进包里（`python/tgai_go/wiki/`），让 wheel 带上词条。

为什么需要（2026-11-21 发版前发现）：`wiki/` 在**仓库根**，而 setuptools 只打包
`python/tgai_go/**` ⇒ 从 PyPI 安装的用户拿不到任何词条，`T.help` 会是空的。
但把作者入口挪进包目录又反直觉（朋友应该看到仓库顶层的 `wiki/`）。

所以：**根 `wiki/` 是唯一权威来源**，本脚本把它**逐字节复制**到 `python/tgai_go/wiki/`，
由 `tests/test_help_wiki_packaged.py` 保证两者不漂移（改了一边没同步 ⇒ CI 红）。

用法：
    python scripts/sync_wiki_package.py          # 同步
    python scripts/sync_wiki_package.py --check  # 只检查，不写（CI 用）
"""
import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "wiki")
DST = os.path.join(REPO, "python", "tgai_go", "wiki")


def entries(root):
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for fn in names:
            if not fn.lower().endswith(".txt"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            out[rel] = full
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="同步 wiki 词条进包")
    ap.add_argument("--check", action="store_true", help="只检查是否一致")
    a = ap.parse_args(argv)

    if not os.path.isdir(SRC):
        print("[wiki] 源目录不存在：%s" % SRC)
        return 2
    src = entries(SRC)
    dst = entries(DST) if os.path.isdir(DST) else {}

    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    differ = []
    for rel in sorted(set(src) & set(dst)):
        with open(src[rel], "rb") as f1, open(dst[rel], "rb") as f2:
            if f1.read() != f2.read():
                differ.append(rel)

    if a.check:
        if missing or extra or differ:
            print("[wiki] 包内副本与仓库根不一致：")
            for rel in missing:
                print("   缺：%s" % rel)
            for rel in extra:
                print("   多：%s" % rel)
            for rel in differ:
                print("   不同：%s" % rel)
            print("修法：运行 python scripts/sync_wiki_package.py")
            return 1
        print("[wiki] 一致：%d 个词条文件" % len(src))
        return 0

    for rel in missing + differ:
        target = os.path.join(DST, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(src[rel], target)
    for rel in extra:
        os.remove(os.path.join(DST, rel))
    # 清掉空的残留目录
    for dirpath, _dirs, _names in os.walk(DST, topdown=False):
        if dirpath != DST and not os.listdir(dirpath):
            os.rmdir(dirpath)
    print("[wiki] 已同步 %d 个文件（新增 %d / 更新 %d / 删除 %d）"
          % (len(src), len(missing), len(differ), len(extra)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
