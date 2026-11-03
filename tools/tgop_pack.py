# -*- coding: utf-8 -*-
"""把目录打包成 `.tgop` 通用算子插件包（一个文件装下所有厂商的实现）。

用法：
    python tools/tgop_pack.py myplugin/ -o gelu.tgop
    python tools/tgop_pack.py myplugin/            # 输出 myplugin.tgop

目录里必须有 `manifest.json`（格式见 `python/tgai_go/train/ops_bundle.py` 顶部注释）。
打包时会做基本校验：schema、name、arity、python 兜底档、以及每个后端引用的文件是否存在。
"""
import argparse
import json
import os
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.ops_bundle import SCHEMA, REQUIRED_BACKEND  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("srcdir", help="插件目录（含 manifest.json）")
    ap.add_argument("-o", "--out", default=None, help="输出 .tgop 路径")
    a = ap.parse_args(argv)

    src = os.path.abspath(a.srcdir)
    if not os.path.isdir(src):
        print("目录不存在：%s" % src)
        return 2
    mpath = os.path.join(src, "manifest.json")
    if not os.path.isfile(mpath):
        print("缺少 %s。修法：先写 manifest.json（格式见 ops_bundle.py 顶部注释）" % mpath)
        return 2
    with open(mpath, "r", encoding="utf-8-sig") as f:
        m = json.load(f)

    errs = []
    if int(m.get("schema", 0)) != SCHEMA:
        errs.append("manifest.schema 必须是 %d，收到 %r" % (SCHEMA, m.get("schema")))
    if not m.get("name"):
        errs.append("manifest.name 不能为空")
    if int(m.get("arity", 0)) not in (1, 2):
        errs.append("manifest.arity 必须是 1 或 2")
    bk = m.get("backends") or {}
    if REQUIRED_BACKEND not in bk:
        errs.append("缺少 python 兜底档（没有它的机器装了跑不了）")
    ref_files = set()
    for b, spec in bk.items():
        for key in ("source", "c", "spirv", "module"):
            if isinstance(spec, dict) and spec.get(key):
                ref_files.add(spec[key])
    for rel in sorted(ref_files):
        if not os.path.isfile(os.path.join(src, rel)):
            errs.append("manifest 引用的文件不存在：%s" % rel)
    if errs:
        print("打包失败，共 %d 个问题：" % len(errs))
        for e in errs:
            print("  - " + e)
        return 1

    out = a.out or (os.path.basename(src.rstrip("/\\")) + ".tgop")
    out = os.path.abspath(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(mpath, "manifest.json")
        for rel in sorted(ref_files):
            zf.write(os.path.join(src, rel), rel)
    size = os.path.getsize(out)
    print("已生成 %s（%.1f KB，后端：%s）" % (out, size / 1024.0, ", ".join(sorted(bk))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
