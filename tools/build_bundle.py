# -*- coding: utf-8 -*-
"""把插件包目录打成 `.tgop`（就是一个 zip，改后缀），并校验元信息。

用法：
    python tools/build_bundle.py examples/bundles/svs-ops
    python tools/build_bundle.py examples/bundles/svs-ops -o dist

产物：`<输出目录>/<name>-<version>.tgop`，与包目录内容一一对应。
打包前会先跑 `plugin_meta.validate`：**元信息不合法就不出包**。
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from tgai_go.train import plugin_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir")
    ap.add_argument("-o", "--out", default="dist")
    args = ap.parse_args()

    root = os.path.abspath(args.bundle_dir)
    mpath = os.path.join(root, "manifest.json")
    if not os.path.exists(mpath):
        print("找不到 manifest.json：%s" % mpath)
        return 2
    meta = plugin_meta.load_manifest_file(mpath)
    print("元信息通过：%s" % plugin_meta.summary_line(meta))

    # 校验 impl 里声明的文件真的在包内
    missing = []
    for op in meta["ops"]:
        m = op.get("impl", {}).get("module")
        if m and not os.path.exists(os.path.join(root, m)):
            missing.append((op["name"], m))
    if missing:
        for nm, m in missing:
            print("算子 %s 的实现文件缺失：%s" % (nm, m))
        return 3

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "%s-%s.tgop" % (meta["name"], meta["version"]))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                z.write(full, rel)
    size = os.path.getsize(out)
    print("已打包：%s（%d 个文件，%d 字节）" % (out, len(zipfile.ZipFile(out).namelist()), size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
