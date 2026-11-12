# -*- coding: utf-8 -*-
"""推理端冒烟：原生库能否加载 + 能否读 .TG（并分辨 v3 / v4）。

## 为什么单独写一个

外部报告说 `pip install` 后 `tgai_go.load()` 直接失败。要修这条，先得能**分别**回答：

1. 原生库本身能不能加载（PE 头是否正常）；
2. 能不能读 **v3** 的 `.TG`（现有 V2.7 产线模型都是 v3）；
3. 能不能读 **v4** 的 `.TG`（ORV 训练产出的容器）—— 这一条大概率**不行**，
   因为 C 引擎只认 v1/v3（`tgai_engine.c:2369`），正好把"训推一体"断在门口。

用法：
    python probe_infer.py                  # 只测加载
    python probe_infer.py <a.tg> [b.tg]    # 再逐个试读
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

TG_MAGIC = 0x49414754


def tg_version(path):
    """只读头 12 字节：magic(4) version(4) ..."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError as exc:
        return None, "读取失败: %s" % exc
    if len(head) < 8:
        return None, "文件太短"
    magic, ver = struct.unpack_from("<II", head, 0)
    if magic != TG_MAGIC:
        return None, "magic=%#x 不是 TGAI" % magic
    return ver, "version=%d" % ver


def main():
    import tgai_go
    print("tgai_go:", tgai_go.__version__)
    print("native_available:", tgai_go.native_available())
    print("native_dir:", tgai_go.native_dir())
    files = sorted(os.listdir(tgai_go.native_dir())) if tgai_go.native_available() else []
    print("原生库:", files)

    # ① 单独试 CDLL，把"库坏"和"库好但模型不兼容"分开
    import ctypes
    for f in files:
        if not f.endswith(".dll"):
            continue
        try:
            ctypes.WinDLL(os.path.join(tgai_go.native_dir(), f))
            print("  CDLL OK  %s" % f)
        except OSError as exc:
            print("  CDLL 失败 %s : %s" % (f, exc))

    if len(sys.argv) < 2:
        print("\n（没有给 .TG 路径，略过模型加载）")
        return 0

    rc = 0
    for path in sys.argv[1:]:
        ver, note = tg_version(path)
        print("\n=== %s" % path)
        print("  头: %s" % note)
        try:
            m = tgai_go.load(path)
        except Exception as exc:                 # noqa: BLE001
            print("  加载失败 %s: %s" % (type(exc).__name__, str(exc)[:260]))
            if ver == 4:
                print("  判读：这是 v4 容器，C 引擎只认 v1/v3 ⇒ 属于**已知不支持**，"
                      "不是库坏了")
            else:
                rc = 1
            continue
        print("  加载成功，模型句柄: %r" % (m,))
    return rc


if __name__ == "__main__":
    sys.exit(main())
