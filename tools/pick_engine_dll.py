# -*- coding: utf-8 -*-
"""挑出**能真正加载**的推理引擎 DLL。

## 为什么要这个

外部报告指出随包的 `libtgai_engine.dll` 的 PE 头错位 1 字节：
DOS 头 `e_lfanew` 指向 `0xF8`，而真正的 `PE\\0\\0` 在 `0xF7`。
`ctypes.CDLL` 于是报 `WinError 193`。仓库里有 30 多个历史副本，
必须先程序化筛出好的那个，不能靠"看起来像"。

## 判据（三重，缺一不可）

1. `e_lfanew` 处的 4 字节等于 `PE\\0\\0`；
2. `COFF` 头的 Machine == `0x8664`（x64）；
3. **`ctypes.CDLL(路径)` 真的成功**（前两条对了也可能缺依赖）。

用法：`python pick_engine_dll.py [更多目录...]`
"""
import ctypes
import glob
import os
import struct
import sys

#: 候选目录（可再加命令行参数）
DIRS = [
    r"D:\tgai_nlp\TGAI GO",
    r"D:\tgai_nlp\TGAI Studio\dist\TGAI Studio\_internal\engine",
    r"D:\tgai_nlp\experiment_ollama\tgai-go-fix",
    r"D:\DSH-Workspace\TGAI-对话档案\_repos\TGAI-GO",
    r"D:\DSH-Workspace\TGAI-对话档案\_tgai_go_real",
    r"D:\DSH-Workspace\TGAI-对话档案\_clean_check",
    r"D:\DSH-Workspace\TGAI-对话档案\_a10_stage\tgai",
    r"D:\DSH-Workspace\TGAI-对话档案\_stage_tgaigo",
    r"D:\DSH-Workspace\TGAI-对话档案\_tgai_orv",
]


def pe_report(path):
    """返回 (ok, 说明)。只读前 4 KB。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError as exc:
        return False, "读取失败: %s" % exc
    if len(head) < 64 or head[:2] != b"MZ":
        return False, "不是 MZ 文件"
    e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
    if e_lfanew + 24 > len(head):
        return False, "e_lfanew=%#x 超出已读范围" % e_lfanew
    sig = head[e_lfanew:e_lfanew + 4]
    if sig != b"PE\0\0":
        # 常见病：整体错位 1 字节
        near = [d for d in (-2, -1, 1, 2) if head[e_lfanew + d:e_lfanew + d + 4] == b"PE\0\0"]
        return False, "e_lfanew=%#x 处不是 PE 签名（偏移 %s 处才是）" % (e_lfanew, near or "无")
    machine = struct.unpack_from("<H", head, e_lfanew + 4)[0]
    if machine != 0x8664:
        return False, "Machine=%#x 不是 x64(0x8664)" % machine
    try:
        ctypes.WinDLL(path)          # noqa: F841  —— 能加载就足够
    except OSError as exc:
        return False, "PE 头正常但加载失败: %s" % exc
    return True, "OK（PE 正常且可加载）"


def main():
    dirs = DIRS + sys.argv[1:]
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for pat in ("**/libtgai_engine*.dll", "**/libtgai_engine*.so"):
            paths += glob.glob(os.path.join(d, pat), recursive=True)
    paths = sorted(set(paths))

    good, bad = [], []
    for p in paths:
        ok, why = pe_report(p)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        mtime = os.path.getmtime(p) if os.path.exists(p) else 0
        row = (size, mtime, p, why)
        (good if ok else bad).append(row)

    print("=" * 100)
    print("可用（%d 个），按修改时间倒序：" % len(good))
    for size, mtime, p, why in sorted(good, key=lambda r: -r[1]):
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print("  %6.2f MB  %s  %s" % (size / 1e6, ts, p))

    print()
    print("不可用（%d 个）：" % len(bad))
    for size, mtime, p, why in sorted(bad, key=lambda r: -r[1])[:12]:
        print("  %6.2f MB  %s" % (size / 1e6, p))
        print("              %s" % why)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
