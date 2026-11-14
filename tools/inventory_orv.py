# -*- coding: utf-8 -*-
"""扫描 ORV 引擎，产出内核/启动器/模块清单（只读）。

用法：python tools/inventory_orv.py > docs/_inventory_raw.txt
"""
import io
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "python", "tgai_go", "train")


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def kernels(src):
    """提取 `extern "C" __global__ void NAME(` 及其上方注释的首行。"""
    out = []
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        m = re.search(r'__global__\s+void\s+(\w+)\s*\(', ln)
        if not m:
            continue
        name = m.group(1)
        note = ""
        # 向上找最近的注释块（最多 8 行）
        for j in range(i - 1, max(-1, i - 9), -1):
            s = lines[j].strip()
            if s.startswith("/*") or s.startswith("*") or s.startswith("//"):
                t = s.lstrip("/*").lstrip("*").lstrip("/").strip()
                if t and not t.startswith("--"):
                    note = t[:80]
                    break
            elif s:
                break
        out.append((name, note))
    return out


def launchers(src):
    """提取 CudaRuntime / 类里的启动器方法名与 docstring 首行。"""
    out = []
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"    def (\w+)\(self", ln)
        if not m:
            continue
        name = m.group(1)
        doc = ""
        for j in range(i + 1, min(len(lines), i + 6)):
            s = lines[j].strip()
            if s.startswith('"""'):
                doc = s.strip('"').strip()[:80]
                break
            if s and not s.startswith('"""'):
                break
        out.append((name, doc))
    return out


def main():
    cuda = read(os.path.join(ROOT, "cuda.py"))
    ks = kernels(cuda)
    ls = launchers(cuda)
    print("### 内核 %d 个" % len(ks))
    for n, d in ks:
        print("%-34s %s" % (n, d))
    print()
    print("### 启动器 %d 个" % len(ls))
    for n, d in ls:
        print("%-34s %s" % (n, d))
    print()
    print("### 模块")
    for f in sorted(os.listdir(ROOT)):
        if f.endswith(".py"):
            src = read(os.path.join(ROOT, f))
            first = ""
            for ln in src.splitlines()[:6]:
                if ln.startswith('"""'):
                    first = ln.strip('"').strip()[:90]
                    break
            print("%-22s %6d 行  %s" % (f, len(src.splitlines()), first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
