# -*- coding: utf-8 -*-
"""发版前检查：版本一致性 + 内核能否编译 + 融合注意力是否真的可用。

为什么需要它：2026-11-15 发布 0.9.74 时，内核里一处改动插错了函数分支，
导致 NVRTC 编译失败（`identifier "dK" is undefined`）。后果是
`sdpa_lowmem_ready()` 返回 False，**所有卡静默回退到物化注意力**（P100 前向
11.7 -> 30.3 ms），而本机 pytest 全程 788 passed —— 因为**本机没有 GPU，
pytest 根本不编译 CUDA 内核**。

所以发版前必须额外跑这一遍。在有 GPU 的机器上：

    python tools/verify_release.py

无 GPU 时会跳过内核检查并明确说明（而不是假装通过）。
"""
import io
import os
import re
import subprocess
import sys


def check_version():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = io.open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', py, re.M)
    if not m:
        return False, "pyproject.toml 里找不到 version"
    want = m.group(1)
    init = io.open(os.path.join(root, "python", "tgai_go", "__init__.py"), encoding="utf-8").read()
    lits = re.findall(r'__version__\s*=\s*"([^"]+)"', init)
    if not lits:
        return False, "tgai_go/__init__.py 里找不到 __version__ 字面量"
    if lits[-1] != want:
        return False, "版本不一致：pyproject=%s，退回字面量=%s" % (want, lits[-1])
    return True, "版本一致：%s（字面量 %d 处均为该值）" % (want, len(lits))


def check_kernel():
    """编译内核并确认融合注意力真的可用。无 GPU 时明确跳过。"""
    try:
        import tgai_go.train as T
        T.set_default_device("cuda")
    except Exception as e:  # noqa: BLE001
        return None, "无 GPU 或无法初始化 CUDA（%s），内核检查已跳过 —— 这不等于通过" % type(e).__name__
    try:
        ok = bool(T.functional.sdpa_lowmem_ready())
    except Exception as e:  # noqa: BLE001
        return False, "内核编译/加载失败：%s: %s" % (type(e).__name__, str(e)[:200])
    if not ok:
        return False, "sdpa_lowmem_ready() 返回 False —— 融合注意力不可用（会静默回退物化）"
    return True, "内核编译通过，融合注意力可用"


def main():
    print("版本检查：", flush=True)
    vok, vmsg = check_version()
    print("  [%s] %s" % ("OK" if vok else "FAIL", vmsg), flush=True)

    print("内核检查：", flush=True)
    kok, kmsg = check_kernel()
    tag = {True: "OK", False: "FAIL", None: "SKIP"}[kok]
    print("  [%s] %s" % (tag, kmsg), flush=True)

    print("全套测试：", flush=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=root, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()][-1:]
    print("  [%s] %s" % ("OK" if p.returncode == 0 else "FAIL", tail[0] if tail else "?"), flush=True)

    # ★ 内核检查 SKIP（无 GPU）**不能算通过** —— 0.9.74 就是这样发出去的：
    #   本机 788 passed，但内核编译失败，所有卡静默回退物化路径。
    #   确实要在无 GPU 机器上发版时，显式设 ALLOW_SKIP=1 并自己承担风险。
    allow_skip = os.environ.get("ALLOW_SKIP") == "1"
    bad = (not vok) or (kok is False) or (kok is None and not allow_skip) or p.returncode != 0
    if kok is None and allow_skip:
        print("\n⚠ 内核检查被跳过，且你设了 ALLOW_SKIP=1 —— 发版即自担风险。", flush=True)
    print("\n结论：%s" % ("可以发版" if not bad else "**不要发版**"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
