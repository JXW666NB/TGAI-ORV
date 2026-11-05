# -*- coding: utf-8 -*-
"""一条命令跑完本地全部门禁（不需要 GPU）。

覆盖：
1. 单元测试（`pytest tests`）—— 形状、错误信息、存档往返、Qwen2.5/Qwen3 前向对拍
2. 打包自检（`tools/check_package.py --build`）—— 文件齐 / 语法净 / 干净 venv 可 import / `.cu` 随包
3. LLaMA 结构对拍（`examples/llama_tiny.py --preset tiny`）—— 与 PyTorch 孪生逐步对齐
4. Qwen3 真权重切片对拍（`tools/qwen3_tg_parity.py`）—— 只在本地有 `Qwen3-fix.TG` 时跑

为什么要有这个脚本：
    判据写进文档只是一半，**能一条命令重跑**才是可执行的标准。
    发布前、改完引擎后、怀疑回归时，先跑它，再谈别的。

用法：
    python tools/verify_all.py            # 全跑
    python tools/verify_all.py --quick    # 跳过打包（省 ~30 秒）
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _run(name, cmd, timeout=1800):
    t0 = time.time()
    print("\n=== %s ===" % name)
    print("$ " + " ".join(cmd))
    skipped_local = False
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        tail = "\n".join([ln for ln in out.strip().split("\n") if ln.strip()][-12:])
        print(tail)
        # 77 = 约定“环境不满足”（如主机内存不足）→ SKIP，**不算失败**
        if p.returncode == 77:
            print("[skip] 环境不满足，本项跳过（不计入失败）")
            ok = True
            skipped_local = True
        else:
            ok = p.returncode == 0
    except subprocess.TimeoutExpired:
        print("超时（%ds）" % timeout)
        ok = False
    except Exception as ex:                        # noqa: BLE001
        print("异常：%s" % (ex,))
        ok = False
    return ok, time.time() - t0, skipped_local


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过打包自检")
    a = ap.parse_args(argv)

    py = sys.executable
    results = []

    results.append(("单元测试",) + _run(
        "单元测试（纯 CPU）", [py, "-m", "pytest", "tests", "-q"]))

    if not a.quick:
        results.append(("打包自检",) + _run(
            "打包 + 发布前自检", [py, "-X", "utf8",
                                  os.path.join("tools", "check_package.py"), "--build"]))

    results.append(("LLaMA 结构对拍",) + _run(
        "LLaMA 结构 vs PyTorch 孪生", [py, "-X", "utf8",
                                     os.path.join("examples", "llama_tiny.py"),
                                     "--preset", "tiny", "--steps", "8"]))

    tg = r"D:\DSH-Workspace\_gguf_test\Qwen3-fix.TG"
    if os.path.isfile(tg):
        results.append(("Qwen3 真权重对拍",) + _run(
            "Qwen3 真权重切片 vs 官方 HF", [py, "-X", "utf8",
                                          os.path.join("tools", "qwen3_tg_parity.py"),
                                          "--tg", tg]))
    else:
        print("\n（跳过 Qwen3 真权重对拍：找不到 %s）" % tg)
        results.append(("Qwen3 真权重对拍", True, 0.0, True))

    print("\n" + "=" * 60)
    print("%-22s %-6s %8s" % ("门禁", "结果", "耗时"))
    print("-" * 60)
    all_ok = True
    for row in results:
        name, ok, dt = row[0], row[1], row[2]
        skipped = row[3] if len(row) > 3 else False
        status = "SKIP" if (ok and skipped) else ("PASS" if ok else "FAIL")
        print("%-22s %-6s %7.1fs" % (name, status, dt))
        all_ok = all_ok and ok
    print("=" * 60)
    print("总判定：%s" % ("PASS —— 可以发布" if all_ok else "FAIL —— 先修再发"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
