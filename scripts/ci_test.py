#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跑 pytest，把失败用例名写进 GitHub Job Summary —— **跨平台，不依赖 shell 语法**。

为什么要有它（2026-11-21 三次踩坑）：
CI 只在某一个 runner 上偶发失败时，公开仓库的 actions logs API 也返 403，
于是"到底是哪条测试红的"只能靠人截图 —— 太浪费时间。
这个脚本把 `FAILED/ERROR` 行和末几行汇总写进 `$GITHUB_STEP_SUMMARY`
（GitHub 会渲染在 run 页面上），并以 pytest 的返回码退出。

用法：
    python scripts/ci_test.py                 # 跑全套
    python scripts/ci_test.py tests/test_x.py  # 跑指定
"""
import os
import re
import subprocess
import sys

LOG = os.path.join(os.getcwd(), "pytest-ci.log")
FAIL_RE = re.compile(r"^(FAILED|ERROR)\s+\S+")


def main(argv):
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "--tb=short", "-rf"]
    cmd += argv[1:] if len(argv) > 1 else ["tests"]
    print("[ci] %s" % " ".join(cmd), flush=True)

    lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    rc = proc.wait()

    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except OSError:
        pass

    fails = [ln.strip() for ln in lines if FAIL_RE.match(ln.strip())]
    tail = [ln.rstrip() for ln in lines if ln.strip()][-3:]

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    body = ["## pytest 结果", ""]
    body.append("返回码：`%d`" % rc)
    if fails:
        body.append("")
        body.append("### 失败用例（%d）" % len(fails))
        body.append("```")
        body.extend(fails[:40])
        body.append("```")
    body.append("")
    body.append("### 末尾输出")
    body.append("```")
    body.extend(tail)
    body.append("```")
    text = "\n".join(body) + "\n"

    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
    # 本地跑时也打出来，省得再去翻日志文件
    if fails and not summary:
        print("\n[ci] 失败用例：")
        for f_ in fails[:20]:
            print("   %s" % f_)
    print("[ci] 返回码 %d（日志：%s）" % (rc, LOG))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
