# -*- coding: utf-8 -*-
"""在 GPU 机器上验证 `test_moe_grouped_fallback.py` 的断言。

该机器**没有 pytest**（已确认：No module named pytest），所以这里直接 import
测试模块并逐个调用 `test_*` 函数，把断言结果写文件。跑的是**同一批断言**，
只是去掉了 pytest 的外壳。

用法：python run_grouped_tests.py
"""
import io
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tgf.txt")
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))
sys.path.insert(0, HERE)


def main():
    lines = []
    rc = 0
    try:
        import test_moe_grouped_fallback as M
    except Exception:
        lines.append("导入失败:\n" + traceback.format_exc())
        _write(lines, 1)
        return 1

    names = [n for n in dir(M) if n.startswith("test_")]
    names.sort()
    for n in names:
        fn = getattr(M, n)
        try:
            fn()
            lines.append("PASS  %s" % n)
        except Exception as exc:
            kind = type(exc).__name__
            if kind == "Skipped" or "Skip" in kind:
                lines.append("SKIP  %s（%s）" % (n, str(exc)[:80]))
                continue
            rc = 1
            lines.append("FAIL  %s -> %s: %s" % (n, kind, str(exc)[:200]))
            lines.append(traceback.format_exc())
    _write(lines, rc)
    print("rc=%d -> %s" % (rc, OUT))
    return rc


def _write(lines, rc):
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % rc)
        f.write("\n".join(lines))
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
