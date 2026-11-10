# -*- coding: utf-8 -*-
"""诊断两个不稳定测试：是否由 PYTHONHASHSEED 引起。

用法：python tools/flaky_probe.py
      分别以固定 seed 与随机 seed 各跑 6 次，统计通过率。
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    "tests/test_clip_accum.py::test_accumulation_sum_mode_matches_torch_sum_reduction",
    "tests/test_muon.py::test_step_layers_matches_step_within_fp32_noise",
]


def run_once(seed):
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run([sys.executable, "-m", "pytest"] + TARGETS + ["-q", "--tb=no"],
                       cwd=HERE, capture_output=True, text=True, env=env)
    tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l]
    line = tail[-1] if tail else "(无结果)"
    bad = "failed" in line
    names = re.findall(r"FAILED (\S+)", r.stdout) or re.findall(r"FAILED (\S+)", r.stderr)
    return bad, line.strip(), names


def main():
    for label, seed in (("PYTHONHASHSEED=0", 0), ("默认（随机哈希）", None)):
        bad = 0
        names = {}
        for i in range(6):
            b, line, ns = run_once(seed)
            if b:
                bad += 1
                for n in ns:
                    names[n] = names.get(n, 0) + 1
        print("%-20s 6 次里失败 %d 次%s"
              % (label, bad, "" if not names else "；失败项：" + str(names)))


if __name__ == "__main__":
    main()
