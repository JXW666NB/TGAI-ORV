# -*- coding: utf-8 -*-
"""归因矩阵驱动：把 `attr_vocab_event.py` 在 4 个口径上各跑一遍。

口径（seq512，12 步，跳过前 4 步；其余与 bench_stable.py 相同）：
  1) v96000_untie : data=v96000 vocab=96000         ← 现状
  2) v96000_tie   : data=v96000 vocab=96000 --tie   ← 假设检验
  3) v96000model  : data=v4096  vocab=96000         ← 只有"模型侧词表"变大（数据不变）
  4) v4096        : data=v4096  vocab=4096          ← 原「快」的一侧
  5) cast96000 / 6) cast4096 : 同上 (1)/(4)，但 ATTR_CAST=1（逐次给 cast kernel 打点）

产出：D:\\orv_bench\\res_attr_all.json（原始 attr json 也各存一份 attr_*.json）
"""
import json
import os
import re
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
SCRIPT = os.path.join(ROOT, "attr_vocab_event.py")
OUT = os.path.join(ROOT, "res_attr_all.json")
STEPS = 12

MATRIX = [
    # tag,              data,          vocab, tie, seq, cast
    ("v96000_untie", "data2048_v96000", 96000, False, 512, False),
    ("v96000_tie", "data2048_v96000", 96000, True, 512, False),
    ("v96000model", "data2048_v4096", 96000, False, 512, False),
    ("v4096", "data2048_v4096", 4096, False, 512, False),
    ("cast96000", "data2048_v96000", 96000, False, 512, True),
    ("cast4096", "data2048_v4096", 4096, False, 512, True),
]


def one(tag, data, vocab, tie, seq, cast):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["ATTR_TAG"] = tag
    env["ATTR_OUT"] = os.path.join(ROOT, "attr_%s.json" % tag)
    env["ATTR_SKIP"] = "4"
    env["ATTR_CAST"] = "1" if cast else "0"
    if tie:
        env["ORV_TIE"] = "1"
    else:
        env.pop("ORV_TIE", None)
    cmd = [PY, SCRIPT,
           "--data", os.path.join(ROOT, data), "--vocab", str(vocab),
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^\[attr\] (\{.*\})$", out, re.M)
    if not m:
        print(out[-2000:], flush=True)
        return {"err": "no [attr]", "tail": out.splitlines()[-4:]}
    r = json.loads(m.group(1))
    r["tag"] = tag
    r["data"] = data
    return r


def main():
    out = {}
    for tag, data, vocab, tie, seq, cast in MATRIX:
        print("===== attr %s (data=%s vocab=%d tie=%s cast=%s) ====="
              % (tag, data, vocab, tie, cast), flush=True)
        r = one(tag, data, vocab, tie, seq, cast)
        out[tag] = r
        if "err" in r:
            print("  失败: %s" % (r,), flush=True)
            continue
        s = r["sections"]
        print("  wall=%.3f s  measured=%d  cast_kernels/step=%.0f"
              % (r["step_s_wall_mean"] or 0, r["measured_steps"],
                 r.get("cast_kernels_per_step", 0)), flush=True)
        for k in sorted(s):
            print("    %-10s %8.3f ms/step" % (k, s[k]["ms_per_step"]), flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print()
    print("[attr_all] wrote %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
