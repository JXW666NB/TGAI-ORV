# -*- coding: utf-8 -*-
"""稠密基准的分段耗时剖析（在能看完整 stdout 的机器上跑，例如 V100）。

做两件事：
  1. 复制 bench/external/bench_p100_full.py -> <tmp>/bench_prof.py
  2. 在**ORV 那个 blk**（用 Fn.scaled_dot_product_attention 作锚点）里插计时，
     每段后强制 rt.sync()，跳过前 42 块（首次编译 + 预热），跑满后打印四段均值。

为什么不用文本整段匹配：该文件里有**两个** blk（torch 版用 F，ORV 版用 Fn），
整段匹配会命中 torch 那个（已踩过三次）。这里按行号定位。

用法：python lab/v3exp2_modules/prof_dense.py [HB] [POL]
"""
import io
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "bench", "external", "bench_p100_full.py")

HEAD = [
    "        import time as _t",
    '        _s = globals().setdefault("_SEG", {"p": 0.0, "a": 0.0, "t": 0.0, "n": 0})',
    "        _RT = T._cuda()",
    "        _t0 = _t.time()",
]
MID = [
    "        _RT.sync(); _t1 = _t.time()",
    "        if _s['n'] > 42: _s['p'] += _t1 - _t0",
]
TAIL = [
    "        _RT.sync(); _t2 = _t.time()",
    "        if _s['n'] > 42:",
    "            _s['a'] += _t2 - _t1",
    "            _s['t'] += _t2 - _t0",
    "        _s['n'] += 1",
    "        if _s['n'] == 84:",
    '            print("SEG p=%.1f a=%.1f rest=%.1f ms/block" % (_s["p"]/42*1000, _s["a"]/42*1000, (_s["t"]-_s["p"]-_s["a"])/42*1000), flush=True)',
]


def build(dst):
    L = io.open(SRC, encoding="utf-8").read().splitlines()
    att = [i for i, l in enumerate(L) if "Fn.scaled_dot_product_attention" in l]
    if len(att) != 1:
        raise SystemExit("ORV 的 attention 行应唯一，实际 %d 处" % len(att))
    i_att = att[0]
    i_blk = [i for i in range(i_att, -1, -1) if L[i].lstrip().startswith("def blk")][0]
    end = [i for i in range(i_att, len(L))
           if L[i].startswith("        return ") or L[i].startswith("    def ")][0]
    out = L[:i_blk + 1] + HEAD + L[i_blk + 1:i_att] + MID + L[i_att:end] + TAIL + L[end:]
    io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    return i_blk + 1, i_att + 1, end + 1


def main():
    hb = sys.argv[1] if len(sys.argv) > 1 else "4"
    pol = sys.argv[2] if len(sys.argv) > 2 else "speed"
    tmp = tempfile.mkdtemp(prefix="orvprof_")
    dst = os.path.join(tmp, "bench_prof.py")
    b, a, e = build(dst)
    print("patched blk@%d att@%d end@%d -> %s" % (b, a, e, dst), flush=True)
    env = dict(os.environ)
    env["HB"] = hb
    env["POL"] = pol
    cmd = [sys.executable, "-u", dst, "orv"]
    print("run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, cwd=tmp)


if __name__ == "__main__":
    main()
