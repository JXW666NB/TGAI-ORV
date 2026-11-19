# -*- coding: utf-8 -*-
"""定案：seq512 的 0.239 s 到底来自哪个词表？

同一台机、同一协议（预热 1 次 + 3 次中位数），只换数据/词表：
  A) data2048        + vocab 4096   ← split_orv.py 注释示例用的就是这个
  B) data2048_v96000 + vocab 96000  ← 现在一直用的
其余全部相同：12 层 / d=1024 / d_ff=3328 / E=4 / n_q=16 / top_k=2 / csa=4 /
             opt int8gpu / amp gemm / batch 1 / 20 步
"""
import json
import os
import re
import statistics
import subprocess

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"


def one(data, vocab, seq=512, steps=20):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, data), "--vocab", str(vocab),
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    if not m:
        tail = " | ".join(out.splitlines()[-2:])
        return None, tail
    return float(m.group(1)), None


def bench(tag, data, vocab):
    print("===== %s（%s, vocab=%d）=====" % (tag, data, vocab), flush=True)
    r, err = one(data, vocab)
    if err:
        print("  预热失败: %s" % err)
        return None
    print("  [warmup] %.3f s" % r, flush=True)
    vals = []
    for i in range(3):
        v, err = one(data, vocab)
        if err:
            continue
        vals.append(v)
        print("  [run %d] %.3f s" % (i + 1, v), flush=True)
    if not vals:
        return None
    med = statistics.median(vals)
    print("[%s] %s" % (tag, json.dumps({"median": round(med, 3),
                                        "min": round(min(vals), 3),
                                        "max": round(max(vals), 3)})), flush=True)
    return med


a = bench("A_vocab4096", "data2048_v4096", 4096)
b = bench("B_vocab96000", "data2048_v96000", 96000)
print()
print("[结论] " + json.dumps({"v4096": a, "v96000": b,
                             "ratio": round(b / a, 2) if (a and b) else None}))
