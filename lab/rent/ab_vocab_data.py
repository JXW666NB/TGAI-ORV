# -*- coding: utf-8 -*-
"""判别实验：0.278 s 是「词表/模型侧」还是「数据侧」？

背景：之前的 0.215 vs 0.493 是**同时**换了 `--data` 目录和 `--vocab`，两者被绑在一起，
不能直接归因给词表。本脚本把两者拆开：

  A) data=data2048_v4096  vocab=4096      ← 原「快」的一侧
  B) data=data2048_v96000 vocab=96000     ← 原「慢」的一侧（现状）
  C) data=data2048_v4096  vocab=96000     ← ★ 只有词表变大，数据仍是 v4096
  （D) 反向的 data=v96000 + vocab=4096 跑不了：v96000 的 id 会 >= 4096，
      `corpus.check_vocab()` 会直接拒绝 —— 这正是"数据与词表必须匹配"的硬约束）

判据：
  C ≈ A (0.215) ⇒ 0.278 s 来自**数据**（分片内容/读取），不是词表；
  C ≈ B (0.493) ⇒ 0.278 s 确实来自**词表相关的模型/优化器成本**。

协议与 `bench_stable.py` 一致：预热 1 次（不计）+ 3 次取中位数，20 步。
产出：D:\\orv_bench\\res_vocab_data_ab.json
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
OUT = os.path.join(ROOT, "res_vocab_data_ab.json")
RUNS = 3


def one(data, vocab, seq=512, steps=20):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("ORV_TIE", None)
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, data), "--vocab", str(vocab),
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return {"err": "no RESULT", "tail": out.splitlines()[-3:]}
    r = json.loads(m.group(1))
    dline = next((l for l in out.splitlines() if l.startswith("[data] ")), "")
    return {"step": float(r["step_s_steady"]), "params": int(r["params"]),
            "peak_live": float(r["pool_peak_live_gib"]),
            "peak_mib": r.get("nvidia_peak_mib"),
            "data_line": dline}


def bench(tag, data, vocab, seq=512):
    print("===== %s (data=%s, vocab=%d, seq=%d) =====" % (tag, data, vocab, seq),
          flush=True)
    w = one(data, vocab, seq)
    if "err" in w:
        print("  预热失败: %s" % (w,), flush=True)
        return None
    print("  [warmup] %.3f s   %s" % (w["step"], w["data_line"]), flush=True)
    vals, last = [], w
    for i in range(RUNS):
        r = one(data, vocab, seq)
        if "err" in r:
            print("  第 %d 次失败" % (i + 1), flush=True)
            continue
        vals.append(r["step"])
        last = r
        print("  [run %d] %.3f s  params=%.2fM peak_live=%.2f GiB"
              % (i + 1, r["step"], r["params"] / 1e6, r["peak_live"]), flush=True)
    if not vals:
        return None
    med = statistics.median(vals)
    res = {"tag": tag, "data": data, "vocab": vocab, "seq": seq,
           "median": round(med, 3), "min": round(min(vals), 3),
           "max": round(max(vals), 3),
           "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 2),
           "runs": [round(v, 3) for v in vals],
           "params": last["params"], "pool_peak_live_gib": last["peak_live"],
           "nvidia_peak_mib": last["peak_mib"],
           "data_line": last["data_line"]}
    print("[%s] %s" % (tag, json.dumps(res, ensure_ascii=False)), flush=True)
    return res


def main():
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    out = {}
    out["A_v4096"] = bench("A_v4096", "data2048_v4096", 4096, seq)
    out["B_v96000"] = bench("B_v96000", "data2048_v96000", 96000, seq)
    out["C_vocab96000_on_v4096data"] = bench("C_vocab96000_on_v4096data",
                                             "data2048_v4096", 96000, seq)
    print()
    print("[ab] " + json.dumps(out, ensure_ascii=False), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[ab] wrote %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
