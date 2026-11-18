# -*- coding: utf-8 -*-
"""vocab A/B：定位 seq2048 时"其余"部分里 head/loss 占多少。

设计（2×2）：
  seq ∈ {512, 2048}  ×  vocab ∈ {4096, 96000}
  同一 seq 下把 vocab 从 96000 降到 4096，减少的是：
    (a) logits/CE 的 T×vocab 计算与读写   ← 正比于 T
    (b) embedding/head 的参数与优化器更新  ← 不随 T 变
  两个 seq 的差值相减，就能把 (a) 单独分离出来。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
PY = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 3
LAYERS = 12


def _data_for(vocab):
    """小词表必须配小 id 的数据，否则 token id 越界。"""
    d = os.path.join(ROOT, "data2048_v%d" % vocab)
    if not os.path.isdir(d):
        subprocess.run([PY, os.path.join(ROOT, "make_packed.py"), "--out", d,
                        "--seq", "2048", "--vocab", str(vocab), "--shards", "8",
                        "--rows", "16"], cwd=ROOT)
    return d


def one(seq, vocab):
    cmd = [PY, os.path.join(ROOT, "split_orv.py"),
           "--data", _data_for(vocab), "--vocab", str(vocab),
           "--seq", str(seq), "--d", "1024", "--layers", str(LAYERS),
           "--E", "4", "--n_q", "16", "--d_ff", "3328", "--top_k", "2",
           "--csa", "4", "--opt", "int8gpu", "--steps", str(STEPS),
           "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"\[split\] (\{.*\})", out)
    if not r:
        return None
    res = json.loads(r.group(1))
    attn = json.loads(m.group(1))["attn_s_total"] / float(STEPS) if m else float("nan")
    return {"step": res["step_s_steady"], "attn": attn,
            "rest": res["step_s_steady"] - attn, "params": res["params"],
            "mem": res["pool_peak_live_gib"]}


rows = {}
for seq in (512, 2048):
    for vocab in (4096, 96000):
        d = one(seq, vocab)
        rows[(seq, vocab)] = d
        if d is None:
            print("seq%-5d vocab%-6d FAIL" % (seq, vocab))
        else:
            print("seq%-5d vocab%-6d step=%6.3f s  attn=%6.3f  rest=%6.3f  "
                  "params=%.1fM  mem=%.1f GiB"
                  % (seq, vocab, d["step"], d["attn"], d["rest"],
                     d["params"] / 1e6, d["mem"]), flush=True)

print()
d512 = rows.get((512, 96000)), rows.get((512, 4096))
d2048 = rows.get((2048, 96000)), rows.get((2048, 4096))
if all(d512) and all(d2048):
    big512 = d512[0]["step"] - d512[1]["step"]
    big2048 = d2048[0]["step"] - d2048[1]["step"]
    print("vocab 96000→4096 省下的步时：seq512 %.3f s，seq2048 %.3f s" % (big512, big2048))
    print("差值之差（正比于 T 的那部分，即 head/loss）：%.3f s" % (big2048 - big512))
    print("其中 seq2048 的 head/loss 折算占比：%.1f%%"
          % (100.0 * (big2048 - big512) / d2048[0]["step"]))
