# -*- coding: utf-8 -*-
"""公平性对照：PyTorch 开/关梯度检查点，看我们的对比是否给了对方不利条件。

背景：`tools/v3_pyt_train.py` 默认 `enable_gradient_checkpointing(True)`，
即 PyTorch 那侧一直背着"重算"的代价（换来省显存）。ORV 没有这个开关
（它的省显存靠分块注意力 + 更紧的激活管理）。所以必须给出两个口径：
  ckpt=1：原口径（省显存、多付重算）
  ckpt=0：纯计算口径（显存更高）
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
PYT = r"C:\Python311\python.exe"
STEPS = 3


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def pyt(L, DFF, E, TK, SEQ, ckpt):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(L), "--E", str(E), "--vocab", "96000",
           "--d_ff", str(DFF), "--n_heads", "16", "--top_k", str(TK),
           "--seq", str(SEQ), "--steps", str(STEPS), "--ckpt", str(ckpt)]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0}


CASES = [
    ("MoE   12L", 12, 3328, 4, 2, (512, 1024, 2048)),
    ("稠密  12L", 12, 13312, 1, 1, (512, 2048)),
]

for name, L, DFF, E, TK, seqs in CASES:
    print("===== %s =====" % name)
    for seq in seqs:
        a = pyt(L, DFF, E, TK, seq, 1)
        b = pyt(L, DFF, E, TK, seq, 0)
        if not a or not b:
            print("  seq%-5d ckpt1=%s ckpt0=%s" % (seq, a, b), flush=True)
            continue
        print("  seq%-5d ckpt=1 %6.3f s / %5.1f GiB   ckpt=0 %6.3f s / %5.1f GiB"
              "   ⇒ 检查点代价 %+.1f%%（省显存 %.1f%%）"
              % (seq, a["step"], a["mem"], b["step"], b["mem"],
                 100 * (a["step"] - b["step"]) / b["step"],
                 100 * (b["mem"] - a["mem"]) / b["mem"]), flush=True)
    print()
