# -*- coding: utf-8 -*-
"""判定：PyTorch 的长序列优势在 C 端（CUDA 内核）还是 Py 端（框架/派发）。

方法：把 ORV 的步时拆成「固定成本 c（与 token 数无关）」+「边际成本 a·N（每 token）」。
同一 seq 下改 batch（1 vs 2）即可解出 c 与 a：
    c + a·512  = t(batch=1)
    c + a·1024 = t(batch=2)
再拿 seq2048 的实测去检验：若模型预测远低于实测，说明**边际成本在长 seq 变大**
⇒ 瓶颈在 GPU 侧的大张量效率（C 端）；若吻合，说明只是固定成本不变、线性的（Py 端摊薄问题）。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 20


def orv(seq, batch):
    cmd = [ORV, os.path.join(ROOT, "split_stages_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", str(batch),
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not r:
        return None
    return json.loads(r.group(1))["step_s_steady"]


t512_1 = orv(512, 1)
t512_2 = orv(512, 2)
print("seq512  batch=1 %.3f s   batch=2 %.3f s" % (t512_1, t512_2))
a512 = (t512_2 - t512_1) / 512.0
c512 = t512_1 - 512 * a512
print("  解出：固定成本 c = %.3f s/步，边际成本 a = %.3e s/token" % (c512, a512))
print("  固定成本占 seq512 步时的 %.0f%%" % (100 * c512 / t512_1))

t2048_1 = orv(2048, 1)
t2048_2 = orv(2048, 2)
print()
print("seq2048  batch=1 %.3f s   batch=2 %.3f s" % (t2048_1, t2048_2))
a2048 = (t2048_2 - t2048_1) / 2048.0
c2048 = t2048_1 - 2048 * a2048
print("  解出：固定成本 c = %.3f s/步，边际成本 a = %.3e s/token" % (c2048, a2048))

print()
print("边际成本变化（4× token）：%.3e → %.3e = %.2f×"
      % (a512, a2048, a2048 / a512))
print("若边际成本不变，seq2048 的 batch=1 应为 %.3f s（实测 %.3f s）"
      % (c512 + 2048 * a512, t2048_1))
print("差值 %.3f s 就是「每 token 变贵」的部分 ⇒ 属于 C 端（GPU 内核效率）"
      % (t2048_1 - (c512 + 2048 * a512)))
