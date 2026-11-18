# -*- coding: utf-8 -*-
"""在 V100 上跑组件拆解：注意力占比 × {seq} × {MoE, 稠密}。

每一项跑 `split_orv.py`（真实训练器 + 注意力计时），从输出里取：
  - `[split]` 里的 attn_s_total / attn_calls  ⇒ 每层每次前向的注意力耗时
  - RESULT 里的 step_s_steady                ⇒ 总步时
然后按 步数 与 层数 折算占比。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
PY = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 3


def one(seq, layers, mode):
    dff = 3328 if mode == "moe" else 13312
    e = 4 if mode == "moe" else 1
    tk = 2 if mode == "moe" else 1
    cmd = [PY, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers),
           "--E", str(e), "--n_q", "16", "--d_ff", str(dff), "--top_k", str(tk),
           "--csa", "4", "--opt", "int8gpu", "--steps", str(STEPS),
           "--log-every", "99", "--no-final-save"] + (["--no-shared"] if mode == "dense" else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"\[split\] (\{.*\})", out)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m or not r:
        why = "OOM" if ("显存不足" in out or "out of memory" in out.lower()) else "FAIL"
        return "%s seq%-5d L%-3d FAIL: %s" % (mode.upper().ljust(5), seq, layers, why)
    sp = json.loads(m.group(1))
    res = json.loads(r.group(1))
    step = res["step_s_steady"]
    # 每步的注意力总耗时 = 总累计 / 步数（末步也算了 foreward，近似均匀）
    attn_per_step = sp["attn_s_total"] / float(STEPS)
    pct = 100.0 * attn_per_step / step if step else 0.0
    return ("%s seq%-5d L%-3d step=%6.3f s  attn=%6.3f s (%4.1f%%)  rest=%6.3f s"
            % (mode.upper().ljust(5), seq, layers, step, attn_per_step, pct,
               step - attn_per_step))


def main():
    print("===== ORV 组件拆解（V100，注意力 vs 其余）=====")
    for seq in (512, 1024, 2048):
        print(one(seq, 12, "moe"))
    print()
    print("===== 稠密对照（无专家）=====")
    for seq in (512, 2048):
        print(one(seq, 12, "dense"))


if __name__ == "__main__":
    main()
