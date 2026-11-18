# -*- coding: utf-8 -*-
"""batch>1 的最小复现与根因定位（目标第 3 项）。

已知现象：CSA 路径下 `--batch 2` 报
  `CudaError: reshape (2, 1024, 1024) -> (1024, 1024) 元素个数 2097152 != 1048576`
本脚本用最小配置（2 层、seq64、batch2）复现，并打印完整 traceback 定位到具体行。
"""
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"

cmds = [
    ("batch=1（对照，应成功）", [
        PY, "-m", "tgai_go.train.v3_pretrain",
        "--data", ROOT + r"\data2048_v96000", "--vocab", "96000", "--seq", "64",
        "--d", "128", "--layers", "2", "--E", "2", "--n_q", "4", "--d_ff", "256",
        "--top_k", "2", "--csa", "4", "--opt", "int8gpu", "--batch", "1",
        "--steps", "2", "--log-every", "1", "--no-final-save"]),
    ("batch=2（预期失败）", [
        PY, "-m", "tgai_go.train.v3_pretrain",
        "--data", ROOT + r"\data2048_v96000", "--vocab", "96000", "--seq", "64",
        "--d", "128", "--layers", "2", "--E", "2", "--n_q", "4", "--d_ff", "256",
        "--top_k", "2", "--csa", "4", "--opt", "int8gpu", "--batch", "2",
        "--steps", "2", "--log-every", "1", "--no-final-save"]),
]

for tag, cmd in cmds:
    print("===== %s =====" % tag)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    if '"status": "ok"' in out:
        print("  成功")
    else:
        # 只打印与出错位置相关的行
        lines = out.splitlines()
        keep = []
        for i, ln in enumerate(lines):
            if ("Traceback" in ln or "CudaError" in ln or "reshape" in ln
                    or "v3_csa.py" in ln or "cuda.py" in ln or "line " in ln
                    or "File \"" in ln):
                keep.append(ln.strip())
        print("\n".join("  " + k for k in keep[:24]) or "  （无匹配输出）")
    print()
