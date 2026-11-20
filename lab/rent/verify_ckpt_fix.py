# -*- coding: utf-8 -*-
"""独立验证 ec8051e：seq2048+ckpt 的 old vs new 交替 3 轮 + 跑新增单测。"""
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
NEW = "/root/patched_pkg"
OLD = "/root/old_pkg"
STEPS = 10

# 1) 搭 old_pkg（new 的副本 + 覆盖两个旧文件）
if os.path.isdir(OLD):
    shutil.rmtree(OLD)
shutil.copytree(NEW, OLD)
for nm in ("v3_train_cli.py", "v3_pretrain.py"):
    shutil.copy2(os.path.join("/root/_old_files", nm),
                 os.path.join(OLD, "tgai_go/train", nm))


def run(ppath, seq, ckpt=True, steps=STEPS):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = ppath
    if ckpt:
        env["TG_GRAD_CKPT"] = "1"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    s = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    return (float(s.group(1)) if s else None,
            float(pk.group(1)) if pk else None,
            out.strip().splitlines()[-1][:110] if not s else "")


print("=== 先跑新增的 CUDA 单测（HEAD 应通过）===")
env = dict(os.environ)
env["PYTHONPATH"] = NEW
env["PYTHONIOENCODING"] = "utf-8"
r = subprocess.run([PY, "-m", "pytest", "/root/tests/test_ckpt_parallel_calls.py",
                    "-q", "--no-header", "-p", "no:cacheprovider"],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", cwd="/root", env=env, timeout=1800)
tail = [l for l in ((r.stdout or "") + (r.stderr or "")).splitlines() if l.strip()]
print("  " + (tail[-1] if tail else "无输出"))

print("\n=== seq2048 + ckpt：old(021d66c) vs new(ec8051e)，交替 3 轮 ===")
res = {"old": [], "new": []}
for i in range(3):
    for tag, path in (("old", OLD), ("new", NEW)):
        s, pk, err = run(path, 2048)
        if s is None:
            print("  [%d] %s FAIL %s" % (i + 1, tag, err))
            continue
        res[tag].append((s, pk))
        print("  [%d] %-3s step=%6.4f s  peak=%5.2f GiB" % (i + 1, tag, s, pk or -1),
              flush=True)

print()
for tag in ("old", "new"):
    if not res[tag]:
        continue
    ss = [t[0] for t in res[tag]]
    pp = [t[1] for t in res[tag] if t[1] is not None]
    print("  %-3s 中位 %.4f s (%.4f~%.4f)  peak %.2f GiB"
          % (tag, statistics.median(ss), min(ss), max(ss),
             max(pp) if pp else -1))
if res["old"] and res["new"]:
    mo = statistics.median([t[0] for t in res["old"]])
    mn = statistics.median([t[0] for t in res["new"]])
    print("  => new 相对 old %+.1f%%   （PyT 参考 0.598 s -> new 快 %.2f×）"
          % (100.0 * (mn - mo) / mo, 0.598 / mn))
