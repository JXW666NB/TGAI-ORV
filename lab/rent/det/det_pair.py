# -*- coding: utf-8 -*-
"""同一命令跑**两个独立进程**，逐位比较每步 loss / gnorm。

为什么要独立进程：同进程内第二次运行会命中热缓存、且 CUDA 上下文状态不同；
验收要求的是"换一个进程重跑仍逐位相同"。

跑法（远端）：
    D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_pair.py --tag A --pkg D:\\orv_det_work\\new_pkg --env TG_DETERMINISTIC=1

`--pkg` 是**冻结包**的父目录（里面有 tgai_go\\），通过 PYTHONPATH 抢在 site-packages 之前。
不给 `--pkg` 就用 site-packages。
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
WORK = r"D:\orv_det_work"
DROP = ("TG_DETERMINISTIC", "TG_POOL_GB", "TG_AMP_BMM", "TG_GRAD_CKPT",
        "TG_MOE_MERGED", "TG_FP16_MAX_K", "TG_AMP_MLP", "TG_MT")


def _sha(obj):
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:16]


def run_once(tag, pkg, env_extra, seq, steps, amp, opt, layers, extra_args):
    out = os.path.join(WORK, "runs", tag)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for k in DROP:
        env.pop(k, None)
    if pkg:
        env["PYTHONPATH"] = pkg
    else:
        env.pop("PYTHONPATH", None)
    for kv in env_extra:
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = [PY, "-X", "utf8", "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048_v96000"),
           "--vocab", "96000", "--seq", str(seq), "--d", "1024",
           "--layers", str(layers), "--E", "4", "--n_q", "16", "--d_ff", "3328",
           "--top_k", "2", "--csa", "4", "--opt", opt, "--amp", amp,
           "--batch", "1", "--steps", str(steps), "--log-every", "0",
           "--no-final-save", "--ckpt-every", "0", "--out", out] + list(extra_args)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    log = os.path.join(out, "train_log.jsonl")
    recs = []
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    if not recs:
        tail_o = (p.stdout or "")[-3000:]
        tail_e = (p.stderr or "")[-3000:]
        print("[%s] FAILED rc=%d\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
              % (tag, p.returncode, tail_o, tail_e))
        raise SystemExit(1)
    return {"recs": recs, "rc": p.returncode, "out": out}


def compare(tag, A, B):
    la = [r["loss"] for r in A["recs"]]
    lb = [r["loss"] for r in B["recs"]]
    ga = [r["gnorm"] for r in A["recs"]]
    gb = [r["gnorm"] for r in B["recs"]]
    n = min(len(la), len(lb))
    first_l = next((i for i in range(n) if la[i] != lb[i]), None)
    first_g = next((i for i in range(n) if ga[i] != gb[i]), None)
    worst = 0.0
    for i in range(n):
        if la[i] != lb[i]:
            d = abs(la[i] - lb[i]) / max(abs(la[i]), 1e-30)
            worst = max(worst, d)
    res = {
        "tag": tag,
        "steps": n,
        "loss_bitwise_equal": bool(la[:n] == lb[:n]),
        "gnorm_bitwise_equal": bool(ga[:n] == gb[:n]),
        "first_loss_diff_step": first_l,
        "first_gnorm_diff_step": first_g,
        "loss_sha_A": _sha(la), "loss_sha_B": _sha(lb),
        "gnorm_sha_A": _sha(ga), "gnorm_sha_B": _sha(gb),
        "worst_rel_loss_diff": worst,
        "loss_A": la, "loss_B": lb, "gnorm_A": ga, "gnorm_B": gb,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pkg", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", default="gemm")
    ap.add_argument("--opt", default="int8gpu")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--args", default="", help="额外命令行，空格分隔")
    a = ap.parse_args()
    extra = [x for x in a.args.split(" ") if x]
    runs = [run_once("%s_%d" % (a.tag, i), a.pkg, a.env, a.seq, a.steps,
                     a.amp, a.opt, a.layers, extra) for i in range(a.runs)]
    out = {"tag": a.tag, "pkg": a.pkg or "<site-packages>", "env": a.env,
           "seq": a.seq, "steps": a.steps, "amp": a.amp, "opt": a.opt,
           "layers": a.layers}
    print("[%s] pkg=%s env=%s seq=%d amp=%s opt=%s runs=%d"
          % (a.tag, a.pkg or "<site-packages>", a.env, a.seq, a.amp, a.opt, a.runs))
    pairs = []
    for i in range(1, len(runs)):
        r = compare("%s_A%d" % (a.tag, i), runs[0], runs[i])
        pairs.append(r)
        print("  run0 vs run%d: loss_bitwise=%s (first_diff_step=%s, worst_rel=%.3e) "
              "gnorm_bitwise=%s (first_diff_step=%s)"
              % (i, r["loss_bitwise_equal"], r["first_loss_diff_step"],
                 r["worst_rel_loss_diff"], r["gnorm_bitwise_equal"],
                 r["first_gnorm_diff_step"]))
        print("    loss_sha A=%s B=%s   gnorm_sha A=%s B=%s"
              % (r["loss_sha_A"], r["loss_sha_B"], r["gnorm_sha_A"], r["gnorm_sha_B"]))
    r0 = pairs[0] if pairs else None
    if r0:
        print("  loss_A = %r" % (r0["loss_A"],))
        print("  loss_B = %r" % (r0["loss_B"],))
        print("  gnorm_A = %r" % (r0["gnorm_A"],))
        print("  gnorm_B = %r" % (r0["gnorm_B"],))
    out["pairs"] = [{k: v for k, v in p.items() if not k.endswith("_A") and not k.endswith("_B")}
                    for p in pairs]
    out["loss_A"] = r0["loss_A"] if r0 else []
    out["loss_B"] = r0["loss_B"] if r0 else []
    out["gnorm_A"] = r0["gnorm_A"] if r0 else []
    out["gnorm_B"] = r0["gnorm_B"] if r0 else []
    path = os.path.join(WORK, "det_%s.json" % a.tag)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[det_pair] wrote %s" % path)


if __name__ == "__main__":
    sys.exit(main())
