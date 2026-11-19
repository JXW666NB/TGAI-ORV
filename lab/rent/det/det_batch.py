# -*- coding: utf-8 -*-
"""批量可复现性对照：一次 ssh 调用里跑多组 (冻结包 × 环境变量)，每组 2 个独立进程。

跑法（远端）：
    D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_batch.py --spec D:\\orv_bench\\spec_phase1.json

spec 形状：
    {"seq": 2048, "steps": 8, "amp": "gemm", "opt": "int8gpu", "layers": 12, "runs": 2,
     "configs": [{"tag": "orig_off", "pkg": "D:\\\\orv_det_work\\\\orig_pkg", "env": []},
                 {"tag": "orig_on",  "pkg": "D:\\\\orv_det_work\\\\orig_pkg", "env": ["TG_DETERMINISTIC=1"]}]}
"""
from __future__ import print_function

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import det_pair as DP                                        # noqa: E402

WORK = DP.WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    seq = int(spec.get("seq", 2048))
    steps = int(spec.get("steps", 8))
    amp = spec.get("amp", "gemm")
    opt = spec.get("opt", "int8gpu")
    layers = int(spec.get("layers", 12))
    runs = int(spec.get("runs", 2))
    summary = []
    first_run = {}
    for cfg in spec["configs"]:
        tag = cfg["tag"]
        pkg = cfg.get("pkg", "")
        envs = cfg.get("env", [])
        c_seq = int(cfg.get("seq", seq))
        c_amp = cfg.get("amp", amp)
        c_opt = cfg.get("opt", opt)
        c_layers = int(cfg.get("layers", layers))
        got = [DP.run_once("%s_%d" % (tag, i), pkg, envs, c_seq, steps, c_amp, c_opt,
                           c_layers, []) for i in range(runs)]
        first_run[tag] = got[0]
        res = DP.compare(tag, got[0], got[1])
        res["pkg"] = pkg or "<site-packages>"
        res["env"] = envs
        summary.append(res)
        print("[%-14s] pkg=%-28s env=%-28s seq=%d amp=%s opt=%s steps=%d "
              "loss_bitwise=%s gnorm_bitwise=%s"
              % (tag, os.path.basename(pkg) or "<sys>", ",".join(envs) or "-",
                 c_seq, c_amp, c_opt, res["steps"],
                 res["loss_bitwise_equal"], res["gnorm_bitwise_equal"]),
              flush=True)
        if not res["loss_bitwise_equal"]:
            i = res["first_loss_diff_step"]
            print("    first loss diff at step %s: A=%r B=%r (rel %.3e)"
                  % (i, res["loss_A"][i], res["loss_B"][i],
                     abs(res["loss_A"][i] - res["loss_B"][i]) / abs(res["loss_A"][i])),
                  flush=True)
        print("    loss_A=%r" % (res["loss_A"],), flush=True)
        print("    loss_B=%r" % (res["loss_B"],), flush=True)
        print("    gnorm_A=%r" % (res["gnorm_A"],), flush=True)
        print("    gnorm_B=%r" % (res["gnorm_B"],), flush=True)
    abs_ = []
    for pair in spec.get("ab", []):
        ta, tb = pair[0], pair[1]
        if ta not in first_run or tb not in first_run:
            print("[ab] skip %s vs %s (missing)" % (ta, tb), flush=True)
            continue
        r = DP.compare("AB_%s__%s" % (ta, tb), first_run[ta], first_run[tb])
        r["left"] = ta
        r["right"] = tb
        abs_.append(r)
        print("[AB][%-24s vs %-14s] loss_bitwise=%s (first_diff_step=%s worst_rel=%.3e) "
              "gnorm_bitwise=%s"
              % (ta, tb, r["loss_bitwise_equal"], r["first_loss_diff_step"],
                 r["worst_rel_loss_diff"], r["gnorm_bitwise_equal"]), flush=True)
        print("    loss_L=%r" % (r["loss_A"],), flush=True)
        print("    loss_R=%r" % (r["loss_B"],), flush=True)
        print("    gnorm_L=%r" % (r["gnorm_A"],), flush=True)
        print("    gnorm_R=%r" % (r["gnorm_B"],), flush=True)
    path = os.path.join(WORK, "det_batch_%s.json" % os.path.splitext(os.path.basename(a.spec))[0])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seq": seq, "steps": steps, "amp": amp, "opt": opt,
                   "layers": layers, "summary": summary, "ab": abs_},
                  f, ensure_ascii=False, indent=1)
    print("[det_batch] wrote %s" % path)


if __name__ == "__main__":
    sys.exit(main())
