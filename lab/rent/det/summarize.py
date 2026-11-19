# -*- coding: utf-8 -*-
"""本地汇总：从远端拉回的 det_batch 结果算 loss/gnorm 数组哈希（报告用）。"""
import hashlib
import json


def h(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


for tag in ("phase3", "phase2"):
    path = r"D:\orv_det_work\%s.json" % tag
    try:
        d = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        print("missing", path)
        continue
    print("== %s  seq=%d steps=%d amp=%s opt=%s" % (tag, d["seq"], d["steps"],
                                                    d.get("amp"), d.get("opt")))
    for s in d["summary"]:
        la, lb = s["loss_A"], s["loss_B"]
        ga, gb = s["gnorm_A"], s["gnorm_B"]
        print("  %-14s pkg=%-10s env=%-18s loss_eq=%-5s gnorm_eq=%-5s "
              "sha(loss)=%s sha(gnorm)=%s n=%d"
              % (s["tag"], s.get("pkg", "?").split("\\")[-1],
                 ",".join(s.get("env") or ["-"])[:18], la == lb, ga == gb,
                 h(la), h(ga), len(la)))
    for a in d.get("ab", []):
        print("  AB %s vs %s: loss_eq=%s worst_rel=%.3e"
              % (a["left"], a["right"], a["loss_bitwise_equal"],
                 a["worst_rel_loss_diff"]))
