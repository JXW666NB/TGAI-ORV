# -*- coding: utf-8 -*-
"""CPU vs CUDA 逐配置对齐（同一步数），以及把 GPU 200 步曲线截断到 150 步。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "remote_out")


def load(name):
    for sub in ("", "gpu"):
        p = os.path.join(ROOT, sub, name + ".json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def g(v, p=4):
    if v is None:
        return "-"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
        return "%.*e" % (max(1, p - 1), v)
    return "%.*g" % (p, v)


def last10(curve, n):
    c = [x for x in curve[:n] if x is not None]
    tail = c[-10:]
    return sum(tail) / len(tail), min(c)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cpu = load("p12a_cpu_toy600")["records"]
    gpu = load("g10a_toy")["records"]
    cm = {(r["opt"], round(r["lr"], 12)): r for r in cpu if "error" not in r}
    gm = {(r["opt"], round(r["lr"], 12)): r for r in gpu if "error" not in r}
    print("## 玩具回归 CPU(600) vs CUDA(600)：末10 excess")
    print("| 优化器 | lr | CPU | CUDA | CUDA/CPU | CPU 步数→1e-3 | CUDA 步数→1e-3 |")
    print("|---|---|---|---|---|---|---|")
    ords = []
    for r in cpu:
        if r["opt"] not in ords:
            ords.append(r["opt"])
    for o in ords:
        lrs = sorted({k[1] for k in cm if k[0] == o})
        for lr in lrs:
            c, gg = cm.get((o, lr)), gm.get((o, lr))
            ce = c["excess_last10"] if c else None
            ge = gg["excess_last10"] if gg else None
            ratio = (ge / ce) if (ce and ge) else None
            print("| %s | %s | %s | %s | %s | %s | %s |"
                  % (o, g(lr), g(ce), g(ge), g(ratio, 3),
                     g(c["steps_to_excess_1e-3"] if c else None),
                     g(gg["steps_to_excess_1e-3"] if gg else None)))

    print()
    print("## 迷你 V3 过拟合 CPU(150) vs CUDA(截断到 150)")
    cpu2 = [r for r in load("p12b_cpu_overfit")["records"] if "error" not in r]
    gpu2 = [r for r in load("g10b_model")["records"] if "error" not in r]
    gm2 = {(r["opt"], round(r["lr"], 12)): r for r in gpu2}
    print("| 优化器 | lr | CPU 末10(150) | CUDA 末10(150) | CPU min | CUDA min | CUDA/CPU |")
    print("|---|---|---|---|---|---|---|")
    for r in cpu2:
        key = (r["opt"], round(r["lr"], 12))
        gg = gm2.get(key)
        if gg is None or gg.get("curve") is None:
            continue
        gl10, gmin = last10(gg["curve"], 150)
        cl10 = r["last10_mean"]
        cmin = r["min"]
        print("| %s | %s | %s | %s | %s | %s | %s |"
              % (r["opt"], g(r["lr"]), g(cl10), g(gl10), g(cmin, 3), g(gmin, 3),
                 g(gl10 / cl10 if cl10 else None, 3)))

    print()
    print("## Polaris 稳定性：CUDA 200 步 loss 序列（lr=3e-3 / 1e-2）")
    for r in gpu2:
        if r["opt"] == "Polaris" and r["lr"] in (3e-3, 1e-2):
            c = r["curve"]
            print("lr=%s 末10=%s min=%s" % (g(r["lr"]), g(r["last10_mean"]),
                                            g(r["min"])))
            for a in range(0, len(c), 10):
                seg = c[a:a + 10]
                print("  step %3d-%3d: %s" % (a + 1, a + len(seg),
                                              " ".join(g(x, 4) if x is not None else "-"
                                                       for x in seg)))


if __name__ == "__main__":
    main()
