# -*- coding: utf-8 -*-
"""把 remote_out/ 下的 JSON 汇总成 markdown 表（报告用）。"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "remote_out")


def load(name):
    for sub in ("", "gpu"):
        p = os.path.join(ROOT, sub, name if name.endswith(".json") else name + ".json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def g(v, p=6):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if abs(v) != 0 and (abs(v) < 1e-4 or abs(v) >= 1e5):
        return "%.*e" % (max(1, p - 1), v)
    return "%.*g" % (p, v)


def enrich(recs):
    """补 t_step_ms / ns_saved 两个派生列。"""
    seen = set()
    out = []
    for r in list(recs):
        if id(r) in seen:
            continue
        seen.add(id(r))
        if "error" not in r:
            if r.get("t_step_ms") is None and r.get("t_step_s") is not None:
                r["t_step_ms"] = r["t_step_s"] * 1e3
            st = r.get("opt_status")
            r["ns_saved"] = st.get("ns_saved_pct") if st else None
            r["orth_freq"] = st.get("orth_freq") if st else None
            r["gate_hit"] = st.get("gate_hit") if st else None
        out.append(r)
    return out


def best_by(recs, key="last10_mean"):
    best = {}
    for r in recs:
        if "error" in r or r.get(key) is None or not r.get("all_finite", True):
            continue
        if r["opt"] not in best or r[key] < best[r["opt"]][key]:
            best[r["opt"]] = r
    return best


def get(r, k):
    if k == "t_step_ms":
        v = r.get("t_step_ms")
        if v is None and r.get("t_step_s") is not None:
            v = r["t_step_s"] * 1e3
        return v
    if k == "ns_saved":
        st = r.get("opt_status")
        return st.get("ns_saved_pct") if st else None
    return r.get(k)


def table_full(recs, cols):
    head = "| 优化器 | lr | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---|---|" + "|".join("---" for _ in cols) + "|"
    out = [head, sep]
    ords = []
    for r in recs:
        if r["opt"] not in ords:
            ords.append(r["opt"])
    for o in ords:
        rs = sorted([r for r in recs if r["opt"] == o],
                    key=lambda r: (r.get("lr") or 0))
        for r in rs:
            if "error" in r:
                out.append("| %s | %s | ERROR |" % (o, g(r.get("lr"))))
                continue
            out.append("| %s | %s | %s |" % (o, g(r.get("lr")),
                                             " | ".join(g(get(r, k)) for _, k in cols)))
    return "\n".join(out)


def table_best(recs, cols, key="last10_mean"):
    best = best_by(recs, key)
    head = "| 优化器 | 最优 lr | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---|---|" + "|".join("---" for _ in cols) + "|"
    out = [head, sep]
    for o in [r["opt"] for r in recs if r["opt"] in best]:
        pass
    seen = []
    for r in recs:
        if r["opt"] in best and r["opt"] not in seen:
            seen.append(r["opt"])
    for o in seen:
        r = best[o]
        out.append("| %s | %s | %s |" % (o, g(r["lr"]),
                                         " | ".join(g(get(r, k)) for _, k in cols)))
    return "\n".join(out)


def cpu_gpu_diff(cpu, gpu, lr_key="lr"):
    """逐 lr 比较 toy 的 excess_last10。"""
    cmap = {}
    for r in cpu:
        if "error" in r:
            continue
        cmap[(r["opt"], r[lr_key])] = r
    gmap = {}
    for r in gpu:
        if "error" in r:
            continue
        gmap[(r["opt"], r[lr_key])] = r
    rows = []
    ords = []
    for r in cpu + gpu:
        if r["opt"] not in ords:
            ords.append(r["opt"])
    for o in ords:
        lrs = sorted({k[1] for k in list(cmap) + list(gmap) if k[0] == o})
        for lr in lrs:
            c = cmap.get((o, lr))
            gg = gmap.get((o, lr))
            ce = c.get("excess_last10") if c else None
            ge = gg.get("excess_last10") if gg else None
            ratio = (ge / ce) if (ce and ge and ce > 0) else None
            rows.append((o, lr, ce, ge, ratio,
                         c.get("steps_to_excess_1e-3") if c else None,
                         gg.get("steps_to_excess_1e-3") if gg else None))
    out = ["| 优化器 | lr | CPU excess(末10) | CUDA excess(末10) | CUDA/CPU | "
           "CPU 步数到1e-3 | CUDA 步数到1e-3 |", "|---|---|---|---|---|---|---|"]
    for o, lr, ce, ge, ratio, cs, gs in rows:
        out.append("| %s | %s | %s | %s | %s | %s | %s |"
                   % (o, g(lr), g(ce, 4), g(ge, 4), g(ratio, 3), g(cs), g(gs)))
    return "\n".join(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("## 1. 玩具回归 - CPU 1200 步（p10）")
    d = load("p10_cpu_toy")
    if d:
        print(table_best(d["records"],
                         [("末10均值", "last10_mean"), ("最低值", "min"),
                          ("超额(末10)", "excess_last10"),
                          ("||W-W*||/||W*||", "rel_dist_to_Wstar"),
                          ("步数→超额1e-3", "steps_to_excess_1e-3"),
                          ("步数→超额1e-5", "steps_to_excess_1e-5"),
                          ("步时 ms", "t_step_ms")]))
        print()
        for r in d["records"]:
            r["t_step_ms"] = None if r.get("t_step_s") is None else r["t_step_s"] * 1e3

    print("=" * 70)
    print("## 2. 玩具回归 - CPU/CUDA 逐 lr（各 600 步, p12a vs g10a）")
    c = load("p12a_cpu_toy600")
    dg = load("g10a_toy")
    if c and dg:
        print(cpu_gpu_diff(c["records"], dg["records"]))
    elif dg:
        print("(CPU 600 步还没跑完)")

    print("=" * 70)
    print("## 3. 迷你 V3 固定批过拟合 - CPU 150 步（p12b）")
    d = load("p12b_cpu_overfit")
    if d:
        print(table_full(d["records"],
                         [("末10均值", "last10_mean"), ("最低值", "min"),
                          ("loss@25", "loss_at_25"), ("loss@50", "loss_at_50"),
                          ("步时 ms", "t_step_s"), ("ns_saved%", "ns_saved")]))
    else:
        print("(未完成)")

    print("=" * 70)
    print("## 4. 迷你 V3 固定批过拟合 - CUDA 200 步（g10b）")
    d = load("g10b_model")
    if d:
        print(table_full(d["records"],
                         [("末10均值", "last10_mean"), ("最低值", "min"),
                          ("loss@25", "loss_at_25"), ("loss@50", "loss_at_50"),
                          ("步时 ms", "t_step_ms"), ("ns_saved%", "ns_saved")]))

    print("=" * 70)
    print("## 5. 速度/显存（g10c, CUDA event）")
    d = load("g10c_speed")
    if d:
        out = ["| 优化器 | 整步 ms(中位) | 整步 3 次 | opt.step ms | tokens/s | 峰值 GiB |",
               "|---|---|---|---|---|---|"]
        for k, v in d["speed"].items():
            if "error" in v:
                out.append("| %s | ERROR | | | | |" % k)
                continue
            out.append("| %s | %s | %s | %s | %s | %s |"
                       % (k, g(v["step_ms_median"], 4),
                          "/".join(g(x, 4) for x in v["step_ms_all"]),
                          g(v["opt_step_ms_median"], 4), g(v["tokens_per_s"], 6),
                          g(v["peak_gib"], 4)))
        print("\n".join(out))

    print("=" * 70)
    print("## 6. AMP 开/关（g10d）")
    d = load("g10d_amp")
    if d:
        for k, v in d.items():
            if isinstance(v, dict) and "last10_mean" in v:
                print("- %-18s 末10=%-10s min=%-10s loss@50=%-10s ms/step=%-6s finite=%s"
                      % (k, g(v["last10_mean"], 6), g(v["min"], 6),
                         g(v.get("loss_at_50"), 6),
                         g(None if v.get("t_step_s") is None else v["t_step_s"] * 1e3, 4),
                         v.get("all_finite")))
            elif isinstance(v, dict) and "error" in v:
                print("- %-18s ERROR %s" % (k, v["error"][:80]))

    print("=" * 70)
    print("## 7. PyTorch AdamW（p30）")
    d = load("p30_pyt_model")
    if d:
        print("parity: 初始 loss torch=%s orv=%s diff=%s device=%s" % (
            g(d["parity"].get("init_loss_b8_torch"), 8),
            g(d["parity"].get("init_loss_b8_orv"), 8),
            g(d["parity"].get("init_loss_absdiff"), 3), d["parity"].get("device")))
        print(table_best(d["records"],
                         [("末10均值", "last10_mean"), ("最低值", "min"),
                          ("loss@50", "loss_at_50"), ("步时 ms", "t_step_ms")]))
        print("speed: " + json.dumps(d.get("speed"), ensure_ascii=True))

    print("=" * 70)
    print("## 8. 流式（每步换批）@各自最优 lr（p12c）")
    d = load("p12c_cpu_stream")
    if d:
        print(table_full(d["records"],
                         [("末10均值", "last10_mean"), ("最低值", "min"),
                          ("loss@25", "loss_at_25"), ("loss@50", "loss_at_50"),
                          ("步时 ms", "t_step_ms")]))


if __name__ == "__main__":
    main()
