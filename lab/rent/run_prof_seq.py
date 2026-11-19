# -*- coding: utf-8 -*-
"""跑 seq512 / seq2048 两档的 event 归因，出表 A / B / C（+两张补充表）。

口径（与既有实测一致）：
  V3 MoE（d=1024 / d_ff=3328 / E=4+1 共享 / top_k=2 / vocab 96000），12 层，
  batch=1，--opt int8gpu --amp gemm --csa 4 --n_q 16，20 步，前 3 步不计入。

表 A：kernel 级热门榜（seq2048），按 每步 ms 降序；含 前向/反向 分摊、调用次数。
表 B：forward / backward / clip / optimizer 四段的 event 时间，以及每段里
      「被叶子 event 捕获到的 GPU 时间」与「段内空档」。
表 B2（补充）：空档最大的相邻叶子对 —— 空档 = GPU 在等主机或存在未打点操作。
表 B3（补充）：主机侧同步/分配类调用（cuMemAlloc / cuCtxSynchronize / 同步 memcpy）。
表 C：seq512 vs seq2048 同一张表，按「每步耗时增长倍数」排序（token ×4）。

用法：
    python run_prof_seq.py            # 两档都跑
    python run_prof_seq.py --reuse    # 只读已有 json 出表
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PROF = os.path.join(ROOT, "prof_event_kernels.py")
STEPS = 20
SKIP = 3
OUTS = {512: os.path.join(ROOT, "prof_512.json"),
        2048: os.path.join(ROOT, "prof_2048.json")}

PEAK_GBS = 900.0             # V100 HBM2 理论 900 GB/s（STREAM 实测约 830）
PHASES = ("forward", "backward", "clip", "optimizer")


def run_one(seq):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PROF_OUT"] = OUTS[seq]
    env["PROF_SKIP"] = str(SKIP)
    env["PROF_TAG"] = "seq%d" % seq
    cmd = [ORV, PROF,
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    print("[run] seq=%d ..." % seq, flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        sys.stderr.write(out[-4000:])
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    steady = json.loads(m.group(1))["step_s_steady"] if m else None
    for line in out.splitlines():
        if line.startswith("[prof]") or line.startswith("[run]") or line.startswith("RESULT"):
            print("   " + line, flush=True)
    if not os.path.exists(OUTS[seq]):
        raise SystemExit("seq=%d 没产出 %s" % (seq, OUTS[seq]))
    with open(OUTS[seq], encoding="utf-8") as f:
        d = json.load(f)
    d["trainer_step_s_steady"] = steady
    return d


def per_label(d):
    agg = {}
    for r in d["ops"]:
        a = agg.get(r["label"])
        if a is None:
            agg[r["label"]] = a = {"ms": 0.0, "n": 0.0, "ph": {}}
        a["ms"] += r["ms_per_step"]
        a["n"] += r["n_per_step"]
        for k, v in r["phase"].items():
            a["ph"][k] = a["ph"].get(k, 0.0) + v
    return agg


def phase_leaf(d):
    """每个阶段里被叶子 event 捕获到的 GPU 毫秒（含阶段外的 'other'）。"""
    out = {}
    for r in d["ops"]:
        for k, v in r["phase"].items():
            out[k] = out.get(k, 0.0) + v
    return out


def shape_desc(d, label, aux):
    if label.startswith("cublas"):
        if len(aux) == 4:
            return "m=%d n=%d k=%d batch=%d" % tuple(aux)
        return "m=%d n=%d k=%d" % tuple(aux)
    if label.startswith("mem:"):
        return "%d B" % (aux[0] if aux else 0)
    smp = (d["samples"].get(label) or {}).get(str(tuple(aux)))
    if smp:
        return "ints=(%s) p=%d" % (",".join(str(x) for x in smp["ints"]),
                                   smp.get("nptr", 0))
    return "ints=(%s)" % (",".join(str(x) for x in aux),)


def table_a(d, top=20):
    wall_ms = d["step_s_wall_mean"] * 1000.0
    tot = d["sum_leaf_ms_per_step"]
    print()
    print("=" * 136)
    print("表 A  kernel 级热门榜 — seq2048 / batch1 / AMP=gemm / int8gpu / %d 步稳态（跳过前 %d 步）"
          % (d["measured_steps"], d["skipped_steps"]))
    print("     步时(墙钟) %.1f ms ；叶子 GPU 区间合计 %.1f ms（%.1f%% 步时）；叶子调用 %.0f 次/步"
          % (wall_ms, tot, 100.0 * tot / wall_ms, d["n_leaf_per_step"]))
    print("=" * 136)
    print("%-3s %-29s %-30s %9s %7s %7s %9s %9s %8s" % (
        "#", "kernel / leaf", "shape（本行实测）", "ms/步", "%步时", "次/步",
        "前向", "反向", "其它"))
    print("-" * 136)
    for i, r in enumerate(d["ops"][:top], 1):
        lbl = r["label"]
        ph = r["phase"]
        other = (ph.get("other", 0.0) + ph.get("clip", 0.0)
                 + ph.get("optimizer", 0.0))
        print("%-3d %-29s %-30s %9.3f %6.1f%% %7.1f %9.3f %9.3f %8.3f" % (
            i, lbl[:29], shape_desc(d, lbl, r["aux"])[:30],
            r["ms_per_step"], 100.0 * r["ms_per_step"] / wall_ms,
            r["n_per_step"], ph.get("forward", 0.0), ph.get("backward", 0.0),
            other))
    top5 = sum(r["ms_per_step"] for r in d["ops"][:5])
    print("-" * 136)
    print("前 5 合计 %.1f ms/步 = 步时的 %.1f%%（占叶子 GPU 区间的 %.1f%%）"
          % (top5, 100.0 * top5 / wall_ms, 100.0 * top5 / tot))
    print("形状种类（label×shape）共 %d 个；叶子总调用 %.0f 次/步"
          % (d["n_label_shape"], d["n_leaf_per_step"]))
    return d["ops"][:top]


def table_b(d):
    wall_ms = d["step_s_wall_mean"] * 1000.0
    ph = d["phases"]
    leaf = phase_leaf(d)
    print()
    print("=" * 118)
    print("表 B  四段 event 计时（GPU 时间轴）— seq2048 / batch1 / AMP=gemm / int8gpu")
    print("=" * 118)
    print("%-12s %10s %8s %7s %12s %11s %8s"
          % ("阶段", "段跨度ms", "占步时", "调用/步", "段内叶子ms", "段内空档ms", "叶子占比"))
    print("-" * 118)
    acc = 0.0
    for k in PHASES:
        if k not in ph:
            continue
        v = ph[k]
        lf = leaf.get(k, 0.0)
        acc += v["ms_per_step"]
        print("%-12s %10.3f %7.1f%% %7.1f %12.3f %11.3f %7.1f%%"
              % (k, v["ms_per_step"], 100.0 * v["ms_per_step"] / wall_ms,
                 v["n_per_step"], lf, v["ms_per_step"] - lf,
                 100.0 * lf / v["ms_per_step"] if v["ms_per_step"] else 0.0))
    print("-" * 118)
    print("%-12s %10.3f %7.1f%% %7s %12.3f %11.3f %7.1f%%"
          % ("四段合计", acc, 100.0 * acc / wall_ms, "", sum(leaf.get(k, 0.0) for k in PHASES),
             acc - sum(leaf.get(k, 0.0) for k in PHASES),
             100.0 * sum(leaf.get(k, 0.0) for k in PHASES) / acc if acc else 0.0))
    oth = leaf.get("other", 0.0)
    print("%-12s %10.3f %7.1f%% %7s %12.3f" % ("阶段外(other)", wall_ms - acc,
                                               100.0 * (wall_ms - acc) / wall_ms, "", oth))
    print("%-12s %10.3f %7.1f%%" % ("步时(墙钟)", wall_ms, 100.0))
    return acc


def table_b2(d, top=14):
    print()
    print("=" * 118)
    print("表 B2（补充）  段内空档最大的相邻叶子对 — 空档 = GPU 在等主机，或存在未打点的 GPU 操作")
    print("=" * 118)
    print("%-27s %-27s %8s %10s" % ("前一个叶子", "后一个叶子", "次/步", "ms/步"))
    print("-" * 118)
    for g in d["gaps"][:top]:
        print("%-27s %-27s %8.1f %10.3f"
              % (g["prev"][:27], g["next"][:27], g["n_per_step"], g["ms_per_step"]))
    print("-" * 118)
    print("空档合计 %.1f ms/步（占步时 %.1f%%）"
          % (d["sum_gap_ms_per_step"],
             100.0 * d["sum_gap_ms_per_step"] / (d["step_s_wall_mean"] * 1000.0)))
    return d["gaps"][:top]


def table_b3(d):
    wall_ms = d["step_s_wall_mean"] * 1000.0
    print()
    print("=" * 126)
    print("表 B3（补充）  CPU 侧成本拆解 — 「主机在干什么」")
    print("=" * 126)
    print("%-26s %10s %12s %10s %s" % ("项", "次/步", "CPU ms/步", "占步时", "说明"))
    print("-" * 126)
    rows = sorted(d["host"].items(), key=lambda kv: -kv[1]["ms_per_step"])
    for k, v in rows:
        ph = " ".join("%s=%.1f" % (pk, pv) for pk, pv in
                      sorted(v["phase"].items(), key=lambda x: -x[1]) if pv > 0.3)
        print("%-26s %10.1f %12.3f %9.1f%%  %s"
              % (k, v["n_per_step"], v["ms_per_step"],
                 100.0 * v["ms_per_step"] / wall_ms, ph[:44]))
    hl = d["host_leaf"]
    tot_h = sum(v["host_ms_per_step"] for v in hl.values())
    tot_n = sum(v["n_per_step"] for v in hl.values())
    print("%-26s %10.1f %12.3f %9.1f%%  %s"
          % ("【叶子发射 CPU 合计】", tot_n, tot_h, 100.0 * tot_h / wall_ms,
             "含 2 次 eventRecord + 包装"))
    print("-" * 126)
    print("GPU 叶子区间 %.1f ms / CPU 叶子发射 %.1f ms / 主机侧驱动调用 %.1f ms / 步时 %.1f ms"
          % (d["sum_leaf_ms_per_step"], tot_h,
             sum(v["ms_per_step"] for _, v in rows), wall_ms))
    return rows


def table_b4(d):
    pol = d.get("pool") or {}
    if not pol:
        return
    print()
    print("=" * 100)
    print("表 B4（补充）  显存池统计（按阶段增量）— hits = 池命中，misses = 必须 cuMemAlloc")
    print("=" * 100)
    print("%-12s %10s %10s %10s %10s" % ("阶段", "hits/步", "misses/步", "命中率",
                                         "trims/步"))
    print("-" * 100)
    th = tm = tt = 0.0
    for k in ("forward", "backward", "clip", "optimizer", "other"):
        v = pol.get(k)
        if not v:
            continue
        h, m, t = v
        th += h
        tm += m
        tt += t
        tot = h + m
        print("%-12s %10.1f %10.1f %9.1f%% %10.1f"
              % (k, h, m, 100.0 * h / tot if tot else 0.0, t))
    print("-" * 100)
    print("%-12s %10.1f %10.1f %9.1f%% %10.1f"
          % ("合计", th, tm, 100.0 * th / (th + tm) if (th + tm) else 0.0, tt))
    ps = d.get("pool_stats") or {}
    if ps:
        print("进程累计：hits=%s misses=%s trims=%s 命中率=%.1f%% cap=%.1f GiB 峰值占用=%.1f GiB"
              % (ps.get("hits"), ps.get("misses"), ps.get("trims"),
                 100.0 * ps.get("hit_rate", 0.0),
                 ps.get("cap_bytes", 0) / float(1 << 30),
                 ps.get("peak_bytes", 0) / float(1 << 30)))
    return pol


def table_c(d5, d2, top=24):
    a5, a2 = per_label(d5), per_label(d2)
    wall5 = d5["step_s_wall_mean"] * 1000.0
    wall2 = d2["step_s_wall_mean"] * 1000.0
    print()
    print("=" * 122)
    print("表 C  seq512 → seq2048（token ×4）。增长倍数按「每步 ms」算；纯按 token 线性放大应≈4.0×")
    print("=" * 122)
    print("步时：seq512 %.1f ms  →  seq2048 %.1f ms    （×%.2f）"
          % (wall5, wall2, wall2 / wall5))
    print("-" * 122)
    print("%-30s %10s %10s %8s %9s %9s" % ("kernel / leaf", "512 ms", "2048 ms", "×",
                                           "×/4", "次/步"))
    print("-" * 122)
    rows = []
    for lbl, v2 in a2.items():
        v5 = a5.get(lbl)
        ms5 = v5["ms"] if v5 else 0.0
        if v2["ms"] < 0.05 and ms5 < 0.05:
            continue
        ratio = (v2["ms"] / ms5) if ms5 > 1e-9 else float("inf")
        rows.append((ratio, lbl, ms5, v2["ms"], v2["n"]))
    rows.sort(key=lambda r: -r[0])
    for ratio, lbl, ms5, ms2, n in rows[:top]:
        rs = "  inf" if ratio == float("inf") else "%5.2f" % ratio
        ex = "  inf" if ratio == float("inf") else "%5.2f" % (ratio / 4.0)
        print("%-30s %10.3f %10.3f %8s %9s %9.0f"
              % (lbl[:30], ms5, ms2, rs, ex, n))
    tot5 = sum(v["ms"] for v in a5.values())
    tot2 = sum(v["ms"] for v in a2.values())
    print("-" * 122)
    print("%-30s %10.3f %10.3f %8.2f %9.2f" % ("【叶子合计】", tot5, tot2,
                                               (tot2 / tot5) if tot5 else 0.0,
                                               (tot2 / tot5 / 4.0) if tot5 else 0.0))
    print()
    print("按阶段对比（event 段跨度）：")
    print("%-12s %10s %10s %8s %10s" % ("阶段", "512 ms", "2048 ms", "×", "增ms/步"))
    for k in PHASES:
        v5 = d5["phases"].get(k)
        v2 = d2["phases"].get(k)
        if not v5 or not v2:
            continue
        print("%-12s %10.3f %10.3f %8.2f %10.3f"
              % (k, v5["ms_per_step"], v2["ms_per_step"],
                 v2["ms_per_step"] / v5["ms_per_step"],
                 v2["ms_per_step"] - v5["ms_per_step"]))
    print("%-12s %10.3f %10.3f %8.2f %10.3f"
          % ("步时(墙钟)", wall5, wall2, wall2 / wall5, wall2 - wall5))
    return rows


def _elems(label, ints):
    """从「唯一的那个 int 参数」还原元素数。

    `*_v4` 系列（float4 向量化）传的是 **n4 = 元素数/4**（见 cuda.py `_vec4_grid`
    调用点 `(n >> 2, c_int)`），其余逐元素内核传的就是元素数本身。
    """
    n = int(ints[0])
    return n * 4 if label.endswith("_v4") else n


def table_d(d, top=16):
    """热点 kernel 的单位成本与隐含带宽（只看表 A 里的热点行）。"""
    wall_ms = d["step_s_wall_mean"] * 1000.0
    print()
    print("=" * 132)
    print("表 D（补充）  热点 kernel 的单次成本 / 隐含带宽  —— 只算「恰好 1 个 int 参数且有指针」的逐元素类")
    print("=" * 132)
    print("%-26s %-20s %8s %8s %9s %10s %9s %8s"
          % ("kernel", "shape", "次/步", "ms/步", "µs/次", "隐含MB/次", "有效GB/s", "占900"))
    print("-" * 132)
    rows = []
    for r in d["ops"][:80]:
        lbl = r["label"]
        if lbl.startswith("cublas") or lbl.startswith("mem:"):
            continue
        smp = (d["samples"].get(lbl) or {}).get(str(tuple(r["aux"])))
        if not smp:
            continue
        ints = smp.get("ints") or []
        nptr = smp.get("nptr") or 0
        if len(ints) != 1 or nptr < 2:
            continue
        n = _elems(lbl, ints)
        mb = 4.0 * n * nptr / 1e6
        us = 1000.0 * r["ms_per_step"] / r["n_per_step"] if r["n_per_step"] else 0.0
        # GB/s = (mb × 1e6 B) / (us × 1e-6 s) / 1e9 = mb × 1000 / us
        gbs = (mb * 1000.0 / us) if us > 0 else 0.0
        rows.append((gbs, lbl, r["aux"], r["n_per_step"], r["ms_per_step"], us, mb))
    rows.sort(key=lambda x: x[0])
    for gbs, lbl, aux, nps, ms, us, mb in rows[:top]:
        print("%-26s %-20s %8.1f %8.3f %9.2f %10.1f %9.1f %7.1f%%"
              % (lbl[:26], ("n=%d" % (aux[0] if aux else 0))[:20],
                 nps, ms, us, mb, gbs, 100.0 * gbs / PEAK_GBS))
    print("-" * 132)
    print("口径：MB/次 = 4 × 元素数 × 指针数（读+写的下限估计，不含重复读）；"
          "`*_v4` 的 n 已按 ×4 还原")
    print("参考：单次 GPU 时间 <~10 µs 的叶子会被 event 间隔的抖动放大，"
          "所以 µs/次 很小的行不要按带宽解读")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    if a.reuse:
        with open(OUTS[512], encoding="utf-8") as f:
            d5 = json.load(f)
        with open(OUTS[2048], encoding="utf-8") as f:
            d2 = json.load(f)
        d5["trainer_step_s_steady"] = None
        d2["trainer_step_s_steady"] = None
    else:
        d2 = run_one(2048)
        d5 = run_one(512)
    print()
    print("训练器自报 step_s_steady：seq512=%s  seq2048=%s（判断 profiling 有没有拖慢）"
          % (d5.get("trainer_step_s_steady"), d2.get("trainer_step_s_steady")))
    table_a(d2)
    table_b(d2)
    table_b2(d2)
    table_b3(d2)
    table_b4(d2)
    table_c(d5, d2)
    table_d(d2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
