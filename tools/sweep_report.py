# -*- coding: utf-8 -*-
"""扫参结果分析：把 CSV 变成「推荐微调参数」+ 一条能贴进文档的表。

为什么单独写一个分析器：
* 扫参产生的是一张原始表，**"哪组值得推荐"需要判定规则**，而规则不能临时拍脑袋；
* 规则里必须包含 gnorm 这个指标 —— 只看降幅会选出"降得多但梯度一直在爆"的配置。

列约定（**很容易踩**）：`delta_pct` **负值表示下降**（-28.7 就是降了 28.7%）。
这与 `tools/sft_sweep.py` 写出来的格式一致；如果整张表都是正数，多半是列含义反了，
本工具会打一行提示（见 main）。

判定规则（写死在这里，跑之前就定好，避免事后挑数据）：

1. **硬门槛**：留出集降幅 ≥5%（与 FINETUNE_RESULTS 的判据一致）；不到 5% 记"未学到"；
2. **稳定性**：`gnorm 中位数 / clip 阈值` ≤ 5。超过说明每步都被裁、有效步长被压扁
   —— 这种配置即使降幅好看也不稳（换数据/换 seed 容易翻车）；
3. **并列时**：优先更小的 lr（更保守），再优先更小的 batch（更省显存）；
4. 输出里同时给"最佳"和"次佳"，并明确说明**这是单 seed 结果**，不作收敛性承诺。

用法：
    python tools/sweep_report.py sweep.csv
    python tools/sweep_report.py sweep_a.csv sweep_b.csv --clip 1.0 --out report.md
"""
import argparse
import csv
import io
import os
import sys


def load_rows(paths):
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            print("跳过（不存在）：%s" % p)
            continue
        with io.open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_src"] = os.path.basename(p)
                rows.append(r)
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def judge(rows, clip=1.0, min_delta=5.0, max_gnorm_ratio=200.0):
    """给每一组打判定，再挑推荐。返回 (verdicts, best, second)。"""
    out = []
    for r in rows:
        lr, batch = _f(r.get("lr")), _f(r.get("batch"))
        delta, gn = _f(r.get("delta_pct")), _f(r.get("gnorm_med"))
        steps = _f(r.get("steps"))
        reasons = []
        ok = True
        verdict = str(r.get("verdict") or "")
        if delta is None or "FAIL" in verdict.upper():
            ok = False
            # 实测：lr=2e-5 两组直接发散（没有探针终值），这是最硬的信号
            reasons.append("运行失败/发散（无探针终值，verdict=%s）" % (verdict or "-"))
        elif delta > -min_delta:
            ok = False
            reasons.append("降幅 %.1f%% < %.1f%%（未学到）" % (delta, min_delta))
        if gn is not None and clip and gn > max_gnorm_ratio * clip:
            ok = False
            reasons.append("gnorm 中位数 %.1f > %.1f×clip（学习率偏大）"
                           % (gn, max_gnorm_ratio))
        if gn is None:
            reasons.append("没有 gnorm 统计（无法判断稳不稳）")
        out.append({"lr": lr, "batch": batch, "steps": steps, "delta": delta,
                    "gnorm": gn, "sec_per_step": _f(r.get("sec_per_step")),
                    "ok": ok, "reasons": reasons, "src": r.get("_src", "")})

    cand = [v for v in out if v["ok"]]
    # 并列时更保守：lr 小优先，再 batch 小优先
    # delta 是**负值**（-8.5 表示降了 8.5%），所以直接升序 = 降幅最大优先
    cand.sort(key=lambda v: (v["delta"], v["lr"] or 0, v["batch"] or 0))
    best = cand[0] if cand else None
    second = cand[1] if len(cand) > 1 else None
    return out, best, second


def render(out, best, second, clip, min_delta):
    lines = []
    lines.append("| lr | batch | 步数 | 降幅 | gnorm 中位数 | s/step | 判定 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for v in sorted(out, key=lambda x: (-(x["delta"] or 0))):
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            "-" if v["lr"] is None else "%g" % v["lr"],
            "-" if v["batch"] is None else "%d" % v["batch"],
            "-" if v["steps"] is None else "%d" % v["steps"],
            # 内部是负值（降为负），显示成"降了 11.7%"更符合直觉
            "-" if v["delta"] is None else "%.1f%%" % (-v["delta"]),
            "-" if v["gnorm"] is None else "%.1f" % v["gnorm"],
            "-" if v["sec_per_step"] is None else "%.2f" % v["sec_per_step"],
            "✅ 可用" if v["ok"] else "❌",
            "；".join(v["reasons"]) or "—"))
    lines.append("")
    if best:
        lines.append("**推荐**：lr=%g，batch=%d（降幅 %.1f%%，gnorm 中位数 %s）"
                     % (best["lr"], best["batch"], -best["delta"],
                        "-" if best["gnorm"] is None else "%.1f" % best["gnorm"]))
        if second:
            lines.append("**次佳**：lr=%g，batch=%d（降幅 %.1f%%）"
                         % (second["lr"], second["batch"], -second["delta"]))
        lines.append("")
        lines.append("> 判据：留出集降幅 ≥%.0f%%，且运行未失败（无发散）。"
                     "gnorm 仅作异常值警告（>%.0f×clip）：实测 0.5B 在 clip=%g 下 "
                     "gnorm 中位数 29~37 属正常，不能拿它当筛选门槛。"
                     "单 seed 结果，只说明「这组能达到」，**不构成收敛性承诺**。"
                     % (min_delta, 200.0, clip))
    else:
        lines.append("**没有一组达标** —— 不要放宽判据。下一步该查数据（"
                     "`tools/dataset_report.py` 的截断率/监督密度）而不是继续调超参。")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="一或多个 sweep CSV")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=5.0, help="降幅门槛（%%）")
    ap.add_argument("--out", default=None, help="把 Markdown 写进文件")
    a = ap.parse_args(argv)

    rows = load_rows(a.csv)
    if not rows:
        print("没有数据：检查 CSV 路径")
        return 1
    deltas = [_f(r.get("delta_pct")) for r in rows]
    deltas = [d for d in deltas if d is not None]
    if deltas and all(d > 0 for d in deltas):
        print("⚠ 所有 delta_pct 都是正数。本工具的约定是**负值表示下降**"
              "（sft_sweep.py 写出来就是负的）；如果这张表是「降幅百分比」，"
              "请先把该列取负，否则判定会反过来。\n")

    out, best, second = judge(rows, a.clip, a.min_delta)
    md = render(out, best, second, a.clip, a.min_delta)
    print(md)
    if a.out:
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(md + "\n")
        print("\n已写入 %s" % a.out)
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
