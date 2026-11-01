# -*- coding: utf-8 -*-
"""微调扫参：跑一组配置，抽日志里的关键指标，输出 CSV + Markdown 表。

为什么要这个脚本：R12 的结论是"学习率相对这批数据偏高（gnorm 中位数 50~63）"，
但那是**推断**。要定论就得一次只动一个变量地扫，并且把 gnorm 一起记下来 ——
gnorm 中位数是判断学习率是否合适最直接的指标，比盯着 loss 曲线快得多。

用法：
    python tools/sft_sweep.py --data D:/x/sft_big.jsonl --tokenizer ... \
        --model /mnt/workspace/qwen25 --device cuda --steps 600 \
        --grid lr=5e-6,1e-5,2e-5 --batch 4 --out sweep.csv

    # 先看要跑什么，不真跑
    python tools/sft_sweep.py ... --dry-run

    # CPU 冒烟（用 --tiny，验证脚本本身能跑通）
    python tools/sft_sweep.py --tiny --device cpu --steps 6 --grid lr=1e-3,3e-3
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_BASE = re.compile(r"探针基线（训练前，留出集平均）=\s*([0-9.]+)")
RE_FINAL = re.compile(r"探针 loss：\s*([0-9.]+)\s*→\s*([0-9.]+)（变化\s*([0-9.]+)%）")
RE_TINY = re.compile(r"探针 loss（固定序列，可比）：\s*([0-9.]+)\s*→\s*([0-9.]+)")
RE_GNORM = re.compile(r"gnorm 统计：中位数\s*([0-9.]+)\s*/\s*p90\s*([0-9.]+)")
RE_STEP = re.compile(r"([0-9.]+)\s*s/step")
RE_VERDICT = re.compile(r"结论：(PASS|FAIL|未学到|过拟合|发散)")


def parse_log(text, steps):
    """从一次运行的 stdout 里抽指标。抽不到的项留 None，不猜。"""
    out = {"base": None, "final": None, "delta_pct": None,
           "gnorm_med": None, "sec_per_step": None, "verdict": None}
    m = RE_BASE.search(text)
    if m:
        out["base"] = float(m.group(1))
    m = RE_FINAL.search(text)
    if m:
        out["base"] = out["base"] if out["base"] is not None else float(m.group(1))
        out["final"] = float(m.group(2))
        out["delta_pct"] = -float(m.group(3))       # 日志里是"变化 N%"，降为负
    else:
        m = RE_TINY.search(text)
        if m:                                        # tiny 模式没有基线行
            out["base"], out["final"] = float(m.group(1)), float(m.group(2))
            if out["base"]:
                out["delta_pct"] = 100.0 * (out["final"] - out["base"]) / out["base"]
    m = RE_GNORM.search(text)
    if m:
        out["gnorm_med"] = float(m.group(1))
    sp = RE_STEP.findall(text)
    if sp:
        out["sec_per_step"] = float(sp[-1])
    m = RE_VERDICT.search(text)
    if m:
        out["verdict"] = m.group(1)
    elif out["delta_pct"] is not None:
        out["verdict"] = "PASS" if out["delta_pct"] <= -5.0 else "未学到"
    return out


def build_cmd(base, lr, batch, steps, tiny=False):
    cmd = [sys.executable, "-X", "utf8", os.path.join(REPO, "examples", "qwen25_sft.py")]
    # ★ tiny 分支读的是 --tiny-lr，不是 --lr。传错了会发现"不同 lr 结果完全一样"
    #   （实测踩过：四个组合输出一模一样，因为 lr 根本没生效）。
    lr_flag = "--tiny-lr" if tiny else "--lr"
    cmd += base + [lr_flag, repr(lr), "--batch", str(batch), "--steps", str(steps)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="lr=2e-5", help="逗号分隔的 lr 列表，如 lr=5e-6,1e-5")
    ap.add_argument("--batch", default="4", help="逗号分隔的 batch 列表")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--model", default=None, help="safetensors 目录")
    ap.add_argument("--tg", default=None, help=".TG 权重（与 --model 二选一）")
    # 别名：与 examples/qwen25_sft.py 的参数名一致。
    # 实测踩过：守护脚本按示例的写法传 --safetensors，这里只有 --model，
    # 于是 argparse 直接报 unrecognized arguments —— 白等 5 小时。
    ap.add_argument("--safetensors", default=None, help="同 --model（别名）")
    # 默认 256：实测 sft_big 总长 p90=197 / p99=278，seq_len=128 会截断 **27.6%**
    # 的样本（回答被从中间切断），256 降到 2.1%。截断率是"学不动"的常见隐形原因。
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tiny", action="store_true", help="CPU 冒烟：用随机迷你权重")
    ap.add_argument("--out", default="sft_sweep.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    lrs = [float(x) for x in a.grid.split("=", 1)[1].split(",")]
    batches = [int(x) for x in a.batch.split(",")]

    if a.safetensors and not a.model:
        a.model = a.safetensors
    if not a.model and not a.tg and not a.tiny:
        print("非 --tiny 模式必须给 --model（safetensors 目录）或 --tg（.TG 文件）")
        return 2
    base = []
    if a.tiny:
        base = ["--tiny", "--device", a.device]
    else:
        if not (a.data and a.tokenizer and (a.model or a.tg)):
            print("非 --tiny 模式必须给 --data/--tokenizer/（--model 或 --tg）")
            return 2
        model_args = ["--tg", a.tg] if a.tg else ["--safetensors", a.model]
        base = model_args + ["--data", a.data, "--tokenizer", a.tokenizer,
                "--seq-len", str(a.seq_len), "--limit", str(a.limit),
                "--holdout", str(a.holdout), "--device", a.device,
                "--warmup", str(a.warmup), "--clip", repr(a.clip)]

    rows = []
    for lr, batch in itertools.product(lrs, batches):
        cmd = build_cmd(base, lr, batch, a.steps, tiny=a.tiny)
        print("\n" + "=" * 70)
        print("lr=%g batch=%d steps=%d" % (lr, batch, a.steps))
        print("$ " + " ".join(cmd))
        if a.dry_run:
            continue
        t0 = time.time()
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        text = (p.stdout or "") + (p.stderr or "")
        m = parse_log(text, a.steps)
        row = {"lr": lr, "batch": batch, "steps": a.steps, "wall_s": round(time.time() - t0, 1),
               **m}
        rows.append(row)
        print("  → base=%s final=%s Δ=%s%% gnorm_med=%s %ss/step 判定=%s"
              % (m["base"], m["final"],
                 "None" if m["delta_pct"] is None else "%.1f" % m["delta_pct"],
                 m["gnorm_med"], m["sec_per_step"], m["verdict"]))
        if p.returncode != 0 and m["final"] is None:
            tail = "\n".join([l for l in text.strip().split("\n") if l.strip()][-6:])
            print("  运行失败（rc=%d），末尾：\n%s" % (p.returncode, tail))

    if a.dry_run:
        print("\n（--dry-run：没有真跑）")
        return 0
    if not rows:
        return 1

    cols = ["lr", "batch", "steps", "base", "final", "delta_pct", "gnorm_med",
            "sec_per_step", "verdict", "wall_s"]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    print("\n" + "=" * 70)
    print("| lr | batch | 基线 | 结束 | 降幅 | gnorm 中位数 | s/step | 判定 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print("| %g | %d | %s | %s | %s | %s | %s | %s |"
              % (r["lr"], r["batch"],
                 "-" if r["base"] is None else "%.4f" % r["base"],
                 "-" if r["final"] is None else "%.4f" % r["final"],
                 "-" if r["delta_pct"] is None else "%.1f%%" % r["delta_pct"],
                 "-" if r["gnorm_med"] is None else "%.2f" % r["gnorm_med"],
                 "-" if r["sec_per_step"] is None else "%.2f" % r["sec_per_step"],
                 r["verdict"] or "-"))
    print("CSV 已写：%s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
