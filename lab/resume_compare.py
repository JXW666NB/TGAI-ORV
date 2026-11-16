# -*- coding: utf-8 -*-
"""比较两次训练 run 的 RESULT json：续训后 loss 是否与不中断一致。"""
import json
import sys


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    la, lb = a.get("losses") or [], b.get("losses") or []
    ra, rb = a.get("step_range"), b.get("step_range")
    print("A step_range=%s  n_losses=%d  first=%.6f last=%.6f"
          % (ra, len(la), la[0] if la else float("nan"), la[-1] if la else float("nan")))
    print("B step_range=%s  n_losses=%d  first=%.6f last=%.6f"
          % (rb, len(lb), lb[0] if lb else float("nan"), lb[-1] if lb else float("nan")))
    # B 从 step s 开始，则 B 的第 i 个 loss 对应 A 的第 (s - ra[0] + i) 个
    s = (rb or [0])[0] - (ra or [0])[0]
    n = min(len(lb), len(la) - s)
    if n <= 0:
        print("无法对齐（步区间不重叠）")
        return
    diffs = [abs(lb[i] - la[s + i]) for i in range(n)]
    print("\n对齐 %d 个 step（B 从 A 的第 %d 个开始）" % (n, s))
    for i in range(n):
        print("  step %d: A=%.8f  B=%.8f  |d|=%.3e"
              % ((ra[0] + s + i), la[s + i], lb[i], diffs[i]))
    print("\nmax|d| = %.3e   末值比 = %.9f"
          % (max(diffs), (lb[-1] / la[-1]) if la[-1] else float("nan")))
    print("判定：%s" % ("续训逐位一致（<1e-6）" if max(diffs) < 1e-6
                     else ("续训有漂移（%.2e）" % max(diffs))))


if __name__ == "__main__":
    main()
