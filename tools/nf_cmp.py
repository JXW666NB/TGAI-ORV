# -*- coding: utf-8 -*-
"""比较两条 loss 曲线（噪声底）：python nf_cmp.py a.json b.json"""
import json
import sys


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    la, lb = a["losses"], b["losses"]
    print("A=%s  B=%s  (nl=%s seq=%s lr=%s steps=%s)"
          % (a["path"], b["path"], a.get("nl"), a.get("seq"), a.get("lr"), a.get("steps")))
    print("%5s %14s %14s %12s %10s" % ("step", "A", "B", "|Δ|", "相对"))
    worst = 0.0
    for i, (x, y) in enumerate(zip(la, lb)):
        d = abs(x - y)
        r = d / max(1e-9, abs(y))
        worst = max(worst, r)
        print("%5d %14.6f %14.6f %12.2e %9.2e" % (i, x, y, d, r))
    print("\n最差相对偏差 = %.2e" % worst)
    print("首步 A=%.6f B=%.6f（应相同：都从冷启动载入同一 .tg）" % (la[0], lb[0]))


if __name__ == "__main__":
    main()
