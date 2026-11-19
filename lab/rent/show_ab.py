# -*- coding: utf-8 -*-
"""汇总 ab_route_2d.py 三次运行的每步步时与 loss（本地读日志）。"""
import io
import re
import statistics

RUNS = [
    ("baseline（原样）", r"lab\rent\_ab_off.log"),
    ("路由：投影（不含 dW）", r"lab\rent\_ab_on_notn.log"),
    ("路由：投影 + dW", r"lab\rent\_ab_on.log"),
]
PAT = re.compile(r"step\s+(\d+)/\d+ loss=([0-9.]+) lr=[0-9.e-]+ gnorm=\s*([0-9.]+) "
                 r"([0-9.]+)s")
rows = {}
for name, path in RUNS:
    raw = open(path, "rb").read()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    text = raw.decode(enc, "replace")
    st, lo, gn = {}, {}, {}
    for line in text.splitlines():
        m = PAT.search(line)
        if m:
            i = int(m.group(1))
            st[i] = float(m.group(4))
            lo[i] = float(m.group(2))
            gn[i] = float(m.group(3))
    rows[name] = (st, lo, gn)
    print("%-22s n=%d  steady(1..19) 中位 %.4f s  最小 %.4f  最大 %.4f"
          % (name, len(st), statistics.median([st[i] for i in st if i >= 1]),
             min(st[i] for i in st if i >= 1), max(st[i] for i in st if i >= 1)))

base = rows["baseline（原样）"][0]
bmed = statistics.median([base[i] for i in base if i >= 1])
print("\n%-22s %-9s %-9s %-9s %-9s" % ("run", "中位 s", "Δ vs base", "loss@19", "gnorm@19"))
for name, (st, lo, gn) in rows.items():
    med = statistics.median([st[i] for i in st if i >= 1])
    print("%-22s %-9.4f %+9.4f %-9.6f %-9.4f"
          % (name, med, med - bmed, lo.get(19, float("nan")), gn.get(19, float("nan"))))

print("\nloss 逐步差（同一 step 相对 baseline）")
print("%-6s %-12s %-12s %-12s" % ("step", "base", "proj-only", "proj+dW"))
a = rows["baseline（原样）"][1]
b = rows["路由：投影（不含 dW）"][1]
c = rows["路由：投影 + dW"][1]
for i in range(0, 20):
    if i in a and i in b and i in c:
        print("%-6d %-12.6f %+.2e     %+.2e" % (i, a[i], b[i] - a[i], c[i] - a[i]))
