# -*- coding: utf-8 -*-
"""扫描一个 PyTorch 项目，统计它用到哪些 torch API，并对照 ORV 是否支持。

用法：python tools/analyze_torch_usage.py <项目根>

输出三张表：
  ① ORV 直接有（改 import 即可）
  ② ORV 有但名字/签名不同（要适配层）
  ③ ORV 没有（必须写插件算子）

这是"能不能用 ORV 跑别人项目"的**可行性盘点**，不改对方任何代码。
"""
import io
import os
import re
import sys
from collections import Counter

PATTERNS = [
    (re.compile(r"\btorch\.nn\.functional\.([A-Za-z_]\w*)"), "torch.nn.functional"),
    (re.compile(r"\bnn\.functional\.([A-Za-z_]\w*)"), "torch.nn.functional"),
    (re.compile(r"\bF\.([A-Za-z_]\w*)"), "torch.nn.functional"),
    (re.compile(r"\btorch\.nn\.([A-Za-z_]\w*)"), "torch.nn"),
    (re.compile(r"\bnn\.([A-Za-z_]\w*)"), "torch.nn"),
    (re.compile(r"\btorch\.optim\.([A-Za-z_]\w*)"), "torch.optim"),
    (re.compile(r"\btorch\.linalg\.([A-Za-z_]\w*)"), "torch.linalg"),
    (re.compile(r"\btorch\.([A-Za-z_]\w*)"), "torch"),
]

# ORV 已有、但常常不是同名同签名的
ORV_ALIAS = {
    "torch.nn.functional.softmax", "torch.nn.functional.relu",
    "torch.nn.functional.sigmoid", "torch.nn.functional.tanh",
}


def orv_namespaces():
    import tgai_go.train as T
    ns = {
        "torch": set(dir(T)),
        "torch.nn": set(dir(T.nn)),
        "torch.nn.functional": set(dir(T.nn.functional)),
        "torch.optim": set(dir(T.optim)),
        "torch.linalg": set(dir(getattr(T, "linalg", object()))),
    }
    return ns


def orv_pool():
    """所有命名空间的名字并集 —— 用来判断"ORV 到底有没有这个算子"。

    只看 `dir(T)` 会严重高估缺失：很多算子在 `T.functional` 里而不在顶层。
    """
    import tgai_go.train as T
    pool = {}
    for label, obj in (("T", T), ("T.functional", getattr(T, "functional", None)),
                       ("T.nn", T.nn), ("T.nn.functional", T.nn.functional),
                       ("T.optim", T.optim)):
        if obj is None:
            continue
        for name in dir(obj):
            if not name.startswith("_"):
                pool.setdefault(name, label)
    return pool


def scan(root):
    hits = Counter()
    files = 0
    for dirpath, _, names in os.walk(root):
        if any(p in dirpath for p in ("__pycache__", ".git")):
            continue
        for fn in names:
            if not fn.endswith(".py"):
                continue
            files += 1
            try:
                src = io.open(os.path.join(dirpath, fn), encoding="utf-8",
                              errors="replace").read()
            except Exception:
                continue
            # 去掉注释和字符串里的假命中（粗略但够用）
            lines = [ln.split("#", 1)[0] for ln in src.splitlines()]
            body = "\n".join(lines)
            for pat, ns in PATTERNS:
                for m in pat.finditer(body):
                    name = m.group(1)
                    hits["%s.%s" % (ns, name)] += 1
    return files, hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files, hits = scan(root)
    ns = orv_namespaces()
    pool = orv_pool()

    have, elsewhere, missing = [], [], []
    for full, n in hits.most_common():
        prefix, _, name = full.rpartition(".")
        if name in ns.get(prefix, set()):
            have.append((full, n))
        elif name in pool:
            elsewhere.append((full, n, pool[name]))
        else:
            missing.append((full, n))

    print("扫描 %d 个 .py 文件，命中 %d 个不同的 torch API 名\n" % (files, len(hits)))
    print("① 同名同位（改 import 即可）：%d 个" % len(have))
    for full, n in have[:12]:
        print("   %-42s %d 次" % (full, n))
    print()
    print("② ORV 有、但在别的命名空间（写别名即可）：%d 个" % len(elsewhere))
    for full, n, where in elsewhere[:24]:
        print("   %-42s %d 次   -> %s" % (full, n, where))
    print()
    print("③ ORV 完全没有（必须插件补）：%d 个" % len(missing))
    for full, n in missing[:40]:
        print("   %-42s %d 次" % (full, n))
    if len(missing) > 40:
        print("   ... 另有 %d 个" % (len(missing) - 40))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
