# -*- coding: utf-8 -*-
"""诊断：Muon 分层更新 vs 统一更新 —— 谁的偏差大、分组是否重叠/漏覆盖。"""
import numpy as np

import tgai_go.train as T

m = T.preset.transformer(vocab=32, d_model=16, n_layers=2, n_heads=2)
ps = list(m.parameters())
sd0 = m.state_dict()
rng = np.random.default_rng(0)
ids = rng.integers(0, 32, (8,), dtype=np.int64)
tgt = rng.integers(0, 32, (8,), dtype=np.int64)

groups = {i: list(m.blocks[i].parameters()) for i in range(2)}
covered = set()
dup = []
for i, g in groups.items():
    for p in g:
        if id(p) in covered:
            dup.append((i, id(p)))
        covered.add(id(p))
allp = {id(p): p for p in ps}
missing = [(id(p), tuple(p.shape)) for p in ps if id(p) not in covered]
print("[group] 总参数 %d，分层覆盖 %d，未覆盖 %d 个：" % (len(ps), len(covered), len(missing)))
for i, shp in missing:
    print("   未覆盖 shape=%s" % (shp,))
print("[group] 组内重复: %s" % (dup or "无",))

# 名字映射，便于读
names = {}
for k, v in m.state_dict().items():
    names[id(v)] = k


def run(per_layer):
    m.load_state_dict(sd0)
    opt = T.optim.Muon(ps, lr=0.01, momentum=0.9)
    if per_layer:
        opt.set_layer_groups({i: list(m.blocks[i].parameters()) for i in range(2)})
    m(ids, tgt).backward()
    if per_layer:
        gl = opt.all_groups()
        print("[run] all_groups() = %r" % (gl,))
        for g in gl:
            opt.step_layers([g], t=1, release=False)
    else:
        opt.step()
    return m.state_dict()


a, b = run(False), run(True)
rows = []
for k in a:
    x, y = np.asarray(a[k], np.float32), np.asarray(b[k], np.float32)
    v0 = np.asarray(sd0[k], np.float32)
    if x.size == 0:
        continue
    dev = float(np.abs(x - y).max())
    upd = float(np.abs(x - v0).max())
    rows.append((dev, upd, dev / upd if upd > 0 else float("inf"), k, tuple(x.shape)))
rows.sort(reverse=True)
print("\n[diag] 按绝对偏差排序（dev=分层-统一, upd=本张量更新量）")
print("   %-34s %-12s %10s %10s %8s" % ("key", "shape", "dev", "upd", "dev/upd"))
for dev, upd, r, k, shp in rows[:8]:
    print("   %-34s %-12s %10.3e %10.3e %7.2f%%" % (k, str(shp), dev, upd, 100 * r))
gmax = max(r[1] for r in rows)
gdev = max(r[0] for r in rows)
print("\n[diag] 全局：最大偏差 %.3e，最大更新量 %.3e（占比 %.2f%%）"
      % (gdev, gmax, 100 * gdev / gmax))
