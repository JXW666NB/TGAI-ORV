# -*- coding: utf-8 -*-
"""验证设备侧 permute/unpermute 内核：与参考实现（主机排序 + 逐槽 gather）对拍。

参考实现就是当前 `moe_topk_orv` 的做法：argsort 分组、逐专家切片、再散回。
本探针只比 **permute/unpermute 这一层**，不含专家 GEMM。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, k, d, E = 2048, 2, 512, 8
    rng = np.random.default_rng(0)
    eids_h = rng.integers(0, E, (N, k)).astype(np.int64)
    x_h = rng.normal(0, .3, (N, d)).astype(np.float32)
    g_h = rng.uniform(0.1, 1.0, (N, k)).astype(np.float32)
    C = int(np.ceil(N * k / E * 1.5))          # 容量 = 1.5 倍均值

    # ---- 设备侧 ----
    eids = rt.h2d(eids_h)
    x = rt.h2d(x_h)
    gate = rt.h2d(g_h)
    counts = rt.zeros((E,), dtype=np.int32)
    xp = rt.zeros((E * C, d), dtype=np.float32)
    gp = rt.zeros((E * C,), dtype=np.float32)
    rowid = rt.full((E * C,), -1, dtype=np.int32)
    rt.moe_permute(eids, x, gate, counts, xp, gp, rowid, N, k, d, E, C)
    rt.sync()

    xp_np = xp.to_numpy().reshape(E, C, d)
    gp_np = gp.to_numpy().reshape(E, C)
    rowid_np = rowid.to_numpy().reshape(E, C)
    counts_np = counts.to_numpy()
    dropped = int(np.maximum(counts_np - C, 0).sum())
    print("容量 C=%d，被丢弃的 (row,slot) 数=%d" % (C, dropped))

    # ---- 参考实现 ----
    ref_xp = np.zeros((E, C, d), np.float32)
    ref_gp = np.zeros((E, C), np.float32)
    ref_rowid = -np.ones((E, C), np.int64)
    seen = np.zeros(E, np.int64)
    for row in range(N):
        for slot in range(k):
            e = eids_h[row, slot]
            j = seen[e]
            seen[e] += 1
            if j < C:
                ref_xp[e, j] = x_h[row]
                ref_gp[e, j] = g_h[row, slot]
                ref_rowid[e, j] = row * k + slot

    # 槽位顺序由 atomicAdd 决定，与串行参考不同 —— 所以按**集合**比较，不按位置
    ok_id = True
    for e in range(E):
        got = np.sort(rowid_np[e][rowid_np[e] >= 0])
        want = np.sort(ref_rowid[e][ref_rowid[e] >= 0])
        if got.shape != want.shape or not np.array_equal(got, want):
            ok_id = False
    ok_g = True
    for e in range(E):
        for j in range(C):
            r = rowid_np[e, j]
            if r < 0:
                continue
            if abs(float(gp_np[e, j]) - float(g_h[int(r) // k, int(r) % k])) > 0:
                ok_g = False
    ok_cnt = np.array_equal(np.minimum(counts_np, C),
                            np.array([min(int((eids_h == e).sum()), C)
                                      for e in range(E)]))
    # 逐槽对齐后比 xp（顺序不同也能比）
    max_dx = 0.0
    for e in range(E):
        for j in range(C):
            r = rowid_np[e, j]
            if r < 0:
                continue
            row = int(r) // k
            max_dx = max(max_dx, float(np.abs(xp_np[e, j] - x_h[row]).max()))
    print("rowid 集合一致: %s；gate 一致: %s；counts 一致: %s；xp 最大差: %.3e"
          % (ok_id, ok_g, ok_cnt, max_dx))

    # ---- unpermute ----
    yp_h = rng.normal(0, .3, (E * C, d)).astype(np.float32)
    yp = rt.h2d(yp_h)
    out = rt.zeros((N * k, d), dtype=np.float32)
    rt.moe_unpermute(yp, rowid, out, d, E * C)
    rt.sync()
    out_np = out.to_numpy()
    ref_out = np.zeros((N * k, d), np.float32)
    for e in range(E):
        for j in range(C):
            r = rowid_np[e, j]
            if r >= 0:
                ref_out[int(r)] = yp_h[e * C + j]
    print("unpermute 最大差: %.3e" % float(np.abs(out_np - ref_out).max()))

    # ---- 反向 ----
    dyp = rt.h2d(rng.normal(0, .3, (E * C, d)).astype(np.float32))
    dyp_h = dyp.to_numpy()
    dx = rt.zeros((N, d), dtype=np.float32)
    dgate = rt.zeros((N * k,), dtype=np.float32)
    rt.moe_permute_bwd(dyp, x, rowid, dx, dgate, d, k, E * C)
    rt.sync()
    dx_np = dx.to_numpy()
    dg_np = dgate.to_numpy()

    ref_dx = np.zeros((N, d), np.float32)
    ref_dg = np.zeros((N * k,), np.float32)
    for e in range(E):
        for j in range(C):
            r = rowid_np[e, j]
            if r < 0:
                continue
            row = int(r) // k
            ref_dx[row] += dyp_h[e * C + j]
            ref_dg[int(r)] = float(dyp_h[e * C + j] @ x_h[row])
    print("反向 dx 最大差: %.3e；dgate 最大差: %.3e"
          % (float(np.abs(dx_np - ref_dx).max()), float(np.abs(dg_np - ref_dg).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
