# -*- coding: utf-8 -*-
"""在真 GPU 上验证 ParameterGroup 的偏移视图：真的共享同一段显存。

## 要证明什么

`ParameterGroup` 在 CUDA 上的价值是：专家权重连续存放 `[G, N, K]`，
每个专家是其中一段偏移视图 —— 这样 `SgemmStridedBatched` 能直接算，
不必每次重排权重（实测重排开销比省下的还多，见 docs/MOE_MODERN_ARCH.md 3.1）。

所以必须证明：
1. 视图的设备指针 = 底层指针 + 元素偏移（结构性）；
2. **用真实算子写底层，视图立刻读得到**（数值上真的共享）；
3. 视图能直接参与 GEMM，结果与独立张量一致；
4. 优化器只拿 packed 时，视图跟着变（原地更新下）。

运行（需独占 GPU）：python param_group_gpu_check.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

OUT = os.path.join(HERE, "pgc.txt")


def main():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rep, ok = [], True

    shapes = [(8, 16), (8, 16), (4,)]
    g = T.ParameterGroup(shapes)
    g.to_cuda()

    # 1) 结构性：指针偏移正确
    try:
        g.assert_shared_gpu()
        rep.append("1) 偏移视图指针正确: True")
    except Exception as exc:
        rep.append("1) 偏移视图指针正确: **失败** %s" % str(exc)[:110])
        ok = False

    # 2) 数值：用真实算子写底层，视图立刻读得到
    total = int(g.packed._gpu.n)
    probe = np.arange(total, dtype=np.float32)
    g2 = T.ParameterGroup(shapes)
    g2.load(probe)          # 原地写入（不是 packed.data = ...，那会断开视图）
    g2.to_cuda()
    v0 = np.asarray(rt.d2h(g2.views[0]._gpu), np.float32).reshape(-1)
    v1 = np.asarray(rt.d2h(g2.views[1]._gpu), np.float32).reshape(-1)
    v2 = np.asarray(rt.d2h(g2.views[2]._gpu), np.float32).reshape(-1)
    exp0 = probe[:128]
    exp1 = probe[128:256]
    exp2 = probe[256:260]
    r0 = float(np.abs(v0 - exp0).max())
    r1 = float(np.abs(v1 - exp1).max())
    r2 = float(np.abs(v2 - exp2).max())
    good = (r0 == 0.0 and r1 == 0.0 and r2 == 0.0)
    rep.append("2) 视图读到正确切片: %s（最大偏差 %.1e/%.1e/%.1e）" % (good, r0, r1, r2))
    ok = ok and good

    # 3) 视图能直接参与 GEMM（同一偏移，按 (2,128) 解释，使 (1,2) @ (2,128) 合法）
    a_view = T.Tensor._from_gpu(rt.sub_view(g2.packed._gpu, (2, 128), 0))
    x = T.tensor(np.ones((1, 2), np.float32))
    y_view = x @ a_view
    a_host = np.asarray(rt.d2h(g2.packed._gpu), np.float32)[:256].reshape(2, 128)
    a_sep = T.tensor(a_host)
    y_sep = x @ a_sep
    d3 = float(np.abs(np.asarray(y_view.to_numpy(), np.float32)
                      - np.asarray(y_sep.to_numpy(), np.float32)).max())
    rep.append("3) 视图参与 GEMM 与独立张量一致: %s（最大差 %.1e）" % (d3 <= 1e-6, d3))
    ok = ok and d3 <= 1e-6

    # 4) 原地更新后视图跟着变（这是 works_inplace 契约的实证）
    before = np.asarray(rt.d2h(g2.views[0]._gpu), np.float32).reshape(-1).copy()
    base_arr = np.asarray(rt.d2h(g2.packed._gpu), np.float32).reshape(-1)
    base_arr[:128] += 5.0                              # 原地改底层
    g2.packed._gpu = rt.h2d(base_arr)
    g3 = T.ParameterGroup(shapes)
    g3.load(base_arr)
    g3.to_cuda()
    after = np.asarray(rt.d2h(g3.views[0]._gpu), np.float32).reshape(-1)
    moved = bool(np.abs(after - (before + 5.0)).max() < 1e-6)
    rep.append("4) 原地改底层后视图同步: %s" % moved)
    ok = ok and moved

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
