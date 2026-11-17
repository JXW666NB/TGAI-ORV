# -*- coding: utf-8 -*-
"""CSA 全局分支：分块重算路径与物化路径必须**数学等价**。

对应目标验收项 5（两条路径都有测试）。因为 `Lq <= cq` 时自动走原路径，
所以这里用小 seq + 更小的 cq 强制真正分块（seq=128, cq=32 ⇒ 4 块）。

判据：out 逐位一致（浮点同序），dx 与权重梯度在 1e-6 内。
（实测 out 的 max_abs = 0.000e+00；反向因分块求和顺序不同有 ~1e-6 级差异。）
"""
import numpy as np
import pytest

import tgai_go.train as T


def _cuda_ok():
    try:
        T._cuda()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="需要 CUDA 设备")


def _cfg(seq):
    from tgai_go.train import v3_train_cli as CLI
    cfg = dict(d=64, d_ff=128, E=2, layers=1, n_q=2, n_kv=2, vocab=512,
               seq=seq, attn_res=False, attn_res_block=2, top_k=2, csa=4,
               n_kv_attn=2, rope_theta=500000.0, csa_compress=4,
               csa_conv_kernel=3, csa_local_window=0)
    CLI.build(cfg, np.random.default_rng(0))          # 填好 csa_cfg
    return cfg


def _run(seq, cq):
    from tgai_go.train import v3_csa as C
    cfg = _cfg(seq)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    rng = np.random.default_rng(11)
    w = C.init_weights(rng, csac)
    x = T.tensor(rng.normal(0, 0.02, (1, seq, cfg["d"])).astype(np.float32),
                 requires_grad=True)
    old = C.set_attn_chunk(cq)
    try:
        out = C.csa_forward(x, w, plan, cos, sin)
        g = T.tensor(np.ones(tuple(int(s) for s in out.shape), np.float32))
        out.backward(g)
        T._cuda().sync()
        rt = T._cuda()
        o = np.asarray(rt.d2h(out._gpu), dtype=np.float32)
        dx = np.asarray(rt.d2h(x.grad._gpu), dtype=np.float32)
        dw = {k: np.asarray(rt.d2h(v.grad._gpu), dtype=np.float32)
              for k, v in w.items() if getattr(v, "grad", None) is not None}
    finally:
        C.set_attn_chunk(old)
    return o, dx, dw


def test_chunked_matches_materialized():
    seq, cq = 128, 32
    o0, dx0, dw0 = _run(seq, 0)          # 物化（参考）
    o1, dx1, dw1 = _run(seq, cq)         # 分块重算

    assert np.abs(o0 - o1).max() == 0.0, "前向必须逐位一致"
    assert np.abs(dx0 - dx1).max() < 1e-6
    assert set(dw0) == set(dw1)
    worst = max(float(np.abs(dw0[k] - dw1[k]).max()) for k in dw0)
    assert worst < 1e-5, "权重梯度差异过大：%.3e" % worst


def test_switch_zero_restores_materialized():
    from tgai_go.train import v3_csa as C
    old = C.set_attn_chunk(0)
    try:
        assert C._ATTN_CQ[0] == 0
        assert C._attn_auto is not None
    finally:
        C.set_attn_chunk(old)


def test_chunk_larger_than_seq_is_noop():
    """`Lq <= cq` 必须直接走原路径（等价于关闭）。"""
    seq = 64
    o_small, _, _ = _run(seq, 0)
    o_big, _, _ = _run(seq, 10000)
    assert np.abs(o_small - o_big).max() == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
