# -*- coding: utf-8 -*-
"""融合两段 cast（`cast2_f32_to_f16`）的等价性与接线（2026-11-21）。

背景：`_sgemm_f16` 的两个操作数**都不是参数**时（MoE 的 dW 族：`x` 和 `dy` 都是激活），
原来各发一次 `cast_f32_to_f16`。这台机器上**一次发射值 11.7 µs**
（985 次 cast/步 ⇒ 11.6 ms/步；`PROBE_SKIP` 阶梯消融测得，见
`lab/rent/REPORT_amp_gemm2_20261121.md`），所以把两段并成一次是纯赚。

守两件事：

1. **接线**：`_sgemm_f16` 在"两个操作数都没命中影子缓存"时必须走
   `_cast2_flat_ptr_into`，不许退回连发两次。
2. **逐位等价**：融合版与"连调两次 `_cast_flat_ptr_into`"必须**逐位相同**
   （同一个 `tg_f32_to_f16_bits`、同一套下标），并且与 numpy 的 float16（RNE）一致。

第 2 条需要 GPU（`TG_RUN_CUDA_TESTS=1`），本机（无 CUDA）必 skip；
在 RTX 5090 上用 `lab/rent/probe_cast2_gpu.py` 实跑过 5 组长度，ALL PASS。
"""
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_PY = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")
sys.path.insert(0, os.path.join(ROOT, "python"))

CUDA_ON = bool(os.environ.get("TG_RUN_CUDA_TESTS"))


def _body(name, span=2600):
    src = io.open(CUDA_PY, encoding="utf-8").read()
    i = src.index("def %s" % name)
    seg = src[i:i + span]
    a = seg.find('"""')
    if a >= 0:
        b = seg.find('"""', a + 3)
        if b > a:
            seg = seg[:a] + seg[b + 3:]
    return seg


def test_sgemm_f16_fuses_the_two_operand_casts():
    src = io.open(CUDA_PY, encoding="utf-8").read()
    assert "cast2_f32_to_f16" in src, "融合两段 cast 的内核不见了"
    body = _body("_sgemm_f16", 3400)
    assert "_cast2_flat_ptr_into(A, na, a16, B, nb, b16)" in body, (
        "`_sgemm_f16` 在两个操作数都要现 cast 时没有走融合发射"
        "（会多付一次约 11.7 µs 的发射）")


@pytest.mark.skipif(not CUDA_ON, reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1")
def test_fused_cast_is_bit_identical():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    for n0, n1 in ((4096, 4096), (4097, 1023), (1, 262144), (524288, 1703936)):
        x0 = rng.normal(0, 3, n0).astype(np.float32)
        x1 = rng.normal(0, 5, n1).astype(np.float32)
        a0, a1 = rt.h2d(x0.reshape(1, n0)), rt.h2d(x1.reshape(1, n1))
        r0 = rt.empty((n0,), dtype=np.float16)
        r1 = rt.empty((n1,), dtype=np.float16)
        rt._cast_flat_ptr_into(a0.ptr, n0, r0)
        rt._cast_flat_ptr_into(a1.ptr, n1, r1)
        s0, s1 = rt.d2h(r0).copy(), rt.d2h(r1).copy()
        f0 = rt.empty((n0,), dtype=np.float16)
        f1 = rt.empty((n1,), dtype=np.float16)
        rt._cast2_flat_ptr_into(a0.ptr, n0, f0, a1.ptr, n1, f1)
        assert rt.d2h(f0).tobytes() == s0.tobytes(), (n0, n1, "segment A differs")
        assert rt.d2h(f1).tobytes() == s1.tobytes(), (n0, n1, "segment B differs")
        assert s0.tobytes() == x0.astype(np.float16).tobytes(), (
            (n0, n1), "cast 与 numpy float16(RNE) 不一致")
