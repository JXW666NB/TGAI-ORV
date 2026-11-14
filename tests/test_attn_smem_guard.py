# -*- coding: utf-8 -*-
"""Pascal 消费卡的共享内存守卫：不够时要报**能看懂**的错。

## 背景（外部实测，2026-11-12）

GTX 1060（sm_61）上 `scaled_dot_product_attention(..., lowmem=True)` 前向正常、**反向抛**
`cuFuncSetAttribute(attn_bwd_flash maxdyn) rc=1 invalid argument` —— 完全看不出原因。

真因：融合注意力反向要 `_ATTN_FLASH_SMEM = 68 KB`（g2 变体 84 KB），
而 sm_61 单 block 动态 smem 上限只有 **48 KB** 且不支持 opt-in。

现在改成**先查上限再设 opt-in**，报错里写明"为什么"和"怎么办"。

## 怎么验（本机可能没有 Pascal 卡）

把 `_device_smem_limit` 临时改成返回 48 KB 模拟 sm_61，确认抛出的 `CudaError`
里包含上限数值、`修法`、`lowmem=False` 与卡型提示。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import functional as Fn
from tgai_go.train.cuda import CudaError


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


def _qkv(bh, seq, dh, seed=0):
    """直接建 3 维张量并标记 requires_grad —— **必须是叶子**，否则拿不到 .grad。

    （踩过：用 `flat.reshape(...)` 得到的张量不是叶子，反传后 `.grad` 恒为 None。）
    """
    rt = T._cuda()
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(3):
        a = rng.normal(0, 0.3, (bh, seq, dh)).astype(np.float32)
        t = T.Tensor._from_gpu(rt.h2d(a), requires_grad=True)
        outs.append(t)
    return outs


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_lowmem_reports_smem_limit_clearly():
    T.set_default_device("cuda")
    rt = T._cuda()
    bh, seq, dh = 2, 64, 64          # dh=64 才会走融合注意力
    q, k, v = _qkv(bh, seq, dh)
    o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
    g = T.Tensor._from_gpu(rt.h2d(np.ones((bh, seq, dh), np.float32)))

    orig = rt._device_smem_limit
    try:
        rt._device_smem_limit = lambda: 40 * 1024   # 模拟"连窗口版都装不下"的卡
        rt._attn_flash_names = set()                # 逼它重新检查
        with pytest.raises(CudaError) as ei:
            o.backward(g)
        msg = str(ei.value)
        assert "40" in msg, "报错要写明本卡上限，实际: %s" % msg[:200]
        assert "修法" in msg, "报错必须带修法"
        assert "lowmem=False" in msg, "修法要给出可执行替代方案"
        assert ("Pascal" in msg) or ("sm_61" in msg), "要指出这是哪类卡的限制"
    finally:
        rt._device_smem_limit = orig
        T.set_default_device("cpu")


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_lowmem_still_works_when_limit_is_enough():
    """上限够时不能误伤：V100 的 96 KB 应放行，且 q 真的拿到梯度。"""
    T.set_default_device("cuda")
    rt = T._cuda()
    lim = rt._device_smem_limit()
    if lim and lim < 68 * 1024:
        pytest.skip("本卡上限 %d KB < 68 KB，属于应被拦截的情形" % (lim // 1024))
    bh, seq, dh = 2, 64, 64
    q, k, v = _qkv(bh, seq, dh, seed=1)
    o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
    o.backward(T.Tensor._from_gpu(rt.h2d(np.ones((bh, seq, dh), np.float32))))
    try:
        assert q.grad is not None, "叶子张量必须拿到梯度"
        gq = q.grad.to_numpy() if hasattr(q.grad, "to_numpy") else q.grad
        assert np.all(np.isfinite(np.asarray(gq, np.float32)))
    finally:
        T.set_default_device("cpu")


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_lowmem_works_on_48kb_card():
    """48 KB 上限（P100 / sm_60）本来是"编不出来"的，现在走窗口版必须能跑出梯度。

    这是 2026-11-15 的成果：反向新增 ttn_bwd_flash_w（Q/dO 沿 head_dim 16 宽滑窗，
    动态 smem 45056 B），由 _attn_flash_launch 按设备上限自动选择。
    """
    T.set_default_device("cuda")
    rt = T._cuda()
    bh, seq, dh = 2, 64, 64
    q, k, v = _qkv(bh, seq, dh, seed=2)
    orig = rt._device_smem_limit
    try:
        rt._device_smem_limit = lambda: 48 * 1024   # 模拟 P100
        rt._attn_flash_names = set()
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        o.backward(T.Tensor._from_gpu(rt.h2d(np.ones((bh, seq, dh), np.float32))))
        assert q.grad is not None, "48 KB 上限下必须能跑出梯度"
        gq = np.asarray(q.grad.to_numpy() if hasattr(q.grad, "to_numpy") else q.grad, np.float32)
        assert np.all(np.isfinite(gq)), "梯度必须是有限值"
        assert float(np.abs(gq).max()) > 0.0, "梯度不能全零"
    finally:
        rt._device_smem_limit = orig
        rt._attn_flash_names = set()
        T.set_default_device("cpu")
