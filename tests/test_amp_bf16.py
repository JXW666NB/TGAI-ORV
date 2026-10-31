# -*- coding: utf-8 -*-
"""bf16 转换的**数值**正确性（CPU，对着 torch 的 bf16 实现验）。

为什么这一步能先做：bf16 就是 float32 的高 16 位，舍入规则可以用纯整数运算实现，
于是**不碰 GPU** 也能验证位级语义对不对。GPU 上的内核只是把同一套位运算翻译过去。

这条测试守住三件事：
1. 舍入是 round-to-nearest-even（与 CUDA/torch 一致），不是截断；
2. 往返误差不超过 bf16 的理论上限（8 位尾数 ⇒ 相对误差 ≤ 2^-9 ≈ 1.95e-3）；
3. NaN/Inf/次正规数不炸。
"""
import numpy as np
import pytest

from tgai_go.train import amp


def test_rounding_matches_torch_bfloat16():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = (rng.normal(0, 3, 20000).astype(np.float32) *
         np.float32(10.0) ** rng.integers(-6, 6, 20000).astype(np.float32))
    ours = amp.from_bf16_bits(amp.to_bf16_bits(x))
    theirs = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
    same = float(np.mean(ours.view(np.uint32) == theirs.view(np.uint32)))
    assert same >= 0.999, "与 torch 的 bf16 舍入只有 %.2f%% 的位完全相同" % (100 * same)
    rel = float(np.abs(ours - theirs).max()) / float(np.abs(x).max())
    assert rel < 1e-6, rel


def test_roundtrip_error_within_bf16_precision():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 5000).astype(np.float32)
    # bf16 = 1 符号 + 8 指数 + **7 位显式尾数**（加 1 位隐含 ⇒ 8 位有效位）。
    # 所以 RNE 的最坏相对误差是 2^-8 ≈ 3.9e-3（截断更差，最坏 2^-7）。
    # 我第一版写成 2^-9 是算错了位宽 —— 实测 3.85e-3 正好贴着 2^-8，说明实现是对的。
    assert amp.roundtrip_rel_error(x) <= 2.0 ** -8


def test_truncation_would_be_biased():
    """反证：直接截断（不做 RNE）会产生系统性向零偏置 —— 说明舍入不是可有可无的细节。"""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 5000).astype(np.float32)
    trunc = (x.view(np.uint32) >> 16).astype(np.uint16)
    trunc_f = amp.from_bf16_bits(trunc)
    rne_f = amp.from_bf16_bits(amp.to_bf16_bits(x))
    # 截断的误差均值应当明显偏向"绝对值变小"
    assert float(np.mean(np.abs(trunc_f) - np.abs(x))) < 0
    assert abs(float(np.mean(np.abs(rne_f) - np.abs(x)))) < abs(
        float(np.mean(np.abs(trunc_f) - np.abs(x))))


def test_special_values_survive():
    x = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan,
                  1e-40, 3.4e38], np.float32)
    b = amp.to_bf16_bits(x)
    y = amp.from_bf16_bits(b)
    assert np.isfinite(y[:4]).all()
    assert np.isinf(y[4]) and np.isinf(y[5])
    assert np.isnan(y[6])
    assert y[2] == 1.0 and y[3] == -1.0


def test_env_flag_is_separate_from_tg_bf16(monkeypatch):
    """TG_AMP 管"存储"，TG_BF16 管"算" —— 两个开关不能混。"""
    monkeypatch.delenv("TG_AMP", raising=False)
    assert amp.amp_requested() is False
    monkeypatch.setenv("TG_AMP", "1")
    assert amp.amp_requested() is True
    monkeypatch.setenv("TG_AMP", "off")
    assert amp.amp_requested() is False


def test_bf16_kernel_source_has_no_includes():
    """内核源码不能出现 #include：NVRTC 没配 -I，加了就编不过（仓库里记录过的坑）。"""
    src = amp.BF16_KERNELS
    assert "#include" not in src
    for name in ("ewise_add_bf16", "ewise_scale_bf16", "rmsnorm_bf16"):
        assert name in src, name
    # 计算必须是 float：读 bf16 → 转 float → 算 → 写回
    assert "tg_bf16_to_f32" in src and "tg_f32_to_bf16" in src
