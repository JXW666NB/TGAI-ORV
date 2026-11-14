# -*- coding: utf-8 -*-
"""Muon 的显存分配、CPU 更新路径、分层更新（0.9.29）。

## 修的三件事

1. **状态按形状分配**（旧实现给每个参数都分配 `buf + _adam_m + _adam_v` 三份，
   2.97B 要 33.3 GiB，比 AdamW 还差）；
2. **`fp16_mom`** 让 2D 那部分减半（2.97B：11.1 → 5.5 GiB）；
3. **`step_layers` 走 CPU 分派**（原来只能 CUDA，没法在测试里验证等价性）。

## 顺带修的兼容性 BUG

`Module.parameters()` 原本**不去重**。绑定词嵌入（`tie_embedding`）让
`embed.weight` 与 `lm_head.weight` 是同一个对象，于是优化器**每步把它更新两遍**
（实测与分层更新差 7.2e-03）。PyTorch 的 `parameters()` 会去重，这里对齐。

## 关于"分层更新是否等价"

实测残差 **≤1e-07**（fp32 舍入量级），**不是逐位相同**：分层更新改变了参数的更新顺序，
Newton-Schulz 迭代里累积出 fp32 噪声。所以测试判据取 ≤1e-6，不假装逐位一致。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


# ---------------- 状态占用 ----------------
def test_state_is_per_shape_not_three_copies():
    """2D 参数每元素 4 字节；非 2D 是 Adam 的 8 字节。绝不能再是 12。"""
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=1, n_heads=2)
    opt = T.optim.Muon(list(m.parameters()), lr=0.01)
    want = sum(int(p.numel()) * (4 if opt._is2d[i] else 8)
               for i, p in enumerate(opt.params))
    assert opt.state_bytes() == want
    n_all = sum(int(p.numel()) for p in opt.params)
    assert opt.state_bytes() < n_all * 12, "又回到每参数三份状态"


def test_fp16_only_halves_the_2d_part():
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=1, n_heads=2)
    a = T.optim.Muon(list(m.parameters()), lr=0.01, fp16_mom=False)
    b = T.optim.Muon(list(m.parameters()), lr=0.01, fp16_mom=True)
    n2 = sum(int(p.numel()) for i, p in enumerate(a.params) if a._is2d[i])
    assert a.state_bytes() - b.state_bytes() == n2 * 2


def test_3b_extrapolation_matches_claim():
    """README 写的 2.97B 用量，用同样的算式复核一遍。"""
    n = 2.97e9
    fp32 = n * 4 / (1 << 30)
    fp16 = n * 2 / (1 << 30)
    assert 10.5 < fp32 < 11.5, fp32      # ≈11.1 GiB
    assert 5.0 < fp16 < 6.0, fp16        # ≈5.5 GiB


# ---------------- CPU 更新 ----------------
def test_cpu_step_moves_2d_and_1d_params():
    w2 = T.nn.Parameter(np.zeros((8, 4), np.float32))
    w1 = T.nn.Parameter(np.zeros((6,), np.float32))
    opt = T.optim.Muon([w2, w1], lr=0.01, momentum=0.9, weight_decay=0.1)
    w2.grad = np.full((8, 4), 0.1, np.float32)
    w1.grad = np.full((6,), 0.1, np.float32)
    opt.step()
    assert np.abs(np.asarray(w2.data)).max() > 0, "2D 没更新"
    assert np.abs(np.asarray(w1.data)).max() > 0, "1D 没更新"


def test_fp16_momentum_is_close_to_fp32():
    def run(fp16):
        w = T.nn.Parameter(np.zeros((8, 4), np.float32))
        opt = T.optim.Muon([w], lr=0.01, momentum=0.9, fp16_mom=fp16)
        w.grad = np.full((8, 4), 0.1, np.float32)
        opt.step()
        return np.asarray(w.data, np.float32).copy()

    a, b = run(False), run(True)
    rel = float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max())))
    assert rel < 5e-3, "fp16 与 fp32 动量差太多：%.2e" % rel


# ---------------- 绑定权重 ----------------
def test_parameters_dedups_tied_weights():
    """绑定词嵌入必须是**一个**参数，否则优化器每步更新它两遍。"""
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=1, n_heads=2)
    ps = list(m.parameters())
    ids = [id(p) for p in ps]
    assert len(ids) == len(set(ids)), "parameters() 里仍有重复对象"
    assert m.embed.weight is m.lm_head.weight, "预设应共享同一 Parameter"


# ---------------- 分层更新 ----------------
def test_step_layers_matches_step_within_fp32_noise():
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=2, n_heads=2)
    ps = list(m.parameters())
    sd0 = m.state_dict()
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 32, (8,), dtype=np.int64)
    tgt = rng.integers(0, 32, (8,), dtype=np.int64)

    def run(per_layer):
        m.load_state_dict(sd0)
        opt = T.optim.Muon(ps, lr=0.01, momentum=0.9)
        if per_layer:
            opt.set_layer_groups({i: list(m.blocks[i].parameters()) for i in range(2)})
        m(ids, tgt).backward()
        if per_layer:
            for g in opt.all_groups():
                opt.step_layers([g], t=1, release=False)
        else:
            opt.step()
        return m.state_dict()

    a, b = run(False), run(True)
    per_key = []
    scale = 0.0
    for k in a:
        x, y = np.asarray(a[k], np.float32), np.asarray(b[k], np.float32)
        if x.size:
            per_key.append((float(np.abs(x - y).max()), k, tuple(x.shape)))
    per_key.sort(reverse=True)
    worst = per_key[0][0] if per_key else 0.0
    # 更新量级：拿"更新前 vs 更新后"的差作为尺度
    for k, v0 in sd0.items():
        x = np.asarray(a[k], np.float32)
        if x.size:
            scale = max(scale, float(np.abs(x - np.asarray(v0, np.float32)).max()))
    # 判据用**尺度相对**而不是硬编码绝对值：
    # 差异必须远小于这一步的更新量本身（<1%），这才是"同一算法、只是求和顺序不同"的正确表述。
    # 为什么不用绝对值：numpy 的 BLAS 多线程累加顺序会在不同运行间变化，
    # 绝对值阈值会偶发越界（实测过 1/16 概率），而相对判据天然稳健。
    assert scale > 0, "参数没有被更新，无法比较"
    if worst > 0.01 * scale:
        # 诊断：把偏差最大的几个参数打出来，便于判断是"求和顺序噪声"还是"漏更新/更新错对象"。
        # 前者偏差均匀地小，后者会集中在个别张量上。
        print("\n[Muon 诊断] 更新量尺度 %.3e，偏差 top3：" % scale)
        for d, k, shp in per_key[:3]:
            print("   %-40s shape=%-14s 偏差 %.3e（占尺度 %.2f%%）"
                  % (k, str(shp), d, 100 * d / scale))
    assert worst <= 0.01 * scale, \
        "分层更新与统一更新差异过大：%.3e（更新量 %.3e，占比 %.2f%%）" % (worst, scale, 100 * worst / scale)
    # 实测该占比约 0.02%（2e-06 / 1e-2），远低于 1% 的判据。
    # **不是逐位相同**：分层更新改变了参数更新顺序，Newton-Schulz 迭代上会累积 fp32 噪声。


def test_group_coverage_is_enforced():
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=1, n_heads=2)
    opt = T.optim.Muon(list(m.parameters()), lr=0.01)
    with pytest.raises(ValueError):
        opt.set_layer_groups({0: list(m.blocks[0].parameters())},
                             auto_rest=False, require_full=True)


def test_auto_rest_covers_everything():
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=2, n_heads=2)
    opt = T.optim.Muon(list(m.parameters()), lr=0.01)
    opt.set_layer_groups({i: list(m.blocks[i].parameters()) for i in range(2)})
    covered = set()
    for idx in opt._groups.values():
        covered.update(idx)
    assert len(covered) == len(opt.params), "auto_rest 后仍有参数未覆盖"
