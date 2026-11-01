# -*- coding: utf-8 -*-
"""`.TG` 线性层布局校验：转置要自动修正，形状不对要**大声报错**。

背景（2026-11-01 实测）：V100 上那份 qwen2.5-0.5b-instruct-fp16.TG
头/表全对、fp16 未量化，但 ffn.w1 是 [896, 4864]（我们期望 [4864, 896]）。
旧代码直接放行 → 模型能加载、能训练，**loss 从 10.2 起、gnorm 4237**；
同一份权重换 safetensors 加载则 base probe 1.81、gnorm 28。
"静默算错"比崩溃危险，所以这里必须有一道形状自洽检查。
"""
import numpy as np
import pytest

from tgai_go.train import checkpoints as C

CFG = {"D": 8, "NH": 2, "KV": 1, "DH": 4, "FF": 16, "V": 32}


def _w(shape):
    return np.zeros(shape, np.float32)


def _good():
    return {"blocks.0.attn.q_proj.weight": _w((2 * 4, 8)),
            "blocks.0.attn.k_proj.weight": _w((1 * 4, 8)),
            "blocks.0.attn.v_proj.weight": _w((1 * 4, 8)),
            "blocks.0.attn.out_proj.weight": _w((8, 2 * 4)),
            "blocks.0.ffn.w1.weight": _w((16, 8)),
            "blocks.0.ffn.w2.weight": _w((16, 8)),
            "blocks.0.ffn.w3.weight": _w((8, 16))}


def test_correct_layout_is_untouched():
    w = _good()
    out, flipped = C.fix_linear_layout(w, CFG)
    assert flipped == []
    for k in w:
        assert out[k].shape == w[k].shape


def test_transposed_layout_is_fixed_automatically():
    w = {k: np.ascontiguousarray(v.T) for k, v in _good().items()}
    out, flipped = C.fix_linear_layout(w, CFG)
    # 方阵（q_proj / out_proj 在这里恰好 8x8）转不转置都一样，不该出现在 flipped 里；
    # 其余非方阵必须全部被修正（且是所有层，不只第 0 层）
    non_square = {k for k, v in _good().items() if v.shape[0] != v.shape[1]}
    assert set(flipped) == non_square
    for k in _good():
        assert out[k].shape == _good()[k].shape


def test_wrong_shape_raises_with_fix():
    w = _good()
    w["blocks.0.ffn.w1.weight"] = _w((7, 5))          # 既不是 [16,8] 也不是 [8,16]
    with pytest.raises(C.CheckpointError) as ei:
        C.fix_linear_layout(w, CFG)
    msg = str(ei.value)
    assert "ffn.w1" in msg and "修法" in msg


def test_missing_tensor_is_skipped_not_fatal():
    w = _good()
    w.pop("blocks.0.ffn.w2.weight")
    out, flipped = C.fix_linear_layout(w, CFG)
    assert flipped == [] and "blocks.0.ffn.w2.weight" not in out


def test_strict_false_allows_unknown_layout():
    w = _good()
    w["blocks.0.ffn.w1.weight"] = _w((7, 5))
    out, flipped = C.fix_linear_layout(w, CFG, strict=False)
    assert out["blocks.0.ffn.w1.weight"].shape == (7, 5)
    assert flipped == []
