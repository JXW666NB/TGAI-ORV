# -*- coding: utf-8 -*-
"""`norm_sanity`：归一化权重体检。

实测背景（2026-11-01，V100 上那份 qwen2.5-0.5b-instruct-fp16.TG）：
  2-D 权重统计完全正常（embed std 0.015、w1 std 0.023、q/k 正常），
  但 final_norm.weight 均值 **7.14**、blocks.0.ln1.weight 均值 **0.032** ——
  加载不报错、训练不报错，模型却永远训不好（base probe 10.2 vs safetensors 1.81）。
判据：1-D 且名字含 norm 的权重，均值应≈1；多数可疑 ⇒ 这份文件读错了。
"""
import numpy as np

from tgai_go.train import checkpoints as C


def _gamma(vals):
    return np.asarray(vals, np.float32)


def test_healthy_norms_pass():
    w = {"blocks.0.ln1.weight": _gamma([1.0, 1.2, 0.9, 1.1]),
         "final_norm.weight": _gamma([1.0] * 6)}
    good, bad = C.norm_sanity(w)
    assert good == 2 and bad == []


def test_broken_norm_is_flagged_with_values():
    """实测数据：均值 7.14 的那份 final_norm 必须被点名。"""
    w = {"final_norm.weight": _gamma([7.0, 7.2, 7.1, 6.9])}
    good, bad = C.norm_sanity(w)
    assert good == 0 and len(bad) == 1
    name, mean, lo, hi = bad[0]
    assert name == "final_norm.weight" and 6.5 < mean < 7.5


def test_tiny_norm_is_flagged():
    """实测数据：均值 0.032 的 ln1 必须被点名。"""
    w = {"blocks.0.ln1.weight": _gamma([0.03, 0.032, 0.031])}
    good, bad = C.norm_sanity(w)
    assert good == 0 and len(bad) == 1


def test_extreme_outlier_is_flagged_even_if_mean_ok():
    """均值正常但极值离谱（16.75）也算可疑 —— 实测里它就出现过。"""
    w = {"final_norm.weight": _gamma([1.0, 1.0, 1.0, 16.75])}
    good, bad = C.norm_sanity(w)
    assert len(bad) == 1, bad


def test_non_norm_and_2d_tensors_are_ignored():
    w = {"blocks.0.ffn.w1.weight": np.zeros((4, 4), np.float32),
         "blocks.0.attn.q_proj.bias": _gamma([9.0, 9.0])}
    good, bad = C.norm_sanity(w)
    assert good == 0 and bad == [], "只检查 1-D 且名字含 norm 的张量"
