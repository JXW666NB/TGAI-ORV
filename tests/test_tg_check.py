# -*- coding: utf-8 -*-
"""`tg_check.check_table`：`.TG` 容器自洽性检查（纯函数，CI 可跑）。

背景（2026-11-01 的纠错）：
  我曾用"归一化权重均值应≈1"的常识阈值判定"`.TG` 读错了"，后来对拍官方
  safetensors 发现**官方模型的 ln1 均值本来就是 0.032** —— 那个判据是错的。
  客观的判据是**结构自洽**：容器声明的 hash/off/len/shape 必须互相不矛盾。
"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "tools", "tg_check.py")
    spec = importlib.util.spec_from_file_location("tg_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e(h, off, ln, shape=(4,)):
    return dict(hash=h, off=off, len=ln, dtype=1, ndim=len(shape), shape=list(shape),
                scale=0)


def test_clean_table_passes():
    tc = _load()
    tab = [_e(1, 0, 8), _e(2, 8, 8), _e(3, 16, 4)]
    problems, stats = tc.check_table(tab, file_bytes=1000, data_off=100)
    assert problems == [], problems
    assert stats["entries"] == 3 and stats["unique_hashes"] == 3


def test_duplicate_hash_is_reported():
    tc = _load()
    tab = [_e(7, 0, 8), _e(7, 8, 8)]
    problems, _ = tc.check_table(tab, file_bytes=1000, data_off=0)
    assert any("hash 重复" in p for p in problems), problems


def test_overlapping_ranges_are_reported():
    tc = _load()
    tab = [_e(1, 0, 16), _e(2, 8, 16)]      # 第二个从 8 开始，与第一个重叠
    problems, _ = tc.check_table(tab, file_bytes=1000, data_off=0)
    assert any("重叠" in p for p in problems), problems


def test_out_of_bounds_is_reported():
    tc = _load()
    tab = [_e(1, 0, 500)]
    problems, _ = tc.check_table(tab, file_bytes=600, data_off=200)
    assert any("越界" in p for p in problems), problems


def test_empty_table_is_not_a_crash():
    tc = _load()
    problems, stats = tc.check_table([], file_bytes=128, data_off=128)
    assert problems == [] and stats["entries"] == 0
