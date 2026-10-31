# -*- coding: utf-8 -*-
"""数据体检的统计与自检（纯 CPU，用假编码器，不需要真词表）。"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "tools", "dataset_report.py")
    spec = importlib.util.spec_from_file_location("dataset_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _enc(s):
    return list(range(len(str(s))))          # 每个字符一个 token，便于心算


def test_reports_prompt_completion_signal():
    dr = _load()
    items = [{"prompt": "abc", "completion": "de"},        # 总 5，答案 2
             {"prompt": "a", "completion": "bcdf"}]        # 总 5，答案 4（拼起来与上条不同）
    rep = dr.analyze(items, _enc, seq_len=16)
    assert rep["样本数"] == 2
    assert rep["答案 token 中位数"] == 3
    assert rep["总 token 中位数"] == 5
    assert rep["超过 seq_len 的样本"] == 0
    assert all(rep["checks"].values()), rep["checks"]


def test_truncation_counts_and_clips_answer():
    dr = _load()
    items = [{"prompt": "abcd", "completion": "efghijklmnop"}]   # 16 token，seq_len=8
    rep = dr.analyze(items, _enc, seq_len=8)
    assert rep["超过 seq_len 的样本"] == 1
    # 截断后答案 token = seq_len - len(prompt) = 4
    assert rep["答案 token 最大"] == 4
    assert rep["checks"]["截断比例 < 20%"] is False        # 1/1 全都截断，必须报警


def test_empty_completion_skipped_and_duplicates_counted():
    dr = _load()
    items = [{"prompt": "a", "completion": "  "},
             {"prompt": "a", "completion": "b"},
             {"prompt": "a", "completion": "b"}]
    rep = dr.analyze(items, _enc, seq_len=32)
    assert rep["空答案（已跳过）"] == 1
    assert rep["样本数"] == 2
    assert rep["重复样本"] == 1
    assert rep["checks"]["重复样本 < 10%"] is False


def test_text_only_items_use_whole_text_as_answer():
    dr = _load()
    items = [{"text": "abcdef"}]
    rep = dr.analyze(items, _enc, seq_len=32)
    assert rep["答案 token 中位数"] == 6
    assert rep["掩码比例"] == "0.0%"
