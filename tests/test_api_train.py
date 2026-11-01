# -*- coding: utf-8 -*-
"""`tgai_go.api.train` —— 十行代码入口的契约测试（纯 CPU）。

这一层是**编排层**：它把参数校验成清晰的错误、驱动已验证的训练脚本、
再把结果解析成 dict。所以测试分两类：
  1. 纯函数/解析：快、覆盖判据（`_parse`、`method_supported`、tokenizer 解析）；
  2. 一次 tiny 真跑：3 步、seq 32、CPU，确认端到端能返回数字（约几秒）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go import api


def test_parse_extracts_probe_and_verdict():
    text = ("探针损失：1.969536 → 1.712395（变化 13.1%）\n"
            "结论：PASS —— SFT 链路可用\n")
    r = api._parse(text, 0, ["python", "x.py"])
    assert r["probe_base"] == pytest.approx(1.969536)
    assert r["probe_final"] == pytest.approx(1.712395)
    assert r["drop_pct"] == pytest.approx(13.1, abs=0.1)
    assert r["verdict"] == "PASS"


def test_parse_marks_fail_when_no_pass():
    r = api._parse("结论：FAIL —— 探针没下降\n", 1, ["python"])
    assert r["verdict"] == "FAIL"
    assert r["returncode"] == 1


def test_unsupported_methods_raise_with_plan():
    # pretrain 现在已支持（见 test_api_pretrain.py）；dpo/grpo 仍是"计划中"
    for m in ("dpo", "grpo"):
        with pytest.raises(NotImplementedError) as ei:
            api.method_supported(m)
        assert "修法" in str(ei.value)


def test_missing_data_raises_with_fix():
    with pytest.raises(FileNotFoundError) as ei:
        api.train(data="/definitely/missing.jsonl", weights="/tmp/x")
    assert "修法" in str(ei.value)


def test_missing_weights_raises_with_hint():
    with pytest.raises(ValueError) as ei:
        api.train(data="x.jsonl")
    assert "tiny=True" in str(ei.value)


def test_tokenizer_resolved_from_weights_dir(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert api._resolve_tokenizer(str(tmp_path), None).endswith("tokenizer.json")


def test_tokenizer_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        api._resolve_tokenizer(None, str(tmp_path / "nope.json"))


def test_tiny_end_to_end_returns_numbers():
    r = api.train(tiny=True, steps=3, quiet=True)
    assert r.returncode == 0, r.log[-500:]
    assert r.probe_base and r.probe_final, "tiny 跑完必须能解析出探针数字"
    assert r.verdict in ("PASS", "FAIL")
    assert isinstance(r.drop_pct, float)

def test_parse_prefers_probe_line_over_training_loss():
    """回归：示例同时打印「训练 loss（不可比）」与「探针 loss（可比）」，
    取第一个匹配会报出不可比的数字（实测踩过：5.5995→5.541 vs 真值 5.5789→5.2458）。"""
    text = ("训练 loss（轮转样本，不可比）：5.5995 → 5.5410\n"
            "探针 loss（固定序列，可比）：  5.578872 → 5.245787\n"
            "结论：PASS\n")
    r = api._parse(text, 0, ["python", "x.py"])
    assert r["probe_base"] == pytest.approx(5.578872)
    assert r["probe_final"] == pytest.approx(5.245787)
    assert r["drop_pct"] == pytest.approx(6.0, abs=0.2)
