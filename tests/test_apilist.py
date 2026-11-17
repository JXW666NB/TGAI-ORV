# -*- coding: utf-8 -*-
"""`T.help()`（库内自省）的测试。

重点：**不需要 GPU**，所以这些用例在没有显卡的机器上也能跑。
"""
import io
import sys

import tgai_go.train as T


def _txt(**kw):
    kw.setdefault("print", False)
    return T.help(**kw)


def test_overview_has_bilingual_prompts():
    s = _txt()
    assert "能力总览" in s and "capability overview" in s
    # 三类各列 10 个
    assert s.count("[kernels]") == 1
    assert s.count("[api]") == 1
    assert s.count("[op]") == 1
    # 每条都有中英两行
    assert "中文:" in s and "EN  :" in s


def test_kernels_listed_with_params_and_paging():
    s = _txt(tag="内核", n=5, page=1)
    assert "[kernels]" in s
    assert "CUDA 内核" in s
    assert "参数/Params" in s or "用法/Usage" in s
    # 分页提示存在，且第 2 页与第 1 页内容不同
    s2 = _txt(tag="内核", n=5, page=2)
    assert s != s2
    assert "page 2" in s2


def test_api_category_resolves_without_gpu():
    s = _txt(tag="API")
    assert "[api]" in s
    # 运行时类必须能被取到（不需要 GPU）；这里只要求有内容
    assert "用法/Usage" in s


def test_keyword_search_across_categories():
    s = _txt(s="flash")
    assert "flash" in s.lower()
    assert "搜索" in s


def test_positional_topic_acts_as_keyword():
    a = T.help("flash", print=False)
    b = _txt(s="flash")
    assert a == b


def test_tag_plus_keyword():
    s = _txt(tag="算子", s="mul")
    assert "mul" in s.lower()


def test_unknown_keyword_is_reported():
    s = _txt(s="zzz_definitely_not_present")
    assert "没有命中" in s


def test_bad_kwargs_raise_with_hint():
    try:
        T.help(bogus=1, print=False)
    except TypeError as e:
        assert "修法" in str(e)
    else:
        raise AssertionError("应当抛出 TypeError")


def test_print_true_writes_to_stdout():
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        T.help(tag="算子", n=2)
    finally:
        sys.stdout = old
    assert "张量算子" in buf.getvalue()
