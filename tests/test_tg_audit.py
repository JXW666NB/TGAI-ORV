# -*- coding: utf-8 -*-
"""`.TG` 体检工具的判定逻辑测试。

工具本身要跑真模型（在 GPU 机器上），但**判定规则**是纯函数，可以在 CI 里钉住：
CE 差 ≤ 0.5 判等价，> 0.5 判不等价并给出三个最可能原因。
实测基线：同一份 Qwen2.5-0.5B，`.TG`(坏布局) 8.3 vs safetensors 3.5 ⇒ 必须判 FAIL。
"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "tools", "tg_audit.py")
    spec = importlib.util.spec_from_file_location("tg_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equivalent_checkpoints_pass():
    ta = _load()
    ok, why = ta.compare(3.50, 3.55)
    assert ok, why


def test_different_checkpoints_fail_with_likely_causes():
    ta = _load()
    ok, why = ta.compare(8.30, 3.50)          # 实测：坏布局 vs 正确加载
    assert not ok
    for kw in ("[in,out]", "RoPE", "量化"):
        assert kw in why, "结论里应提示可能原因：%s" % kw


def test_missing_side_fails_explicitly():
    ta = _load()
    ok, why = ta.compare(None, 3.5)
    assert not ok and "失败" in why


def test_prompts_cover_both_scripts():
    """体检至少要覆盖中英各一句 —— 只测英文可能漏掉中文词表/嵌入问题。"""
    ta = _load()
    assert any(any("\u4e00" <= ch <= "\u9fff" for ch in p) for p in ta.PROMPTS), \
        "PROMPTS 里应含中文样本"
    assert any(all(ord(ch) < 128 for ch in p) for p in ta.PROMPTS), \
        "PROMPTS 里应含纯英文样本"


def test_tolerance_is_documented_value():
    ta = _load()
    ok, _ = ta.compare(3.50, 4.01, tol=0.5)
    assert not ok, "差 0.51 > 0.5 应判失败"
    ok, _ = ta.compare(3.50, 3.99, tol=0.5)
    assert ok, "差 0.49 ≤ 0.5 应判通过"
