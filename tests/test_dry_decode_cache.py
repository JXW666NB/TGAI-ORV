# -*- coding: utf-8 -*-
"""DRY 的 token→文本缓存回归测试 —— 缺陷 E。

背景：`_sampling._dry_penalize` 每个生成步都要把最近 N（默认 256）个 token
解码成文本（判断 DRY 断句符）。改造前 `_session.generate` 直接把
`engine.decode` 传给它，于是：
  · 每 token 多出最多 256 次 ctypes 跨界调用（tg_token_to_str）；
  · engine.decode 的 SentencePiece detokenize 状态机被 DRY 反复推进
    ⇒ 打乱主输出的 ▁ 还原。

修复：`TGEngine.dry_decode` —— 无状态（不碰 detok 状态机）+ 按 token id
缓存。原 tgai_chat.py 里的 `_dry_raw_decode` 就是同样设计（抽取成库时退化
了一次）。本测试锁住"每个 token 只跨界解码一次"。
"""
import types

import numpy as np

from tgai_go._engine import TGEngine
from tgai_go._sampling import _dry_penalize


def _fake_engine(texts, sidecar=None):
    """够 dry_decode 用的最小 engine 替身（记录 _token_str 调用）。"""
    calls = []
    eng = types.SimpleNamespace(
        sidecar=sidecar,
        _dry_decode_cache=None,
        _calls=calls,
    )

    def _token_str(tid):
        calls.append(tid)
        return texts.get(tid)

    eng._token_str = _token_str
    return eng


def test_dry_decode_caches_by_token_id():
    eng = _fake_engine({7: '的', 8: '，'})
    call = lambda tid: TGEngine.dry_decode(eng, tid)      # noqa: E731
    assert call(7) == '的'
    assert call(7) == '的'
    assert call(8) == '，'
    assert call(8) == '，'
    assert eng._calls == [7, 8], '同一 token 不应重复跨界解码'
    for _ in range(10):
        call(7)
    assert eng._calls == [7, 8]
    # 未知 token（_token_str 返回 None）缓存成空串，也只查一次
    assert call(99) == ''
    assert call(99) == ''
    assert eng._calls == [7, 8, 99]


def test_dry_decode_uses_sidecar_when_present():
    class _FakeSidecar:
        def __init__(self):
            self.calls = []

        def decode(self, ids):
            self.calls.append(tuple(ids))
            return 'X'

    sc = _FakeSidecar()
    eng = _fake_engine({}, sidecar=sc)
    assert TGEngine.dry_decode(eng, 3) == 'X'
    assert TGEngine.dry_decode(eng, 3) == 'X'
    assert sc.calls == [(3,)], 'sidecar 路径同样要按 token 缓存'


def test_dry_decode_does_not_touch_detokenize_state():
    """dry_decode 不得调用 engine.decode（会推进 detok 状态机）。

    fake engine 根本没有 _detok_state / decode，能跑通就说明没走那条路。
    """
    eng = _fake_engine({5: 'a'})
    assert TGEngine.dry_decode(eng, 5) == 'a'
    assert not hasattr(eng, 'decode')
    assert not hasattr(eng, '_detok_tl')


def test_dry_penalize_only_decodes_each_unique_token_once():
    """连续 5 步 DRY（每步 256 个历史 token）只应有 4 次跨界解码。"""
    texts = {0: '的', 1: '，', 2: '。', 3: 'a'}
    eng = _fake_engine(texts)
    hist = [i % 4 for i in range(256)]
    logits = np.zeros(16, dtype=np.float64)
    p = {'dry_multiplier': 0.8, 'dry_penalty_last_n': 256,
         'dry_allowed_length': 3}
    decode_fn = lambda tid: TGEngine.dry_decode(eng, tid)  # noqa: E731
    for _ in range(5):
        _dry_penalize(logits.copy(), hist, p, decode_fn)
    assert len(eng._calls) == 4, (
        '旧实现每步会解码 256 次（5 步共 1280 次）；现在应只解 4 个唯一 token，'
        '实际 %d 次：%r' % (len(eng._calls), eng._calls))
