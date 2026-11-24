# -*- coding: utf-8 -*-
"""server.py 语义回归测试 —— 缺陷 D。

背景（修复前）：
  · KV 池满时 kv_for_request 等 5s 返回 None，调用方 kv=None ⇒ generate()
    自建 max_seq×… 的大 KV（V2.7 ≈1.88GB/请求）⇒ 过载雪崩，且没有 429/503；
  · _collect 没有 try/finally，generate 抛异常时 KV 槽泄漏；
  · 每请求往 stderr 打调试块（还会打印用户 prompt 尾巴），其中引用了赋值前的
    `ids`（NameError 被吞）；
  · finish_reason 永远 "stop"，撞 max_seq/max_tokens 截断时无法区分。

本测试不加载真实模型：用 fake session / monkeypatch generate，
只验证服务层语义（503 + Retry-After、槽归还、finish_reason）。
"""
import json
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tgai_go import server


# ── 工具：起一个绑定 fake session 的进程内 HTTP 服务 ────────────────
def _make_handle():
    h = server.ServerHandle(None, None, '127.0.0.1', 0, 'fake-model')
    h.engine = types.SimpleNamespace()
    h.params = dict(server.DEFAULT_PARAMS)
    h.sys_prompt = ''
    h.default_think_level = 'off'

    def new_session():
        return types.SimpleNamespace(
            think_mode='direct', eos=3, eos_set={3},
            build_prompt=lambda msg: ('', [2, 3, 4]))

    h.new_session = new_session
    return h


@pytest.fixture()
def live_server():
    h = _make_handle()
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server._make_handler(h))
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever,
                          name='tgai-test-http', daemon=True)
    th.start()
    url = 'http://127.0.0.1:%d/v1/chat/completions' % httpd.server_address[1]
    try:
        yield h, url
    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=3.0)


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read().decode('utf-8'))


_MSG = {'messages': [{'role': 'user', 'content': 'hi'}]}


class _FakeKV:
    pass


# ══════════════════════════════════════════════════════════════════
#  1) kv_for_request：池满必须抛 ServerOverloaded（不再返回 None）
# ══════════════════════════════════════════════════════════════════
def test_kv_for_request_raises_when_pool_exhausted(monkeypatch):
    monkeypatch.setattr(server, '_kv_pool_acquire', lambda engine: None)
    with pytest.raises(server.ServerOverloaded) as ei:
        server.kv_for_request(object())
    msg = str(ei.value)
    assert 'exhausted' in msg
    assert 'size=' in msg and 'busy=' in msg


def test_kv_for_request_wraps_pool_error(monkeypatch):
    def boom(engine):
        raise RuntimeError('alloc failed')
    monkeypatch.setattr(server, '_kv_pool_acquire', boom)
    with pytest.raises(server.ServerOverloaded) as ei:
        server.kv_for_request(object())
    assert 'alloc failed' in str(ei.value)


# ══════════════════════════════════════════════════════════════════
#  2) 非流式：池满 ⇒ HTTP 503 + Retry-After: 1 + server_overloaded
# ══════════════════════════════════════════════════════════════════
def test_overload_returns_503_with_retry_after(live_server, monkeypatch):
    _h, url = live_server

    def overloaded(engine):
        raise server.ServerOverloaded('KV slot pool exhausted (size=4, busy=4); retry later')

    monkeypatch.setattr(server, 'kv_for_request', overloaded)
    code, headers, body = _post(url, _MSG)
    assert code == 503
    assert headers.get('Retry-After') == '1'
    assert body['error']['type'] == 'server_overloaded'
    assert 'exhausted' in body['error']['message']


def test_overload_returns_503_for_stream_too(live_server, monkeypatch):
    """stream=true 也不能降级/挂起：请求阶段就该 503（SSE 头还没发）。"""
    _h, url = live_server

    def overloaded(engine):
        raise server.ServerOverloaded('KV slot pool exhausted (size=4, busy=4)')

    monkeypatch.setattr(server, 'kv_for_request', overloaded)
    payload = dict(_MSG)
    payload['stream'] = True
    code, headers, body = _post(url, payload)
    assert code == 503
    assert headers.get('Retry-After') == '1'
    assert body['error']['type'] == 'server_overloaded'


# ══════════════════════════════════════════════════════════════════
#  3) _collect：generate 抛异常时也必须归还 KV 槽
# ══════════════════════════════════════════════════════════════════
def test_collect_releases_kv_slot_on_generate_error(monkeypatch):
    h = _make_handle()
    Handler = server._make_handler(h)
    released = []
    kv = _FakeKV()

    monkeypatch.setattr(server, 'kv_for_request', lambda engine: kv)
    monkeypatch.setattr(server, 'kv_slot_release', lambda k: released.append(k))

    def boom(*args, **kwargs):
        yield 'x'
        raise RuntimeError('generate exploded')

    monkeypatch.setattr(server, 'generate', boom)
    chat = types.SimpleNamespace(eos=3, eos_set={3})
    with pytest.raises(RuntimeError, match='generate exploded'):
        Handler._collect(None, chat, [2, 3, 4], {})   # self 未被使用，传 None 即可
    assert released == [kv], '异常路径没有归还 KV 槽（会池泄漏）'


# ══════════════════════════════════════════════════════════════════
#  4) finish_reason：截断=length，正常=stop
# ══════════════════════════════════════════════════════════════════
def _patch_generate(monkeypatch, truncated):
    released = []

    def fake_generate(engine, ids, params, eos, eos_set=None, kv=None, state=None):
        yield '你'
        yield '好'
        if state is not None:
            state['truncated'] = truncated
            state['finish_reason'] = 'length' if truncated else 'stop'

    monkeypatch.setattr(server, 'generate', fake_generate)
    monkeypatch.setattr(server, 'kv_for_request', lambda engine: _FakeKV())
    monkeypatch.setattr(server, 'kv_slot_release', lambda k: released.append(k))
    return released


def test_finish_reason_length_on_truncation(live_server, monkeypatch):
    _h, url = live_server
    released = _patch_generate(monkeypatch, truncated=True)
    code, _headers, body = _post(url, _MSG)
    assert code == 200
    assert body['choices'][0]['finish_reason'] == 'length'
    assert body['choices'][0]['message']['content'] == '你好'
    assert len(released) == 1, '正常路径也没有归还 KV 槽'


def test_finish_reason_stop_by_default(live_server, monkeypatch):
    _h, url = live_server
    _patch_generate(monkeypatch, truncated=False)
    code, _headers, body = _post(url, _MSG)
    assert code == 200
    assert body['choices'][0]['finish_reason'] == 'stop'


def test_stream_finish_reason_length(live_server, monkeypatch):
    _h, url = live_server
    _patch_generate(monkeypatch, truncated=True)
    payload = dict(_MSG)
    payload['stream'] = True
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode('utf-8')
    frames = [ln[len('data: '):] for ln in raw.splitlines()
              if ln.startswith('data: ') and ln != 'data: [DONE]']
    chunks = [json.loads(f) for f in frames]
    assert chunks[-1]['choices'][0]['finish_reason'] == 'length'
