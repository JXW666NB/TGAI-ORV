# -*- coding: utf-8 -*-
"""`T.help` 面向 AI / 脚本的接口测试（JSON 纯净性、过滤、懒加载、离线 Wiki）。

这些用例钉住的是**契约**，不是实现细节：

* `format="json"`：stdout 里除 JSON 外什么都没有（无 ANSI、不分页、无横幅），
  顶层/条目字段名固定；未命中给可执行 hint。
* 默认 `T.help()`：与外部词条无关（改 wiki 不影响它），且**惰性加载**——
  `sys.modules` 里不许出现渲染模块。
* `open="web"`：生成单个自包含 HTML（零外链），文件名/描述里的 `<`、`&`、引号
  必须安全转义；`webbrowser.open` 抛异常不能崩，路径要打印到 stdout。
"""

import json
import os
import re
import subprocess
import sys

import pytest

import tgai_go.train as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAZY_MODULES = ("tgai_go.train.help_web", "tgai_go.train.help_wiki",
                "tgai_go.train.help_l0")

#: JSON 契约（改字段名 = 破坏契约，必须同步改 `apilist.help` 的 docstring 与本测试）
TOP_KEYS = {"schema", "query", "kind", "count", "items", "hint",
            "wiki_version", "coverage", "errors", "warnings"}
ITEM_KEYS = {"id", "kind", "name", "title_zh", "title_en", "zh", "en",
             "usage", "namespace", "category", "documented", "module",
             "wiki_example", "see_also", "author",
             # 可达性自证（L0 现算；见 help_l0 顶部说明）
             "reachable", "form", "form_kind", "reach_evidence", "unreachable"}


@pytest.fixture(autouse=True)
def _reset_wiki_cache():
    """每个用例前后清 wiki 缓存，避免测试之间的热更状态互相污染。"""
    from tgai_go.train import help_wiki
    help_wiki.refresh()
    yield
    help_wiki.refresh()


def _json_of(capsys, *args, **kw):
    kw.setdefault("format", "json")
    T.help(*args, **kw)
    out = capsys.readouterr().out
    return out, json.loads(out)


# ----------------------------------------------------------------- JSON 纯净性
def test_json_is_pure_json(capsys):
    out, payload = _json_of(capsys, tag="op")
    assert out.strip().startswith("{") and out.strip().endswith("}")
    assert "\x1b[" not in out                     # 没有 ANSI 颜色
    assert "====" not in out                      # 没有文本横幅
    assert payload["count"] == len(payload["items"]) > 0


def test_json_schema_contract(capsys):
    _out, payload = _json_of(capsys, format="json")
    assert set(payload) == TOP_KEYS
    assert payload["schema"] == "tgai-help/2"
    for item in payload["items"][:50]:
        assert set(item) == ITEM_KEYS
        assert isinstance(item["id"], str) and item["id"]
        assert item["kind"] in ("op", "api", "kernel", "cli", "env", "plugin")
        assert isinstance(item["documented"], bool)
        assert isinstance(item["see_also"], list)
        # 可达性字段是**机器可判**的（不是只给人看的一句话）
        assert isinstance(item["reachable"], bool)
        assert isinstance(item["form"], str)
        assert isinstance(item["form_kind"], str)
        if item["reachable"]:
            assert item["form"] or item["kind"] == "plugin"
        else:
            assert item["unreachable"] and not item["namespace"]
    assert payload["count"] > 300                  # 引擎条目全量在里面


def test_json_filter_hit_case_insensitive(capsys):
    _o1, lower = _json_of(capsys, "mul", format="json")
    _o2, upper = _json_of(capsys, "MUL", format="json")
    assert lower["count"] > 0 and lower["query"] == "mul"
    ids = {item["id"] for item in lower["items"]}
    assert "op:mul" in ids
    assert {i["id"] for i in upper["items"]} == ids


def test_json_filter_matches_usage_signature(capsys):
    """关键词要能命中用法签名（不是只有名字/说明）。"""
    _out, payload = _json_of(capsys, "axis=None", format="json")
    by_id = {item["id"]: item for item in payload["items"]}
    assert "op:sum" in by_id
    assert "axis=None" not in by_id["op:sum"]["zh"]      # 正文里没有它
    assert "keepdims=False" in by_id["op:sum"]["usage"]  # 命中的是签名


def test_json_filter_miss_gives_actionable_hint(capsys):
    _out, payload = _json_of(capsys, "definitely_no_such_thing_zz", format="json")
    assert payload["count"] == 0 and payload["items"] == []
    assert "T.help(" in payload["hint"]
    assert "json" in payload["hint"] or "check" in payload["hint"]


def test_json_same_name_different_kind_is_distinguishable(capsys):
    _out, payload = _json_of(capsys, "abs", format="json")
    rows = [item for item in payload["items"] if item["name"] == "abs"]
    kinds = {item["kind"] for item in rows}
    assert {"op", "api"} <= kinds
    assert len({item["id"] for item in rows}) == len(rows)


def test_json_kind_filter(capsys):
    _out, payload = _json_of(capsys, tag="op", format="json")
    assert payload["count"] > 0
    assert all(item["kind"] == "op" for item in payload["items"])


def test_json_print_false_returns_string_and_is_silent(capsys):
    text = T.help("mul", format="json", print=False)
    assert capsys.readouterr().out == ""
    assert json.loads(text)["count"] > 0


def test_json_unknown_format_raises_with_fix():
    with pytest.raises(ValueError) as exc:
        T.help(format="yaml")
    assert "修法" in str(exc.value)


# ------------------------------------------------------- 默认行为 / 懒加载
def test_default_help_does_not_consult_wiki(monkeypatch, tmp_path):
    """`T.help()` 无参行为与外部词条无关（逐字不变的守护）。"""
    before = T.help(print=False)
    from tgai_go.train import help_wiki
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "op.var.txt").write_text(
        "[ID] op:var\n[分类] tensor\n=== 中文 ===\n不该影响默认总览\n",
        encoding="utf-8")
    monkeypatch.setenv("TG_WIKI_PATH", str(wiki))
    help_wiki.refresh()
    assert T.help(print=False) == before


def test_default_help_is_lazy_in_process():
    """默认路径不许把渲染/词条模块 import 进 sys.modules。"""
    sys.modules.pop("tgai_go.train.help_web", None)
    for name in LAZY_MODULES:
        sys.modules.pop(name, None)
        if hasattr(T, name.rsplit(".", 1)[-1]):
            delattr(T, name.rsplit(".", 1)[-1])
    T.help(print=False)
    present = [name for name in LAZY_MODULES if name in sys.modules]
    assert present == [], "默认 T.help() 不应导入 %s" % present


def test_default_help_is_lazy_in_fresh_process():
    code = (
        "import sys; sys.path.insert(0, 'python')\n"
        "import tgai_go.train as T\n"
        "T.help(print=False)\n"
        "print(any(m in sys.modules for m in %r))\n" % (LAZY_MODULES,)
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("False"), proc.stdout


# ---------------------------------------------------------------- 离线 Wiki
def _render_web(monkeypatch, capsys, *args, **kw):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda uri, *a, **k: opened.append(uri) or True)
    path = T.help(*args, open="web", **kw)
    out = capsys.readouterr().out
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    return path, html, out, opened


def _embedded_payload(html):
    match = re.search(r'<script type="application/json" id="wiki-data">(.*?)</script>',
                      html, re.S)
    assert match, "离线页必须内嵌一份 JSON 数据"
    return json.loads(match.group(1))


def test_web_creates_self_contained_html(monkeypatch, capsys):
    path, html, out, opened = _render_web(monkeypatch, capsys)
    assert os.path.isfile(path)
    assert out.strip() == path                    # 无头环境必须能从 stdout 拿到路径
    assert opened and opened[0].startswith("file://")
    # 零外链 / 零 CDN / 零外部字体
    assert "http://" not in html and "https://" not in html
    assert "<link" not in html.lower()
    assert "@import" not in html
    assert "<script src" not in html.lower()
    assert "\x1b[" not in html
    data = _embedded_payload(html)
    assert data["schema"] == "tgai-help/2"
    assert data["count"] == len(data["items"]) > 300
    # 页面要求的元素：kind 胶囊 / 分类侧边栏 / 中英 tab / 未编写徽标 / 页脚
    assert 'id="kinds"' in html and 'id="nav"' in html and 'id="detail"' in html
    assert 'id="foot"' in html
    assert "未编写" in html
    assert "中文" in html and "English" in html
    assert "覆盖率" in html and "wiki" in html
    # 分类顺序来自 wiki/_meta.txt（order 轴）
    assert [c["id"] for c in data["categories"]] == [
        "getting-started", "tensor", "model", "memory", "compute", "optim",
        "distributed", "plugin", "debug", "compat", "other"]
    # 体积合理（几百条 ~ 两百 KB；不许把数据复制成多份巨大字符串）
    assert len(html.encode("utf-8")) < 4 * 1024 * 1024


def test_web_query_filters_payload(monkeypatch, capsys):
    _path, html, _out, _opened = _render_web(monkeypatch, capsys)
    total = _embedded_payload(html)["count"]
    _path, html2, _out2, _opened2 = _render_web(monkeypatch, capsys, "mul")
    data = _embedded_payload(html2)
    assert data["query"] == "mul"
    assert 0 < data["count"] < total


def test_web_escapes_html_from_wiki(monkeypatch, capsys, tmp_path):
    """词条里的 `<script>`/`&`/引号必须被转义，不能原样落进 HTML。"""
    from tgai_go.train import help_wiki
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "op.view.txt").write_text(
        "[ID] op:view\n"
        "[分类] tensor\n"
        "[中文名] <script>alert('x')</script>\n"
        "[英文名] a & b \"quoted\"\n"
        "=== 中文 ===\n"
        "正文 </script><script>alert(1)</script> & <b>raw</b>\n",
        encoding="utf-8")
    monkeypatch.setenv("TG_WIKI_PATH", str(wiki))
    help_wiki.refresh()
    _path, html, _out, _opened = _render_web(monkeypatch, capsys)
    assert "<script>alert" not in html
    assert "</script><script>alert" not in html
    assert "<b>raw</b>" not in html
    assert "\\u003cscript\\u003e" in html or "\\u003cscript" in html
    data = _embedded_payload(html)
    item = [i for i in data["items"] if i["id"] == "op:view"][0]
    assert "<script>" in item["title_zh"]          # JSON 解回来后是原字符
    assert "& <b>raw</b>" in item["zh"]


def test_web_browser_failure_does_not_raise(monkeypatch, capsys):
    import webbrowser

    def boom(*args, **kwargs):
        raise RuntimeError("no browser in headless env")

    monkeypatch.setattr(webbrowser, "open", boom)
    path = T.help(open="web")                     # 不许抛异常
    out = capsys.readouterr().out
    assert os.path.isfile(path)
    assert out.strip() == path


def test_web_generation_is_fast(monkeypatch, capsys):
    """几百条目的离线页必须 1 秒内生成（实测 ~0.25s）。"""
    import time
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    start = time.perf_counter()
    T.help(open="web")
    capsys.readouterr()
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, "离线 Wiki 生成耗时 %.3fs，超过 1 秒预算" % elapsed
