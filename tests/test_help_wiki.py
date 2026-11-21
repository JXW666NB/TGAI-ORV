# -*- coding: utf-8 -*-
"""外部词条（`wiki/*.txt`）协议与热更新测试 —— 覆盖 `wiki/README_词条规范.txt` 的每条规则。

规则清单（本文件逐条钉住）：

1. 头部 = 连续的 `[键] 值` 行；`[` 允许全角 `［`；注释 `#` 整行忽略
2. 头部白名单固定：ID / 分类 / 中文名 / 英文名 / 作者；未知键 ⇒ 报错
3. 区块白名单固定：中文 / English / 示例 / 见也；未知区块 ⇒ 报错
4. 只有 `[ID]` 必填；其余可缺；中英只写一个**合法**（页面显示暂无）
5. 没有 `[ID]` 的 .txt ⇒ 加载报错（给文件名 + 最接近候选）
6. `[ID]` 与文件名不一致 ⇒ 警告但加载（以 `[ID]` 为准）
7. 编码：UTF-8/BOM 优先，回退 GB18030/GBK
8. `_meta.txt`：分类表（order 排序、重复报错、缺 other 自动补）
9. 未在 `_meta.txt` 里的分类 ⇒ 归 other 并进 check()
10. 热更新：同一进程内改文件，下一次 `T.help()` 立刻反映（缓存键 = 路径+mtime+size）
11. 缺词条也绝不空白：名字 + 自动示例 + 签名 + `未编写` 徽标
12. `contrib` 骨架 → 加载器读回（闭环）；`check/missing` 输出结构
13. 覆盖顺序：后者覆盖前者 + `shadowed_by`；未知 ID / 死链 / 未知字段都明确报错
"""

import json
import os
import sys

import pytest

import tgai_go.train as T
from tgai_go.train import help_l0, help_wiki

META = ("# 测试分类表\n"
        "[分类] tensor | 张量 | Tensor | 20\n"
        "[分类] debug | 调试 | Debug | 90\n")


# --------------------------------------------------------------------- 工具
def _write(path, text, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding=encoding, newline="\n") as fh:
        fh.write(text)
    return path


def _entry(item_id, zh="中文正文 BODY", en="English body BODY", category="tensor",
           extra_head="", extra_blocks=""):
    return ("[ID] %s\n[分类] %s\n%s\n"
            "=== 中文 ===\n%s\n\n=== English ===\n%s\n%s"
            % (item_id, category, extra_head, zh, en, extra_blocks))


def _json(*args, **kw):
    if "kind" in kw:
        kw["tag"] = kw.pop("kind")
    kw.setdefault("format", "json")
    return json.loads(T.help(*args, print=False, **kw))


def _item(item_id):
    return help_wiki.load()["by_id"][item_id]


@pytest.fixture(autouse=True)
def _isolate_wiki(monkeypatch):
    monkeypatch.delenv("TG_WIKI_PATH", raising=False)
    monkeypatch.delenv("TG_WIKI_DIR", raising=False)
    help_wiki.refresh()
    yield
    monkeypatch.delenv("TG_WIKI_PATH", raising=False)
    monkeypatch.delenv("TG_WIKI_DIR", raising=False)
    help_wiki.refresh()


@pytest.fixture
def wiki(monkeypatch, tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    _write(root / "_meta.txt", META)
    monkeypatch.setenv("TG_WIKI_PATH", str(root))
    help_wiki.refresh()
    return root


def _use_dirs(monkeypatch, *dirs):
    monkeypatch.setenv("TG_WIKI_PATH", os.pathsep.join(str(d) for d in dirs))
    help_wiki.refresh()


# ------------------------------------------------------------ 1/2/3 头部与区块
def test_minimal_entry_only_id_is_legal(wiki):
    _write(wiki / "op.var.txt", "[ID] op:var\n")
    item = _item("op:var")
    assert item["documented"] is False
    assert item["missing_langs"] == ["zh", "en"]
    assert help_wiki.load()["errors"] == []


def test_fullwidth_brackets_and_comments(wiki):
    _write(wiki / "op.var.txt",
           "# 注释整行忽略\n"
           "［ID］ op:var\n"
           "［分类］ tensor\n"
           "［中文名］ 方差\n"
           "=== 中文 ===\n中文正文\n")
    item = _item("op:var")
    assert item["title_zh"] == "方差"
    assert item["zh"] == "中文正文"
    assert item["missing_langs"] == ["en"]


def test_unknown_header_key_is_error(wiki):
    _write(wiki / "op.var.txt", "[ID] op:var\n[签名] x(self)\n=== 中文 ===\nhi\n")
    reg = help_wiki.load()
    assert reg["errors"] and reg["errors"][0]["code"] == "unknown_header"
    with pytest.raises(help_wiki.WikiError) as exc:
        help_wiki.load_strict()
    assert "修法" in str(exc.value)


def test_unknown_section_is_error(wiki):
    _write(wiki / "op.var.txt", "[ID] op:var\n=== 正文 ===\nhi\n")
    reg = help_wiki.load()
    assert reg["errors"] and reg["errors"][0]["code"] == "unknown_section"


def test_duplicate_section_is_error(wiki):
    _write(wiki / "op.var.txt",
           "[ID] op:var\n=== 中文 ===\na\n=== 中文 ===\nb\n")
    reg = help_wiki.load()
    assert reg["errors"] and reg["errors"][0]["code"] == "duplicate_section"


def test_only_one_language_is_legal(wiki):
    _write(wiki / "op.var.txt",
           "[ID] op:var\n=== English ===\nonly english\n")
    item = _item("op:var")
    assert item["documented"] is True
    assert item["zh"] == "" and item["missing_langs"] == ["zh"]


# --------------------------------------------------------------- 4/5/6 ID 规则
def test_missing_id_reports_filename_and_candidates(wiki):
    _write(wiki / "op.reshape.txt", "[分类] tensor\n=== 中文 ===\n没有 ID\n")
    reg = help_wiki.load()
    assert reg["errors"] and reg["errors"][0]["code"] == "missing_id"
    candidates = reg["errors"][0]["candidates"]
    assert "op:reshape" in candidates, candidates
    with pytest.raises(help_wiki.WikiError):
        help_wiki.load_strict()


def test_bad_id_format_is_error(wiki):
    _write(wiki / "op.var.txt", "[ID] mul\n=== 中文 ===\nx\n")
    reg = help_wiki.load()
    assert reg["errors"] and reg["errors"][0]["code"] == "bad_id"


def test_filename_mismatch_warns_but_loads(wiki):
    _write(wiki / "api.named_weird.txt", "[ID] api:help\n=== 中文 ===\nhelp\n")
    reg = help_wiki.load()
    assert reg["errors"] == []
    assert any(w["code"] == "filename_mismatch" for w in reg["warnings"])
    assert _item("api:help")["documented"] is True


# ---------------------------------------------------------------- 7 编码兼容
def test_encoding_utf8_bom(wiki):
    body = "[ID] op:var\n[分类] tensor\n=== 中文 ===\nBOM 编码测试\n"
    (wiki / "op.var.txt").write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    help_wiki.refresh()
    item = _item("op:var")
    assert item["zh"] == "BOM 编码测试"


def test_encoding_gbk_fallback(wiki):
    body = ("[ID] op:var\n[分类] tensor\n=== 中文 ===\n"
            "这是 GBK 编码的中文正文（记事本 ANSI）\n=== English ===\ngbk ok\n")
    (wiki / "op.var.txt").write_bytes(body.encode("gb18030"))
    help_wiki.refresh()
    item = _item("op:var")
    assert "GBK 编码的中文正文" in item["zh"]
    assert item["en"] == "gbk ok"


# ------------------------------------------------------------------ 8 _meta
def test_meta_order_and_other_autofill(monkeypatch, tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    _write(root / "_meta.txt",
           "[分类] debug | 调试 | Debug | 10\n"
           "[分类] tensor | 张量 | Tensor | 20\n")
    _use_dirs(monkeypatch, root)
    cats = [c["id"] for c in help_wiki.load()["categories"]]
    assert cats == ["debug", "tensor", "other"]


def test_meta_duplicate_category_is_error(wiki):
    _write(wiki / "_meta.txt", "[分类] tensor | 张量 | Tensor | 20\n"
                             "[分类] tensor | 重复 | Tensor | 30\n")
    help_wiki.refresh()
    reg = help_wiki.load()
    assert reg["meta_errors"] and reg["meta_errors"][0]["code"] == "meta_duplicate"
    assert help_wiki.check()["ok"] is False


def test_meta_bad_line_is_error(wiki):
    _write(wiki / "_meta.txt", "[分类] too | few | fields\n")
    help_wiki.refresh()
    reg = help_wiki.load()
    assert reg["meta_errors"] and reg["meta_errors"][0]["code"] == "meta_fields"


# ------------------------------------------------- 9 未映射分类 + check 报告
def test_unmapped_category_falls_back_to_other(wiki):
    _write(wiki / "op.var.txt", _entry("op:var", category="benchmark"))
    reg = help_wiki.load()
    assert _item("op:var")["category"] == "other"
    assert _item("op:var")["category_original"] == "benchmark"
    assert any(row["category"] == "benchmark" for row in reg["unmapped_categories"])
    result = help_wiki.check()
    assert any(row["category"] == "benchmark" for row in result["unmapped_categories"])


# ---------------------------------------------------------- 10/11 热更与降级
def test_hot_reload_same_process_without_refresh(wiki):
    _write(wiki / "op.reshape.txt",
           _entry("op:reshape", zh="第一版正文", en="first"))
    first = _json("op:reshape")
    assert "第一版正文" in first["items"][0]["zh"]
    version1 = first["wiki_version"]
    _write(wiki / "op.reshape.txt",
           _entry("op:reshape", zh="第二版正文（更长一点）", en="second"))
    second = _json("op:reshape")                  # 没有调 refresh
    assert "第二版正文" in second["items"][0]["zh"]
    assert second["wiki_version"] != version1


def test_hot_reload_visible_in_text_search(wiki):
    _write(wiki / "op.reshape.txt", _entry("op:reshape", zh="标记甲", en="a"))
    text1 = T.help("reshape", print=False)
    assert "标记甲" in text1
    _write(wiki / "op.reshape.txt", _entry("op:reshape", zh="标记乙", en="b"))
    text2 = T.help("reshape", print=False)
    assert "标记乙" in text2 and "标记甲" not in text2


def test_deleted_entry_still_present_with_auto_example(wiki):
    _write(wiki / "op.reshape.txt", _entry("op:reshape"))
    assert _item("op:reshape")["documented"] is True
    os.remove(str(wiki / "op.reshape.txt"))
    item = _item("op:reshape")
    assert item["documented"] is False
    assert item["example"]                       # L1 自动示例
    assert item["signature"]                     # 真实签名
    assert item["title_zh"] == ""
    payload = _json("op:reshape")
    assert payload["items"] and payload["items"][0]["documented"] is False


# --------------------------------------------------- 12 contrib / check / missing
def test_contrib_roundtrip(tmp_path, monkeypatch, capsys):
    target = tmp_path / "user_wiki"
    monkeypatch.setenv("TG_WIKI_PATH", str(target))
    help_wiki.refresh()
    info = T.help.contrib("op:view")
    out = capsys.readouterr().out
    assert info["created"] and len(info["created"]) == 1
    path = info["created"][0]
    assert os.path.isfile(path)
    assert "已生成骨架" in out and path in out
    text = open(path, encoding="utf-8").read()
    for token in ("[ID] op:view", "[分类] tensor", "=== 中文 ===",
                  "=== English ===", "=== 示例 ===", "=== 见也 ==="):
        assert token in text, token
    # 闭环：加载器能读回刚生成的骨架
    item = help_wiki.load()["by_id"]["op:view"]
    assert item["documented"] is True
    assert item["title_zh"].startswith("【待填")


def test_contrib_existing_prints_path(tmp_path, monkeypatch, capsys):
    target = tmp_path / "user_wiki"
    monkeypatch.setenv("TG_WIKI_PATH", str(target))
    help_wiki.refresh()
    path = T.help.contrib("op:var")["created"][0]
    capsys.readouterr()
    info = T.help.contrib("op:var")
    out = capsys.readouterr().out
    assert info["created"] == [] and info["existing"] == [path]
    assert path in out


def test_contrib_unknown_id_has_candidates_and_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_WIKI_PATH", str(tmp_path / "w"))
    help_wiki.refresh()
    with pytest.raises(ValueError) as exc:
        T.help.contrib("op:views")
    msg = str(exc.value)
    assert "op:view" in msg and "最接近的候选" in msg and "修法" in msg


def test_check_reports_contract(wiki):
    _write(wiki / "op.var.txt", _entry("op:var"))
    result = help_wiki.check()
    assert set(result) >= {"ok", "wiki_version", "coverage", "counts", "errors",
                          "unknown_ids", "duplicate_ids", "dead_links",
                          "unmapped_categories", "missing_translations"}
    assert result["counts"]["parsed_files"] >= 1
    total = len(help_l0.catalog())
    documented = len([it for it in help_wiki.load()["items"]
                      if it["kind"] != "plugin" and it["documented"]])
    assert result["coverage"]["total"] == total
    assert result["coverage"]["documented"] == documented
    assert abs(result["coverage"]["percent"] - round(100.0 * documented / total, 1)) < 1e-9


def test_check_prints_human_readable_report(wiki, capsys):
    _write(wiki / "op.var.txt",
           "[ID] op:var\n[分类] tensor\n=== 中文 ===\n只有中文（缺英文）\n")
    help_wiki.check()
    out = capsys.readouterr().out
    assert "T.help.check()" in out
    assert "覆盖率" in out and "中英缺边" in out and "ok =" in out


def test_missing_returns_grouped_todo_list(wiki, capsys):
    result = T.help.missing()
    out = capsys.readouterr().out
    assert result["total"] > 0
    assert result["total"] == len(result["items"])
    assert "待编写" in out
    assert all("id" in row and "example" in row for row in result["items"][:5])
    documented_ids = {it["id"] for it in help_wiki.load()["items"] if it["documented"]}
    assert not any(row["id"] in documented_ids for row in result["items"])


def test_refresh_forces_reload(wiki):
    _write(wiki / "op.var.txt", _entry("op:var", zh="缓存前"))
    assert "缓存前" in _json("op:var")["items"][0]["zh"]
    _write(wiki / "op.var.txt", _entry("op:var", zh="缓存后更长的内容"))
    T.help.refresh()
    assert "缓存后" in _json("op:var")["items"][0]["zh"]


# ------------------------------------------------- 13 覆盖顺序 / 未知 ID / 死链
def test_shadowing_higher_priority_wins_and_reports(monkeypatch, tmp_path):
    low = tmp_path / "low"
    high = tmp_path / "high"
    for root in (low, high):
        root.mkdir()
        _write(root / "_meta.txt", META)
    _write(low / "op.reshape.txt", _entry("op:reshape", zh="低优先级正文"))
    _write(high / "op.reshape.txt", _entry("op:reshape", zh="高优先级正文"))
    _use_dirs(monkeypatch, low, high)
    item = _item("op:reshape")
    assert "高优先级正文" in item["zh"]
    assert any(p.endswith("op.reshape.txt") for p in item["shadowed_by"])
    reg = help_wiki.load()
    assert reg["shadowed"].get("op:reshape")
    assert help_wiki.check()["counts"]["shadowed"] >= 1


def test_unknown_id_reported_with_candidates(wiki):
    _write(wiki / "op.no_such_engine_op.txt",
           _entry("op:no_such_engine_op"))
    reg = help_wiki.load()
    assert reg["unknown_ids"] and reg["unknown_ids"][0]["id"] == "op:no_such_engine_op"
    assert "candidates" in reg["unknown_ids"][0]
    assert all(it["id"] != "op:no_such_engine_op" for it in reg["items"])
    payload = _json()
    assert any("未知" in row["message"] for row in payload["errors"])
    with pytest.raises(help_wiki.WikiError):
        help_wiki.load_strict()


def test_dead_link_reported(wiki):
    _write(wiki / "op.var.txt",
           _entry("op:var", extra_blocks="\n=== 见也 ===\nop:no_such_link_zz\n"))
    reg = help_wiki.load()
    assert reg["dead_links"] and reg["dead_links"][0]["target"] == "op:no_such_link_zz"
    with pytest.raises(help_wiki.WikiError):
        help_wiki.load_strict()


def test_see_also_to_unwritten_but_real_id_is_not_dead(wiki):
    """`见也` 指向"引擎里有、但还没写词条"的 ID 不算错（那是待写清单）。"""
    _write(wiki / "op.var.txt",
           _entry("op:var", extra_blocks="\n=== 见也 ===\nop:gelu\n"))
    reg = help_wiki.load()
    assert reg["dead_links"] == []
    assert reg["by_id"]["op:gelu"]["documented"] is False   # 确实还没写词条


def test_plugin_entry_is_accepted_without_engine_l0(tmp_path, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    _write(root / "_meta.txt", META)
    _write(root / "plugin.demo.txt",
           _entry("plugin:demo:myop", category="tensor"))
    _use_dirs(monkeypatch, root)
    reg = help_wiki.load()
    assert reg["unknown_ids"] == []
    item = reg["by_id"]["plugin:demo:myop"]
    assert item["kind"] == "plugin" and item["documented"] is True
    payload = _json(kind="plugin")
    assert any(it["id"] == "plugin:demo:myop" for it in payload["items"])


# -------------------------------------------------------- 8' _meta 热更 + 顺序
def test_meta_hot_reload_changes_category_order(monkeypatch, tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    _write(root / "_meta.txt", "[分类] debug | 调试 | Debug | 10\n"
                             "[分类] tensor | 张量 | Tensor | 20\n")
    _use_dirs(monkeypatch, root)
    order1 = [c["id"] for c in help_wiki.web_payload()["categories"]]
    assert order1[:2] == ["debug", "tensor"]
    _write(root / "_meta.txt", "[分类] tensor | 张量 | Tensor | 10\n"
                             "[分类] debug | 调试 | Debug | 20\n")
    order2 = [c["id"] for c in help_wiki.web_payload()["categories"]]
    assert order2[:2] == ["tensor", "debug"]


# ------------------------------------------------------------ 公开解析器自测
def test_parse_text_public_entry_point():
    parsed = help_wiki.parse_text("[ID] op:var\n=== 中文 ===\n你好\n", source="x.txt")
    assert parsed["id"] == "op:var"
    assert parsed["sections"]["zh"] == ["你好"]


def test_parse_text_raises_structured_error():
    with pytest.raises(help_wiki.WikiError) as exc:
        help_wiki.parse_text("[中文名] no id\n", source="bad.txt")
    assert exc.value.errors and exc.value.errors[0]["file"] == "bad.txt"
    assert exc.value.errors[0]["code"] == "missing_id"
