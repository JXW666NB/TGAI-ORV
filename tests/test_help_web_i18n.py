# -*- coding: utf-8 -*-
"""离线 Wiki 的**全局语言切换（i18n）**测试。

用户要求（2026-11-21）：界面要有全局语言开关 `EN | 中文`，默认**英文**：
`?lang=` > localStorage > 英文（**不看** navigator.language）。
本文件钉住：

1. `_I18N` 的 zh/en key 集合**完全相同**、值非空、不返回 key 本身、占位符一致；
2. `_I18N_CORE`（纯逻辑 JS）单独在 Node 里跑：
   * `t(key)` 两种语言都返回真实文案（不是 key），缺 key 才回退/返回 key；
   * `detectLang` 优先级（url > localStorage > 默认 en，不读 navigator）；
   * `<html lang>` / `<title>` 会随 `applyHtmlLang` 更新；
3. 生成出来的 HTML：首屏静态 `<html lang="en">` + 英文 `<title>`、内嵌双语词表、
   语言开关容器存在、零外链、零阴影/渐变、调色板仍然只有灰阶；
4. web payload 的 kind 标签是**分开的 zh/en**（前端按当前语言只显示其一）。
"""

import json
import re
import shutil
import subprocess
import sys

import pytest

from tgai_go.train import help_web, help_wiki

NODE = shutil.which("node")
lang_keys = sorted(help_web._I18N["zh"])
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _render():
    return help_web.render_html(help_wiki.web_payload())


# ------------------------------------------------------------------ 1. 词表
def test_i18n_key_sets_are_identical():
    zh, en = help_web._I18N["zh"], help_web._I18N["en"]
    assert set(zh) == set(en), "zh/en 词表 key 不一致: %r" % (
        (set(zh) ^ set(en)),)


def test_i18n_values_are_real_text_not_keys():
    for lang in ("zh", "en"):
        for key, value in help_web._I18N[lang].items():
            assert isinstance(value, str) and value.strip(), (lang, key)
            assert value != key, "词表 %s[%r] 的值就是 key 本身" % (lang, key)


def test_i18n_placeholders_match_between_languages():
    for key in lang_keys:
        zh = set(PLACEHOLDER_RE.findall(help_web._I18N["zh"][key]))
        en = set(PLACEHOLDER_RE.findall(help_web._I18N["en"][key]))
        assert zh == en, "%r 的占位符不一致: zh=%r en=%r" % (key, zh, en)


def test_i18n_covers_language_switch_labels_and_default_lang_name():
    for lang in ("zh", "en"):
        assert help_web._I18N[lang]["lang_zh"] == "中文"
        assert help_web._I18N[lang]["lang_en"] == "EN"


# ------------------------------------------------- 2. 纯逻辑 JS（Node 单跑）
NODE_HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const dict = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
global.DATA = { i18n: dict };
global.state = { lang: "zh" };
global.document = { documentElement: { lang: "" }, title: "" };
eval(src);

const out = { langs: {}, fmt: {}, detect: {}, htmlLang: {}, doc: {} };
for (const lang of ["zh", "en"]) {
  state.lang = lang;
  out.langs[lang] = {};
  for (const key of Object.keys(dict.zh)) { out.langs[lang][key] = t(key); }
  out.fmt[lang] = fmt("status_total", { n: 7 });
}
state.lang = "zh";
out.t_missing_zh = t("__definitely_missing_key__");
state.lang = "en";
out.t_missing_en = t("__definitely_missing_key__");

out.detect = {
  url_wins_over_stored: detectLang("?lang=en", "zh"),
  stored_wins_over_default: detectLang("", "zh"),
  default_is_en: detectLang("", null),
  url_zh: detectLang("?x=1&lang=zh", "en"),
  url_at_end: detectLang("/index.html?a=1&lang=zh", "en"),
  junk_stored_falls_back_en: detectLang("", "fr"),
  empty: detectLang("", ""),
};
out.htmlLang = { zh: htmlLang("zh"), en: htmlLang("en") };
state.lang = "en";
applyHtmlLang("en");
out.doc.en_lang = document.documentElement.lang;
out.doc.en_title = document.title;
applyHtmlLang("zh");
out.doc.zh_lang = document.documentElement.lang;
out.doc.zh_title = document.title;
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node 不在 PATH（本机/CI 都没有 Node 时跳过 JS 单测）")
def test_js_i18n_core(tmp_path):
    core = tmp_path / "core.js"
    core.write_text(help_web._I18N_CORE, encoding="utf-8")
    i18n = tmp_path / "i18n.json"
    i18n.write_text(json.dumps(help_web._I18N, ensure_ascii=False), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(NODE_HARNESS, encoding="utf-8")
    proc = subprocess.run([NODE, str(harness), str(core), str(i18n)],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)

    # t(key)：两种语言都返回真实文案，绝不返回 key 本身
    for lang in ("zh", "en"):
        for key in lang_keys:
            assert out["langs"][lang][key] != key, (lang, key)
            assert out["langs"][lang][key] == help_web._I18N[lang][key], (lang, key)
    assert out["fmt"]["zh"] == "共 7 条"
    assert out["fmt"]["en"] == "7 entries"
    # 缺 key 才返回 key（便于测试抓漏）
    assert out["t_missing_zh"] == "__definitely_missing_key__"
    assert out["t_missing_en"] == "__definitely_missing_key__"

    # 语言判定优先级：?lang= > localStorage > 默认英文（不读 navigator）
    detect = out["detect"]
    assert detect["url_wins_over_stored"] == "en"
    assert detect["stored_wins_over_default"] == "zh"
    assert detect["default_is_en"] == "en"
    assert detect["url_zh"] == "zh"
    assert detect["url_at_end"] == "zh"
    assert detect["junk_stored_falls_back_en"] == "en"
    assert detect["empty"] == "en"

    # <html lang> / <title> 随语言更新
    assert out["htmlLang"] == {"zh": "zh-CN", "en": "en"}
    assert out["doc"]["en_lang"] == "en"
    assert out["doc"]["zh_lang"] == "zh-CN"
    assert out["doc"]["en_title"] == help_web._I18N["en"]["title"]
    assert out["doc"]["zh_title"] == help_web._I18N["zh"]["title"]


# --------------------------------------------------- 3. 生成出来的 HTML 本身
def test_html_starts_in_english_and_embeds_both_languages():
    html = _render()
    # 首屏（JS 还没跑时）就是英文，避免"先中文后英文"的闪烁
    assert '<html lang="en">' in html
    assert "<title>TGAI-ORV Wiki — offline help</title>" in html
    # 双语词表整份内嵌在 JSON 里（前端按当前语言取值），解析回来必须与 Python 词表一致
    match = re.search(
        r'<script type="application/json" id="wiki-data">(.*?)</script>', html, re.S)
    assert match, "离线页必须内嵌一份 JSON 数据"
    payload = json.loads(match.group(1))
    assert payload["i18n"] == help_web._I18N
    # 语言开关容器 + 切换相关逻辑存在
    assert 'id="lang"' in html and "langlink" in html
    assert "detectLang" in html and "applyLang" in html


def test_html_still_zero_external_resources_and_gray_palette():
    html = _render()
    assert "http://" not in html and "https://" not in html
    assert "<link" not in html.lower() and "@import" not in html
    assert "<script src" not in html.lower()
    assert "box-shadow" not in html and "linear-gradient" not in html
    palette = set(re.findall(r"#[0-9a-fA-F]{6}", html))
    allowed = {"#111111", "#222222", "#666666", "#888888", "#e5e5e5",
               "#fafafa", "#ffffff", "#1a1a1a", "#2a2a2a", "#f5f5f5"}
    assert palette <= allowed, "出现调色板外的颜色: %r" % (palette - allowed)
    for radius in re.findall(r"border-radius:([^;}]+)", html):
        assert radius.strip() in ("2px", "0", "0px"), radius


def test_language_switch_is_in_the_header_and_order_is_en_first():
    html = _render()
    # 顶栏右侧：EN | 中文（两个都可见）
    assert '"en", "zh"' in html or '["en", "zh"]' in html
    header = html.split("</header>")[0]
    assert 'id="lang"' in header
    assert "langswitch" in header


def test_default_language_is_english_without_storage():
    """`detectLang("", None)` 必须是 en —— 清空 localStorage 首访即英文。"""
    assert "return \"en\";" in help_web._I18N_CORE
    assert "navigator" not in help_web._I18N_CORE, \
        "默认英文要求**不读** navigator.language"


# ------------------------------------------------ 4. web payload 的 kind 标签
def test_web_payload_kind_labels_are_per_language():
    payload = help_wiki.web_payload()
    kinds = payload["kinds"]
    assert kinds
    by_id = {k["id"]: k for k in kinds}
    assert by_id["op"]["label_zh"] == "张量算子"
    assert by_id["op"]["label_en"] == "tensor ops"
    for row in kinds:
        assert row["label_zh"] and row["label_en"]
        assert row["label_zh"] != row["label_en"]


def test_web_payload_categories_keep_both_languages():
    payload = help_wiki.web_payload()
    for row in payload["categories"]:
        assert row["zh"] and row["en"]
        # 分类名按当前语言单独显示，不再拼成 "张量 / Tensor"
        assert " / " not in row["zh"], row
