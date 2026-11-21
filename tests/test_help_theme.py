# -*- coding: utf-8 -*-
"""Wiki 的「浅色 / 深色」手动切换（2026-11-21 用户与朋友反馈后新增）。

背景：最初只有 `@media (prefers-color-scheme: dark)`，即**只能跟随系统**，页面上没有开关。
朋友反馈「没有浅色和深色切换」后补上，规则与语言切换一致：

    ?theme=light|dark  >  localStorage(`tgai-wiki-theme`)  >  跟随系统

显式选择写 `<html data-theme>`；没选过时不写该属性，交给媒体查询 —— 所以
**用户点过之后，系统偏好不再覆盖他的选择**（这正是之前缺的能力）。
"""
import re

from tgai_go.train import help_web


def _html():
    payload = {"schema": "tgai-help/2", "count": 0, "items": [], "categories": [],
               "kinds": [], "coverage": {}, "wiki_version": "t", "i18n": help_web._I18N}
    return help_web.render_html(payload)


def test_theme_switch_container_present():
    html = _html()
    assert 'id="theme"' in html, "顶栏必须有主题开关容器"
    assert "themeswitch" in html
    assert "function renderThemeSwitch" in html


def test_theme_css_uses_explicit_attribute_not_only_media_query():
    html = _html()
    # 显式选择
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html
    # 系统偏好只在"没选过"时生效
    assert ":root:not([data-theme])" in html
    assert "prefers-color-scheme: dark" in html


def test_theme_priority_sources_documented_in_code():
    html = _html()
    assert "tgai-wiki-theme" in html, "localStorage 键名必须在页面里"
    assert "URLSearchParams" in html, "?theme= 参数的解析必须在页面里"
    assert "applyTheme(detectTheme(), false)" in html, "首屏必须先定主题再渲染"
    # 主题与语言必须各自独立持久化
    assert "tgai-wiki-lang" in html or "wiki-lang" in html or "lang" in html


def test_theme_labels_exist_in_both_languages():
    zh = help_web._I18N["zh"]
    en = help_web._I18N["en"]
    for key in ("theme_light", "theme_dark"):
        assert key in zh and key in en, "%s 必须在两种语言里都有" % key
        assert zh[key] != en[key], "%s 的中英文案不能相同" % key
    assert zh["theme_light"] == "浅色" and zh["theme_dark"] == "深色"
    assert en["theme_light"] == "Light" and en["theme_dark"] == "Dark"


def test_i18n_key_sets_still_identical():
    assert set(help_web._I18N["zh"]) == set(help_web._I18N["en"])


def test_palette_still_monochrome():
    """加了滚动条与主题规则后，颜色仍必须只来自那套灰阶。"""
    html = _html()
    allowed = {"#111111", "#1a1a1a", "#222222", "#2a2a2a", "#666666",
               "#888888", "#e5e5e5", "#f5f5f5", "#fafafa", "#ffffff"}
    found = set(c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}", html))
    assert found <= allowed, "出现了白名单外的颜色：%s" % sorted(found - allowed)
    assert "box-shadow" not in html and "linear-gradient" not in html
    assert not re.search(r"https?://", html), "离线页不许有外链"


def test_scrollbar_is_thin_and_var_driven():
    html = _html()
    assert "scrollbar-width:thin" in html
    assert "::-webkit-scrollbar-thumb" in html
    # 不引入新颜色：滚动条颜色必须来自 CSS 变量
    thumb = re.search(r"::-webkit-scrollbar-thumb\{([^}]*)\}", html)
    assert thumb and "var(--" in thumb.group(1)


def test_header_is_compact_and_body_line_height_generous():
    html = _html()
    assert "font:14px/1.75" in html, "正文字号/行距应为 14px/1.75（朋友反馈行距偏挤）"
    assert "padding:10px 28px 9px" in html, "导航栏应更矮（朋友反馈过高）"
