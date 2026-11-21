# -*- coding: utf-8 -*-
"""L3 离线网页渲染器 —— `T.help(open="web")` 的单一自包含 HTML。

**只在 `open="web"` 时导入**（`apilist.help()` 里 `from . import help_web`），
所以引擎启动路径不为它付任何代价。

视觉：**黑白极简、文档型终端风**（参考 TGAI 官网的黑白排版感）
------------------------------------------------------------------------------------------
* 纯白底 / 纯黑字 / 发丝级分隔线（1px #e5e5e5）/ 大量留白
* 零彩色、零阴影、零渐变；圆角 ≤ 2px；信息靠字重 + 灰度层级 + 间距
* 调色板只有：`#111111 #222222 #666666 #888888 #e5e5e5 #fafafa #ffffff`
  （dark 模式额外：`#f5f5f5 #2a2a2a #1a1a1a`）
* 字体：系统 UI 字体；代码用 ui-monospace 栈；基准 14px / 1.6
* 布局：左栏 220px（分类树）+ 右侧内容最大 820px 居中，左右留白 ≥ 32px
* kind 过滤、中英切换、**全局语言切换**都是**纯文字**（当前项黑色加粗 + 底部 1px 黑线）
* `prefers-color-scheme: dark` 黑白对调；`@media print` 去侧栏与搜索框

页面（单文件、离线、无外链、无 CDN、无外部字体）：

* 顶部：品牌行（`TGAI-ORV Wiki · 离线 · wiki digest <8位>` + 右侧 `EN | 中文` 语言开关）
  + 搜索框（name / 别名 / 中英正文 / 示例 / 分类，**永远双语**，与界面语言无关）
  + kind 过滤（纯文字链接，带条目数）
* 左栏：按 `wiki/_meta.txt` 的 order 分组的分类树（可折叠、**只显示当前语言**），
  未编写条目也在这里；当前条目所在分组自动展开
* 右栏：条目详情 —— 标题 → mono 元信息行 → 调用示例块 → 中英 tab → 正文 → 补充示例；
  缺词条显示 `未编写` + `T.help.contrib('id')` 命令（永不空白）；
  L0 判定不可调用的条目**不摆假示例**，直接写原因
* 页脚：wiki digest、覆盖率、条目数、可调用/不可调用数、词条问题数

全局语言 / i18n
---------------
* `_I18N`（Python 字典）是界面文案的**唯一来源**，渲染时整份塞进内嵌 JSON，
  前端只用 `t(key)` / `fmt(key, vars)` 取值 —— 模板里不写"中文 / English"并排的硬编码。
* `_I18N_CORE`（JS）只含**纯逻辑**：`detectLang`（`?lang=` > localStorage > 默认英文）、
  `t` / `fmt`、`htmlLang` / `applyHtmlLang`；测试可以单独抽出来在 Node 里跑。
* 初始语言：`?lang=en|zh` > localStorage（`tgai-orv-wiki-lang`）> **英文**。
  **不读 `navigator.language`**：面向国际化，首访（包括中国用户）一律英文，
  中文用户自己点顶栏的「中文」；点过之后 localStorage 记住，下次仍是中文。
  首屏静态 HTML 就是 `<html lang="en">` + 英文 `<title>`，避免"先中文后英文"的闪烁。
* 切换即时生效（不重载）：`<html lang>`、`<title>`、搜索框 placeholder、kind 过滤、
  分类名、tab 默认语言、页脚、按钮全部重渲染；选择写入 localStorage。
* 词条正文 tab：默认跟随全局语言；该语言缺失则回退另一语言并显示灰色提示条；
  手动点 tab 只影响当前条目（换条目/切全局语言即恢复跟随）。

安全 / Safety
-------------
* 词条数据以**一份 JSON** 内嵌在 `<script type="application/json">` 里；JSON 文本里的
  `<` / `>` / `&` 全部转成 `\\u003c` 形式，`</script>` 无法提前闭合（防注入）。
* 前端**只用 `textContent` / `createElement` 渲染**，绝不用 `innerHTML` 拼词条内容。
* 所有动态文本都来自引擎（L0/L1）、wiki 文件（L2）或 `_I18N` 词表，不硬编码词条文案。

浏览器打不开（无头/CI/没有默认浏览器）**不算失败**：文件已经写好，路径照常返回；
`apilist.help()` 会把路径打印到 stdout。
"""

import json
import os
import pathlib
import tempfile
import webbrowser

__all__ = ["render_html", "open_web", "_I18N", "_I18N_CORE"]

# ------------------------------------------------------------------ 界面词表
# ★ 界面文案的唯一来源：zh / en 的 key 集合必须完全相同（tests/test_help_web_i18n.py 钉住）。
#   `{n}` / `{total}` 这类占位符由前端 `fmt(key, vars)` 替换。
_I18N = {
    "zh": {
        "title": "TGAI-ORV Wiki · 离线帮助",
        "tagline": "· 离线 ·",
        "search_ph": "搜索 ID / 名字 / 别名 / 中英正文 / 示例…",
        "lang_zh": "中文",
        "lang_en": "EN",
        "all_kind": "全部",
        "status_total": "共 {n} 条",
        "status_match": "匹配 {n} / {total} 条",
        "status_kind": " · kind={k}",
        "nav_empty": "没有命中。换个关键词试试。",
        "unwritten": "未编写",
        "not_callable": "不可调用",
        "call_heading": "调用示例（引擎按真实签名生成）",
        "no_callable": "⚠ 这一条在引擎里没有可调用的入口：",
        "signature": "签名",
        "module": "所属",
        "evidence": "判定依据",
        "undocumented_title": "这一条还没有词条（未编写）。",
        "undocumented_body": "引擎已经把名字、签名、调用示例列在上面，所以页面不会空白；"
                             "欢迎补一条词条（txt 放进 wiki/ 即热生效）：",
        "copy_cmd": "复制贡献命令",
        "copied": "已复制",
        "manual_copy": "请手动复制上面的命令",
        "tab_zh": "中文",
        "tab_en": "English",
        "no_zh": "暂无中文（不做机翻）",
        "no_en": "暂无 English",
        "fallback": "该条目缺{lang}正文，已回退到{other}。",
        "extra_heading": "补充示例（来自词条）",
        "see_heading": "见也",
        "see_missing": "（不存在）",
        "problems_heading": "词条校验问题（{n}）",
        "problems_hint": "运行 T.help.check() 看明细。",
        "default_meta": "离线帮助 · 单文件自包含 · 无外部资源",
        "hint_heading": "给 AI 的提示",
        "hint_text": "这页和 `T.help()` 存在的唯一理由：**不要靠记忆猜 API**。"
                     "ORV 与 PyTorch 命名相近，但边界处有真实差异 —— 例如 `T.sum(x)` "
                     "不存在（张量是 `x.sum()`），凭记忆猜一次就可能白烧一轮 GPU。"
                     "不确定时先查，别猜。",
        "json_note": "JSON 契约：schema = \"tgai-help/2\"（字段名固定）；"
                     "词条是外部数据（wiki/*.txt），改文件后同一进程内下一次调用立刻生效。",
        "hint_code": "import tgai_go.train as T\n\n"
                     "T.help(\"关键词\")                     # 文本（人看）\n"
                     "T.help(\"关键词\", format=\"json\")      # 纯 JSON：stdout 里只有 JSON\n"
                     "T.help(open=\"web\")                   # 生成本页离线 Wiki 并打印路径\n"
                     "T.help.check() / T.help.missing()    # 词条健康度 / 待写清单",
        "footer_wiki": "wiki 版本 {v}",
        "footer_coverage": "覆盖率 {p}%（{d}/{t}）",
        "footer_items": "本页条目 {n}",
        "footer_callable": "可调用 {r}/{n}",
        "footer_unreachable": "不可调用 {n}",
        "footer_problems": "词条问题 {n}（T.help.check()）",
        "footer_offline": "离线自包含，无外部资源",
        "author": "作者",
        "top": "顶部",
    },
    "en": {
        "title": "TGAI-ORV Wiki — offline help",
        "tagline": "· offline ·",
        "search_ph": "Search ID / name / alias / zh+en text / example…",
        "lang_zh": "中文",
        "lang_en": "EN",
        "all_kind": "all",
        "status_total": "{n} entries",
        "status_match": "{n} / {total} matched",
        "status_kind": " · kind={k}",
        "nav_empty": "No match. Try another keyword.",
        "unwritten": "not written",
        "not_callable": "not callable",
        "call_heading": "Call example (generated from the real signature)",
        "no_callable": "⚠ This entry has no callable entry point in the engine: ",
        "signature": "Signature",
        "module": "Module",
        "evidence": "Evidence",
        "undocumented_title": "This entry has no wiki text yet.",
        "undocumented_body": "The engine already lists the name, signature and call "
                             "example above, so this page is never blank. Contributions "
                             "are welcome (drop a txt into wiki/ and it hot-reloads):",
        "copy_cmd": "copy command",
        "copied": "Copied",
        "manual_copy": "please copy the command above manually",
        "tab_zh": "中文",
        "tab_en": "English",
        "no_zh": "No Chinese text yet (no machine translation)",
        "no_en": "No English text yet",
        "fallback": "No {lang} text for this entry; showing {other} instead.",
        "extra_heading": "Extra example (from the wiki entry)",
        "see_heading": "See also",
        "see_missing": " (missing)",
        "problems_heading": "Wiki problems ({n})",
        "problems_hint": "Run T.help.check() for details.",
        "default_meta": "offline help · single self-contained file · no external resources",
        "hint_heading": "Hints for AI agents",
        "hint_text": "The only reason this page and `T.help()` exist: **do not guess "
                     "APIs from memory**. ORV is close to PyTorch but differs at the "
                     "edges -- e.g. there is no `T.sum(x)` (tensors are `x.sum()`), and "
                     "one wrong guess can burn a whole GPU round. Look it up when unsure.",
        "json_note": "JSON contract: schema = \"tgai-help/2\" (field names are fixed); "
                     "wiki text is external data (wiki/*.txt) and hot-reloads on the next "
                     "call in the same process.",
        "hint_code": "import tgai_go.train as T\n\n"
                     "T.help(\"keyword\")                    # human-readable text\n"
                     "T.help(\"keyword\", format=\"json\")     # pure JSON on stdout\n"
                     "T.help(open=\"web\")                   # render this offline wiki + path\n"
                     "T.help.check() / T.help.missing()    # wiki health / missing entries",
        "footer_wiki": "wiki digest {v}",
        "footer_coverage": "coverage {p}% ({d}/{t})",
        "footer_items": "items on page {n}",
        "footer_callable": "callable {r}/{n}",
        "footer_unreachable": "not callable {n}",
        "footer_problems": "wiki problems {n} (T.help.check())",
        "footer_offline": "offline & self-contained; no external resources",
        "author": "by",
        "top": "Top",
    },
}

# ------------------------------------------------------------------ 纯逻辑 JS
# ★ 只放**不碰 DOM 渲染**的语言逻辑：测试用 Node 单独 eval 这一块（见 tests/test_help_web_i18n.py）。
#   依赖的全局：DATA（内嵌 payload）、state（含 lang）、document（仅 applyHtmlLang）。
_I18N_CORE = r"""
/* i18n-core-begin */
var I18N = (DATA && DATA.i18n) || { zh: {}, en: {} };
var LANG_KEY = "tgai-orv-wiki-lang";

// 优先级：?lang= > localStorage > **默认英文**（故意不读浏览器语言）。
function detectLang(search, stored) {
  var m = /[?&]lang=(zh|en)(?:&|$)/.exec(String(search || ""));
  if (m) { return m[1]; }
  if (stored === "zh" || stored === "en") { return stored; }
  return "en";
}

// 按指定语言取词条文案（缺 key 回退中文，再缺返回 key 本身）
function tIn(lang, key) {
  var dict = I18N[lang] || I18N.zh || {};
  var out = dict[key];
  if (out === undefined || out === null) { out = (I18N.zh || {})[key]; }
  if (out === undefined || out === null) { return key; }
  return out;
}

// 取当前界面语言的文案
function t(key) {
  var lang = (typeof state !== "undefined" && state && state.lang) ? state.lang : "zh";
  return tIn(lang, key);
}

// 占位符替换：fmt("status_total", {n: 42})
function fmt(key, vars) {
  var text = String(t(key));
  var data = vars || {};
  Object.keys(data).forEach(function (k) {
    text = text.replace("{" + k + "}", String(data[k]));
  });
  return text;
}

function htmlLang(lang) { return lang === "zh" ? "zh-CN" : "en"; }

// 切换语言时间步更新 <html lang> 与 <title>
function applyHtmlLang(lang) {
  if (typeof document !== "undefined" && document) {
    if (document.documentElement) { document.documentElement.lang = htmlLang(lang); }
    document.title = tIn(lang, "title");
  }
  return htmlLang(lang);
}
/* i18n-core-end */
"""

# ------------------------------------------------------------------ 静态 CSS
# 调色板只有 8 个 hex（dark 模式额外 3 个），零彩色 / 零阴影 / 零渐变 / 圆角 <= 2px。
_CSS = """
:root{--fg:#111111;--body:#222222;--dim:#666666;--faint:#888888;
--line:#e5e5e5;--code:#fafafa;--bg:#ffffff}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--body);
font:14px/1.6 -apple-system,"Segoe UI","Helvetica Neue",Arial,"Noto Sans SC",sans-serif}
code,pre,.mono{font-family:ui-monospace,"SF Mono",Consolas,ui-monospace,monospace}
header{position:sticky;top:0;z-index:10;background:var(--bg);
border-bottom:1px solid var(--line);padding:16px 32px}
.brand{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.brand-title{margin:0;font-size:24px;font-weight:600;color:var(--fg);letter-spacing:-.01em}
.sub,.ver{font-size:12px;color:var(--faint)}
.ver{font-family:ui-monospace,"SF Mono",Consolas,ui-monospace,monospace}
.langswitch{display:flex;gap:14px;margin-left:auto;align-items:baseline}
.langlink{border:0;background:none;padding:0 0 3px;margin:0;cursor:pointer;font:inherit;
font-size:13px;color:var(--dim);border-bottom:1px solid transparent}
.langlink:hover{color:var(--fg)}
.langlink.on{color:var(--fg);font-weight:600;border-bottom-color:var(--fg)}
.searchrow{display:flex;gap:16px;align-items:center;margin-top:14px;flex-wrap:wrap}
#q{flex:1 1 320px;max-width:520px;padding:7px 10px;border:1px solid var(--line);
border-radius:2px;background:var(--bg);color:var(--fg);font:inherit;font-size:14px;
outline:none}
#q:focus{border-color:var(--fg)}
#status{font-size:12px;color:var(--faint)}
.kinds{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-top:12px}
.kindlink{border:0;background:none;padding:0 0 3px;margin:0;cursor:pointer;font:inherit;
font-size:13px;color:var(--dim);border-bottom:1px solid transparent}
.kindlink:hover{color:var(--fg)}
.kindlink .n{color:var(--faint);font-size:12px;margin-left:4px}
.kindlink.on{color:var(--fg);font-weight:600;border-bottom-color:var(--fg)}
.app{display:grid;grid-template-columns:220px minmax(0,1fr);max-width:1160px;margin:0 auto}
aside{height:calc(100vh - 168px);position:sticky;top:168px;overflow:auto;
border-right:1px solid var(--line);padding:20px 18px 60px}
details.cat{border-top:1px solid var(--line);padding:9px 0}
details.cat>summary{cursor:pointer;list-style:none;display:flex;justify-content:space-between;
gap:10px;padding:1px 0;font-size:13px;color:var(--fg)}
details.cat>summary::-webkit-details-marker{display:none}
details.cat>summary .n{color:var(--faint);font-size:12px;font-variant-numeric:tabular-nums}
.entry{display:block;width:100%;text-align:left;border:0;background:none;font:inherit;
font-size:13px;color:var(--body);cursor:pointer;padding:3px 0 3px 12px;
border-left:2px solid transparent}
.entry:hover{color:var(--fg)}
.entry.on{color:var(--fg);font-weight:600;border-left-color:var(--fg)}
.entry .k{color:var(--faint);font-size:11px;margin-right:6px}
.entry .m{color:var(--dim);font-size:11px;margin-left:6px}
main{min-width:0;max-width:820px;margin:0 auto;padding:28px 32px 72px}
.entry-title{margin:0 0 6px;font-size:24px;font-weight:600;line-height:1.3;
color:var(--fg);letter-spacing:-.01em;word-break:break-word}
.meta{font-family:ui-monospace,"SF Mono",Consolas,ui-monospace,monospace;
font-size:12px;color:var(--faint);margin:0 0 22px;word-break:break-word}
.meta .miss{color:var(--dim)}
.block{border-top:1px solid var(--line);padding:18px 0 4px;margin:0 0 4px}
.block:first-of-type{border-top:0;padding-top:6px}
.block h3{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--dim);
letter-spacing:.05em;text-transform:uppercase}
pre.code{margin:0;background:var(--code);border:1px solid var(--line);border-radius:2px;
padding:10px 12px;overflow:auto;font-size:12.5px;line-height:1.55;color:var(--fg);
white-space:pre}
.sig{font-family:ui-monospace,"SF Mono",Consolas,ui-monospace,monospace;
font-size:12px;color:var(--faint);margin:8px 0 0;word-break:break-word}
.tabs{display:flex;gap:18px;border-bottom:1px solid var(--line);margin:0 0 14px}
.tab{border:0;background:none;font:inherit;font-size:13px;color:var(--faint);
cursor:pointer;padding:0 0 7px;margin:0 0 -1px;border-bottom:1px solid transparent}
.tab:hover{color:var(--fg)}
.tab.on{color:var(--fg);font-weight:600;border-bottom-color:var(--fg)}
.prose{color:var(--body)}
.prose p{margin:0 0 12px}
.prose p:last-child{margin-bottom:0}
.prose code{background:var(--code);border:1px solid var(--line);border-radius:2px;
padding:0 4px;font-size:12.5px;color:var(--fg);word-break:break-word}
.prose strong{color:var(--fg);font-weight:600}
.muted{color:var(--faint)}
.note{border-top:1px solid var(--line);border-bottom:1px solid var(--line);
background:var(--code);padding:14px 16px;margin:0 0 18px;color:var(--body)}
.note p{margin:6px 0}
.note .cmd{margin:10px 0;background:var(--bg);border:1px solid var(--line);
border-radius:2px;padding:8px 10px;overflow:auto;font-size:12.5px;color:var(--fg)}
.note.fallback{background:var(--code);border-top:1px solid var(--line);
border-bottom:1px solid var(--line);padding:10px 12px;margin:0 0 14px;
color:var(--dim);font-size:12.5px}
.linkrow{display:flex;flex-wrap:wrap;gap:16px}
.linkbtn{border:0;background:none;font:inherit;font-size:13px;color:var(--fg);
cursor:pointer;padding:0;margin:0;text-decoration:underline;text-underline-offset:2px}
.linkbtn:hover{color:var(--dim)}
.linkbtn:disabled{color:var(--faint);text-decoration:none;cursor:default}
.empty{border-top:1px solid var(--line);padding:24px 0;color:var(--dim)}
footer{max-width:1160px;margin:0 auto;border-top:1px solid var(--line);
padding:14px 32px 28px;color:var(--faint);font-size:12px;display:flex;
gap:20px;flex-wrap:wrap;
font-family:ui-monospace,"SF Mono",Consolas,ui-monospace,monospace}
footer b{color:var(--fg);font-weight:600}
mark{background:var(--code);color:var(--fg);font-weight:600;padding:0}
#toTop{position:fixed;right:24px;bottom:20px;border:1px solid var(--line);
background:var(--bg);color:var(--faint);border-radius:2px;padding:4px 9px;
font-size:12px;text-decoration:none}
@media (prefers-color-scheme: dark){
:root{--fg:#f5f5f5;--body:#f5f5f5;--dim:#888888;--faint:#888888;--line:#2a2a2a;
--code:#1a1a1a;--bg:#111111}
}
@media (max-width:820px){
.app{grid-template-columns:1fr}
aside{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);
padding:12px 20px 16px}
header{padding:12px 20px}
main{padding:20px 20px 48px}
}
@media print{
aside,#q,#kinds,#status,#toTop,.langswitch{display:none}
.app{display:block}
header{position:static;border-bottom:1px solid var(--line)}
main{max-width:none;padding:0}
body{background:#ffffff;color:#222222}
}
""".strip()

# ------------------------------------------------------------------- 静态 JS
# 注意：这里**不拼任何词条文本**，全部从内嵌 JSON 读；渲染只用 textContent/createElement。
# 界面文案一律走 t()/fmt()（词表在 DATA.i18n），模板里不写"中文 / English"并排的硬编码。
_JS = (r"""
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("wiki-data").textContent);
  var ITEMS = (DATA.items || []).slice();
  var BY_ID = {};
  ITEMS.forEach(function (it) { BY_ID[it.id] = it; });
  var CATS = (DATA.categories || []).slice();
  var CAT_BY_ID = {};
  CATS.forEach(function (c) { CAT_BY_ID[c.id] = c; });
  var KINDS = (DATA.kinds || []).slice();
  var state = { q: (DATA.query || ""), kind: "", current: "", lang: "zh", tab: "" };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

/*__I18N_CORE__*/

  // ---- 语言相关的小工具（全部走 _I18N 词表）----
  function langName(lang) { return lang === "zh" ? t("tab_zh") : t("tab_en"); }
  function otherLang(lang) { return lang === "zh" ? "en" : "zh"; }
  function catName(cat, lang) {
    if (!cat) { return ""; }
    var name = cat[lang];
    if (!name) { name = cat[otherLang(lang)] || cat.id || ""; }
    return name || "";
  }
  function kindName(kind) {
    var label = state.lang === "zh" ? kind.label_zh : kind.label_en;
    return label || kind.id;
  }
  function displayTitle(it, lang) {
    var first = lang === "zh" ? it.title_zh : it.title_en;
    var second = lang === "zh" ? it.title_en : it.title_zh;
    return first || second || it.name;
  }
  function readStoredLang() {
    try {
      var v = window.localStorage.getItem(LANG_KEY);
      return (v === "zh" || v === "en") ? v : null;
    } catch (e) { return null; }
  }
  function storeLang(lang) {
    try { window.localStorage.setItem(LANG_KEY, lang); } catch (e) {}
  }

  // 安全高亮：只用文本节点 + <mark>，不碰 innerHTML
  function appendHighlighted(parent, text, needle) {
    text = String(text === null || text === undefined ? "" : text);
    needle = String(needle || "").toLowerCase();
    if (!needle) { parent.appendChild(document.createTextNode(text)); return; }
    var low = text.toLowerCase(), pos = 0;
    while (pos < text.length) {
      var hit = low.indexOf(needle, pos);
      if (hit < 0) { parent.appendChild(document.createTextNode(text.slice(pos))); break; }
      if (hit > pos) { parent.appendChild(document.createTextNode(text.slice(pos, hit))); }
      var mark = document.createElement("mark");
      mark.textContent = text.slice(hit, hit + needle.length);
      parent.appendChild(mark);
      pos = hit + needle.length;
    }
  }

  // Markdown-lite：只认 `反引号` 与 **加粗**；段间空行分段。全部 DOM 构造，天然防注入。
  function appendInline(parent, text) {
    var re = /(`[^`]*`|\*\*[^*]+\*\*)/g;
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) { parent.appendChild(document.createTextNode(text.slice(last, m.index))); }
      var token = m[0];
      if (token.charAt(0) === "`") {
        parent.appendChild(el("code", null, token.slice(1, -1)));
      } else {
        parent.appendChild(el("strong", null, token.slice(2, -2)));
      }
      last = m.index + token.length;
    }
    if (last < text.length) { parent.appendChild(document.createTextNode(text.slice(last))); }
  }
  function renderProse(container, markdown) {
    var text = String(markdown || "").trim();
    if (!text) { return false; }
    var paras = text.split(/\n{2,}/);
    paras.forEach(function (para) {
      var p = el("p");
      var lines = para.split("\n");
      lines.forEach(function (line, idx) {
        if (idx > 0) { p.appendChild(document.createElement("br")); }
        appendInline(p, line);
      });
      container.appendChild(p);
    });
    return true;
  }

  // ★ 搜索**永远中英双语**：不管界面语言，中英名字/正文/示例/分类/ID 全进同一个 blob。
  function matchItem(it) {
    var q = state.q.trim().toLowerCase();
    if (state.kind && it.kind !== state.kind) { return false; }
    if (!q) { return true; }
    var blob = [it.id, it.kind, it.name, it.title_zh, it.title_en, it.zh, it.en,
                it.usage, it.namespace, it.category, it.module, it.wiki_example,
                it.form, (it.see_also || []).join(" ")].join(" ").toLowerCase();
    return blob.indexOf(q) >= 0;
  }

  function filtered() { return ITEMS.filter(matchItem); }

  function renderKinds() {
    var box = $("kinds"); clear(box);
    var all = el("button", "kindlink" + (state.kind ? "" : " on"));
    all.appendChild(document.createTextNode(t("all_kind")));
    all.appendChild(el("span", "n", String(ITEMS.length)));
    all.onclick = function () { state.kind = ""; renderAll(); };
    box.appendChild(all);
    KINDS.forEach(function (k) {
      var b = el("button", "kindlink" + (state.kind === k.id ? " on" : ""));
      b.appendChild(document.createTextNode(kindName(k)));
      b.appendChild(el("span", "n", String(k.count)));
      b.onclick = function () { state.kind = (state.kind === k.id ? "" : k.id); renderAll(); };
      box.appendChild(b);
    });
  }

  function renderNav(list) {
    var nav = $("nav"); clear(nav);
    if (!list.length) {
      nav.appendChild(el("div", "empty", t("nav_empty")));
      return;
    }
    var catOrder = CATS.map(function (c) { return c.id; });
    var byCat = {};
    list.forEach(function (it) {
      var cid = (it.category && catOrder.indexOf(it.category) >= 0) ? it.category : "other";
      (byCat[cid] = byCat[cid] || []).push(it);
    });
    // 当前条目所在分类必须展开，否则左侧看不到"我选中了哪一条"。
    var current = BY_ID[state.current];
    var currentCat = "";
    if (current) {
      currentCat = (current.category && catOrder.indexOf(current.category) >= 0)
        ? current.category : "other";
    }
    var opened = 0;
    catOrder.forEach(function (cid) {
      var rows = byCat[cid];
      if (!rows) { return; }
      var cat = CAT_BY_ID[cid] || { id: cid };
      var details = document.createElement("details");
      details.className = "cat";
      details.open = (state.q.length > 0) || (cid === currentCat)
                     || (opened === 0 && !currentCat);
      opened += 1;
      var summary = document.createElement("summary");
      summary.appendChild(el("span", null, catName(cat, state.lang) || cid));
      summary.appendChild(el("span", "n", rows.length + ""));
      details.appendChild(summary);
      var body = el("div", "catbody");
      rows.forEach(function (it) {
        var b = el("button", "entry" + (state.current === it.id ? " on" : ""));
        b.appendChild(el("span", "k", it.kind));
        var title = displayTitle(it, state.lang);
        var tnode = el("span");
        appendHighlighted(tnode, title, state.q);
        b.appendChild(tnode);
        if (!it.documented) { b.appendChild(el("span", "m", t("unwritten"))); }
        b.onclick = function () { select(it.id, true); };
        body.appendChild(b);
      });
      details.appendChild(body);
      nav.appendChild(details);
    });
  }

  function contractHint() {
    var pre = el("pre", "code");
    pre.textContent = t("hint_code");
    return pre;
  }

  function metaLine(it, cat) {
    var line = el("div", "meta");
    var parts = [it.id, it.kind];
    if (cat) { parts.push(catName(cat, state.lang) || it.category); }
    parts.forEach(function (part, idx) {
      if (idx > 0) { line.appendChild(document.createTextNode("  ·  ")); }
      line.appendChild(document.createTextNode(part));
    });
    if (!it.documented) { line.appendChild(el("span", "miss", "  ·  " + t("unwritten"))); }
    if (it.reachable === false) {
      line.appendChild(el("span", "miss", "  ·  " + t("not_callable")));
    }
    if (it.author) { line.appendChild(document.createTextNode("  ·  " + t("author") + " " + it.author)); }
    return line;
  }

  function renderDefault(main) {
    main.appendChild(el("h2", "entry-title", "TGAI-ORV Wiki"));
    main.appendChild(el("div", "meta", t("default_meta")));
    var block = el("section", "block");
    block.appendChild(el("h3", null, t("hint_heading")));
    var p = el("p");
    appendInline(p, t("hint_text"));
    block.appendChild(p);
    block.appendChild(contractHint());
    block.appendChild(el("p", "muted", t("json_note")));
    if (DATA.hint) { block.appendChild(el("p", "muted", DATA.hint)); }
    main.appendChild(block);
  }

  function renderDetail() {
    var main = $("detail"); clear(main);
    var it = BY_ID[state.current];
    if (!it) { renderDefault(main); return; }
    var cat = CAT_BY_ID[it.category] || { id: it.category };
    main.appendChild(el("h2", "entry-title", displayTitle(it, state.lang)));
    main.appendChild(metaLine(it, cat));

    // L1：引擎按真实签名机械生成的调用示例与签名（没有词条也永远存在）。
    // 可达性由 L0 现算：不可调用的条目**不许**摆一个假示例，要明说原因。
    var call = el("section", "block");
    call.appendChild(el("h3", null, t("call_heading")));
    if (it.reachable === false) {
      call.appendChild(el("p", "muted", t("no_callable") + (it.unreachable || "?")));
    } else {
      call.appendChild(el("pre", "code", it.namespace || "(?)"));
    }
    call.appendChild(el("p", "sig", t("signature") + ": " + (it.usage || "(?)")));
    if (it.module) { call.appendChild(el("p", "sig", t("module") + ": " + it.module)); }
    if (it.reachable === false && it.reach_evidence) {
      call.appendChild(el("p", "sig", t("evidence") + ": " + it.reach_evidence));
    }
    main.appendChild(call);

    if (!it.documented) {
      var note = el("div", "note");
      note.appendChild(el("strong", null, t("undocumented_title")));
      note.appendChild(el("p", "muted", t("undocumented_body")));
      var cmd = "T.help.contrib('" + it.id + "')";
      note.appendChild(el("pre", "cmd", cmd));
      var btn = el("button", "linkbtn", t("copy_cmd"));
      btn.onclick = function () {
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(cmd);
            btn.textContent = t("copied");
          } else {
            btn.textContent = t("manual_copy");
          }
        } catch (e) { btn.textContent = t("manual_copy"); }
      };
      note.appendChild(btn);
      main.appendChild(note);
    }

    // 中英 Tab：默认跟随全局语言；该语言缺失 ⇒ 回退另一语言 + 灰色提示条。
    var block = el("section", "block");
    var tabs = el("div", "tabs");
    var bodyBox = el("div", "prose");
    var notice = el("div", "note fallback");
    notice.style.display = "none";
    var pref = state.tab || state.lang;
    var shown = pref;
    if (pref === "zh" && !it.zh && it.en) { shown = "en"; }
    if (pref === "en" && !it.en && it.zh) { shown = "zh"; }
    var tabZh = el("button", "tab", t("tab_zh"));
    var tabEn = el("button", "tab", t("tab_en"));
    function show(lang) {
      shown = lang;
      tabZh.className = "tab" + (lang === "zh" ? " on" : "");
      tabEn.className = "tab" + (lang === "en" ? " on" : "");
      clear(bodyBox);
      var text = lang === "zh" ? it.zh : it.en;
      if (!text) {
        bodyBox.appendChild(el("p", "muted",
          lang === "zh" ? t("no_zh") : t("no_en")));
        notice.style.display = "none";
        return;
      }
      renderProse(bodyBox, text);
      var wantGlobal = state.tab || state.lang;
      if (lang !== wantGlobal) {
        // 提示条说的是"缺**想要的那门语言**，已回退到当前显示的语言"
        var missing = wantGlobal === "zh" ? t("tab_zh") : t("tab_en");
        var fallbackTo = lang === "zh" ? t("tab_zh") : t("tab_en");
        notice.textContent = fmt("fallback", { lang: missing, other: fallbackTo });
        notice.style.display = "";
      } else {
        notice.style.display = "none";
      }
    }
    tabZh.onclick = function () { state.tab = "zh"; renderDetail(); };
    tabEn.onclick = function () { state.tab = "en"; renderDetail(); };
    tabs.appendChild(tabZh);
    tabs.appendChild(tabEn);
    block.appendChild(tabs);
    block.appendChild(notice);
    block.appendChild(bodyBox);
    main.appendChild(block);
    show(shown);

    if (it.wiki_example) {
      var ex = el("section", "block");
      ex.appendChild(el("h3", null, t("extra_heading")));
      ex.appendChild(el("pre", "code", it.wiki_example));
      main.appendChild(ex);
    }

    if (it.see_also && it.see_also.length) {
      var see = el("section", "block");
      see.appendChild(el("h3", null, t("see_heading")));
      var row = el("div", "linkrow");
      it.see_also.forEach(function (target) {
        var known = !!BY_ID[target];
        var b = el("button", "linkbtn", target + (known ? "" : t("see_missing")));
        if (known) { b.onclick = function () { select(target, true); }; }
        else { b.disabled = true; }
        row.appendChild(b);
      });
      see.appendChild(row);
      main.appendChild(see);
    }

    if (DATA.errors && DATA.errors.length) {
      var err = el("section", "block");
      err.appendChild(el("h3", null, fmt("problems_heading", { n: DATA.errors.length })));
      err.appendChild(el("p", "muted", t("problems_hint")));
      main.appendChild(err);
    }
  }

  function renderStatus(list) {
    var box = $("status");
    var q = state.q.trim();
    // 关键：kind 过滤生效时显示**过滤后的条数**，不要显示全量（否则看起来像没生效）。
    var shown = (q || state.kind) ? list.length : ITEMS.length;
    box.textContent = (q
      ? fmt("status_match", { n: list.length, total: ITEMS.length })
      : fmt("status_total", { n: shown })) +
      (state.kind ? fmt("status_kind", { k: state.kind }) : "");
  }

  function renderAll() {
    var list = filtered();
    // 先定"选中哪一条"，再渲染侧栏：否则侧栏展开的分类可能不是当前条目所在分类。
    // 条目变了就清掉手动 tab 选择（手动 tab 只影响当时那一条）。
    var next = state.current;
    if (!BY_ID[next] || !list.some(function (it) { return it.id === next; })) {
      next = list.length ? list[0].id : "";
    }
    if (next !== state.current) { state.current = next; state.tab = ""; }
    renderKinds();
    renderNav(list);
    renderStatus(list);
    renderDetail();
  }

  function select(id, scroll) {
    if (!BY_ID[id]) { return; }
    state.current = id;
    state.tab = "";                     // 换条目 ⇒ tab 恢复跟随全局语言
    try { history.replaceState(null, "", "#" + encodeURIComponent(id)); } catch (e) {}
    renderNav(filtered());
    renderDetail();
    if (scroll) {
      var mainEl = $("detail");
      if (mainEl && mainEl.scrollIntoView) { mainEl.scrollIntoView({ block: "start" }); }
    }
  }

  function renderFooter() {
    var foot = $("foot"); clear(foot);
    function add(label, value) {
      var span = el("span");
      span.appendChild(document.createTextNode(label + " "));
      span.appendChild(el("b", null, value));
      foot.appendChild(span);
    }
    var cov = DATA.coverage || { documented: 0, total: 0, percent: 0 };
    add(fmt("footer_wiki", { v: DATA.wiki_version || "?" }), "");
    add(fmt("footer_coverage",
      { p: cov.percent, d: cov.documented, t: cov.total }), "");
    add(fmt("footer_items", { n: DATA.count }), "");
    // 可达性自证结果（L0 现算）：允许"列出来但不可调用"，但必须让人一眼看到有几条。
    var unreachable = ITEMS.filter(function (it) { return it.reachable === false; });
    add(fmt("footer_callable",
      { r: ITEMS.length - unreachable.length, n: ITEMS.length }), "");
    if (unreachable.length) {
      add(fmt("footer_unreachable", { n: unreachable.length }), "");
    }
    if (DATA.errors && DATA.errors.length) {
      add(fmt("footer_problems", { n: DATA.errors.length }), "");
    }
    foot.appendChild(el("span", "muted", t("footer_offline")));
  }

  function renderLangSwitch() {
    var box = $("lang"); if (!box) { return; }
    clear(box);
    // 顶栏右侧固定 `EN | 中文`：两个都可见、两个都能点，中文用户一眼能找到。
    ["en", "zh"].forEach(function (lang) {
      var b = el("button", "langlink" + (state.lang === lang ? " on" : ""),
                 lang === "zh" ? t("lang_zh") : t("lang_en"));
      b.setAttribute("data-lang", lang);
      b.onclick = function () { applyLang(lang, true); };
      box.appendChild(b);
    });
  }

  function applyLang(lang, persist) {
    state.lang = (lang === "en") ? "en" : "zh";
    state.tab = "";
    if (persist) { storeLang(state.lang); }
    applyHtmlLang(state.lang);
    var tag = $("tagline"); if (tag) { tag.textContent = t("tagline"); }
    var q = $("q"); if (q) { q.placeholder = t("search_ph"); }
    var top = $("toTop"); if (top) { top.textContent = t("top"); }
    renderLangSwitch();
    renderAll();
    renderFooter();
  }

  function bindSearch() {
    var q = $("q");
    q.value = state.q;
    q.addEventListener("input", function () {
      state.q = q.value;
      renderAll();          // 搜到谁就选中谁：详情面板跟着搜索结果走，不留旧条目
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); }
      if (e.key === "Escape" && document.activeElement === q) {
        q.value = ""; state.q = ""; renderAll();
      }
    });
  }

  // ---- 初始化：?lang= > localStorage > 默认英文（不读 navigator），然后整页按该语言渲染 ----
  state.lang = detectLang(window.location.search || "", readStoredLang());
  bindSearch();
  applyLang(state.lang, false);
  // 支持直接链接到某条：#op:mul
  try {
    var hash = decodeURIComponent((window.location.hash || "").replace(/^#/, ""));
    if (hash && BY_ID[hash]) { select(hash, false); }
  } catch (e) {}
})();
""").replace("/*__I18N_CORE__*/", _I18N_CORE)


# ------------------------------------------------------------------ 渲染入口
def _json_for_script(payload):
    """内嵌到 <script type="application/json"> 的安全 JSON。

    `<` / `>` / `&` 转成 JSON 的 \\uXXXX 转义，因此词条正文里即便写了
    `</script>` 也无法提前闭合标签；JSON 解析后仍是原字符。
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return text


def _esc(value):
    """静态片段里只用到版本号；统一做 HTML 转义。"""
    return (str(value if value is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_html(payload):
    """把 `help_wiki.web_payload()` 渲染成单个自包含 HTML 字符串。

    界面文案来自 `_I18N`（随 payload 一起内嵌），语言在浏览器端决定/切换。
    """
    payload = dict(payload or {})
    payload.setdefault("i18n", _I18N)
    version = _esc(payload.get("wiki_version") or "")
    parts = [
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>TGAI-ORV Wiki — offline help</title>\n"
        "<style>\n", _CSS, "\n</style>\n</head>\n<body>\n",
        "<header>\n<div class=\"brand\">\n"
        "<h1 class=\"brand-title\">TGAI-ORV Wiki</h1>\n"
        "<span id=\"tagline\" class=\"sub\"></span>\n",
        "<span class=\"ver\">wiki digest ", version, "</span>\n",
        "<div id=\"lang\" class=\"langswitch\"></div>\n",
        "</div>\n<div class=\"searchrow\">\n",
        "<input id=\"q\" type=\"search\" autocomplete=\"off\" autofocus placeholder=\"\">\n",
        "<span id=\"status\"></span>\n</div>\n",
        "<div id=\"kinds\" class=\"kinds\"></div>\n</header>\n",
        "<div class=\"app\">\n<aside id=\"nav\"></aside>\n<main id=\"detail\"></main>\n</div>\n",
        "<footer id=\"foot\"></footer>\n",
        "<script type=\"application/json\" id=\"wiki-data\">",
        _json_for_script(payload),
        "</script>\n<script>\n", _JS, "\n</script>\n",
        "<a id=\"toTop\" href=\"#\"></a>\n",
        "</body>\n</html>\n",
    ]
    return "".join(parts)


def open_web(payload, launch=None):
    """渲染 → 写一个**固定**目录 → 返回路径；**默认不自动弹浏览器**。

    ★ 2026-11-21（用户踩到）：旧实现每次调用都 `mkdtemp` 建新目录 **并且自动
    `webbrowser.open()`**，于是反复调用会不断开新窗口、**把用户的前台窗口抢走**
    （实测在用户机器上留下了 188 个 `tgai-orv-help-*` 目录，还让正在用的
    嘉立创EDA看起来"被搞坏了"）。两处都改了：

      1. 输出到**固定路径** `$TMP/tgai-orv-help/index.html`（覆盖写，不再堆积）；
      2. **默认不弹浏览器**，只打印路径；要弹窗口必须显式 `launch=True`
         （或环境变量 `TG_HELP_LAUNCH=1`）。

    这样 `T.help(open='web')` 在脚本/CI/批量调用里不会产生任何窗口。
    """
    text = render_html(payload)
    directory = os.path.join(tempfile.gettempdir(), "tgai-orv-help")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "index.html")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    if launch is None:
        launch = os.environ.get("TG_HELP_LAUNCH", "").strip().lower() in (
            "1", "true", "yes", "on")
    if launch:
        try:
            webbrowser.open(pathlib.Path(path).as_uri())
        except Exception:
            # 无头环境 / CI / 没有默认浏览器：路径已经拿到，调用方会打印它。
            pass
    return os.path.abspath(path)
