# -*- coding: utf-8 -*-
"""L2 外部词条加载器 —— `T.help` 热更新 Wiki 的"数据层"。

协议（**一个字节都别改**，依据 `wiki/README_词条规范.txt`）：

    # 井号开头是注释，整行忽略
    [ID] op:mul                  ← 唯一必填字段（允许全角 ［ ］）
    [分类] compute               ← 可缺，决定网页侧边栏
    [中文名] 逐元素相乘
    [英文名] Element-wise multiply
    [作者] 张三                  ← 可选

    === 中文 ===
    正文……
    === English ===
    Body...
    === 示例 ===
    y = a.mul(b)
    === 见也 ===
    op:matmul
    op:add

加载顺序（后者覆盖前者，同 ID 覆盖记入 `shadowed_by`）：

    1. 包内 wiki/                （随 pip 安装）
    2. 仓库根 wiki/              （源码树；开发时最常用）
    3. TG_WIKI_PATH 指定的目录    （多个目录用 os.pathsep 分隔；也认 TG_WIKI_DIR）
    4. ~/.tgai/wiki/             （用户私有覆盖）

缓存键是每个文件的 `(绝对路径, mtime_ns, size)`：**同一进程内改文件，下一次
`T.help()` 立刻反映**（不做文件监听线程，避免跨平台 inotify 的坑）。
`T.help.refresh()` 强制清空缓存。

编码：UTF-8（带/不带 BOM）优先，失败回退 GB18030/GBK —— Windows 记事本默认存
ANSI，朋友们一定会踩。

设计纪律 / Design notes
-----------------------
* **本模块不含任何词条文案**：中文/英文正文全部来自 `wiki/*.txt`（L2）。这里只有
  协议解析、校验与渲染数据的组装。
* 解析失败**不炸掉整个帮助**：错误收集进 `registry["errors"]`，由 `T.help.check()`
  和 JSON `errors` 字段明确暴露；`load(strict=True)` 才抛 `WikiError`（给 CI 用）。
* 本模块**不在引擎启动路径上**：只有 `T.help(format="json")` / `open="web"` /
  `check` / `missing` / `contrib` / 关键词搜索时才惰性导入。
"""

import hashlib
import os
import re

from . import help_l0

__all__ = [
    "WikiError", "parse_text", "parse_meta_text", "search_dirs", "load",
    "load_strict", "refresh", "check", "missing", "contrib",
    "json_payload", "web_payload", "text_search",
]

KINDS = ("op", "api", "kernel", "cli", "env", "plugin")

#: 头部允许的键（固定；别的键一律报错，防拼写错静默失效）。
_HEADER_FIELDS = {
    "分类": "category",
    "中文名": "title_zh",
    "英文名": "title_en",
    "作者": "author",
}

#: 允许的区块名（大小写/前后空格容忍）。
_SECTION_FIELDS = {
    "中文": "zh",
    "示例": "example",
    "见也": "see",
}

_HDR_RE = re.compile(r"^[\[［]\s*([^\]］]+?)\s*[\]］]\s*(.*)$")
_SEC_RE = re.compile(r"^={3,}\s*(.*?)\s*={3,}$")
_SEE_RE = re.compile(r"([a-z]+:[^\s,;，；]+)")

_SCHEMA = "tgai-help/2"

_cache = None          # {"dirs": [...], "scan": [...], "registry": {...}}
_file_cache = {}       # path -> (mtime_ns, size, text, parsed-or-None, error-or-None)


# --------------------------------------------------------------------- 异常
class WikiError(Exception):
    """词条协议/校验错误。`errors` 是逐条的结构化明细（file/message/code）。"""

    def __init__(self, message, errors=None):
        self.errors = list(errors or ())
        if self.errors:
            lines = [message]
            for err in self.errors[:10]:
                lines.append("  - %s: %s" % (err.get("file", "?"), err.get("message", "")))
                for cand in err.get("candidates") or ():
                    lines.append("      候选 / did you mean: %s" % cand)
            if len(self.errors) > 10:
                lines.append("  ……共 %d 条（T.help.check() 看全部）" % len(self.errors))
            message = "\n".join(lines)
        super().__init__(message)


class _ParseError(Exception):
    def __init__(self, message, code="parse"):
        self.code = code
        super().__init__(message)


# --------------------------------------------------------------- 编码与读取
def _decode(data):
    """UTF-8（含 BOM）优先，失败回退 GB18030/GBK（Windows 记事本 ANSI）。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise _ParseError("无法解码：既不是 UTF-8（可带 BOM）也不是 GB18030/GBK。"
                      "修法：用 VSCode/记事本另存为 UTF-8 再保存", "encoding")


def _read_text(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise _ParseError("读不到文件：%s。修法：确认文件存在且当前用户有读权限" % exc, "io")
    return _decode(data)


def _strip_trailing_blanks(lines):
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


# ------------------------------------------------------------------ TXT 解析
def _header_field(raw):
    key = (raw or "").strip()
    if not key:
        return None
    if key.upper() == "ID":
        return "id"
    return _HEADER_FIELDS.get(key)


def _section_field(raw):
    name = (raw or "").strip()
    if name in _SECTION_FIELDS:
        return _SECTION_FIELDS[name]
    low = name.lower()
    if low in ("english", "en"):
        return "en"
    if low in ("zh", "cn"):
        return "zh"
    return None


def _parse_txt(text):
    """解析一份词条 TXT → {"header": {...}, "sections": {...}}；格式错误抛 _ParseError。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header = {}
    sections = {}
    mode = "header"
    current = None
    for lineno, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if stripped.startswith("#"):          # 注释整行忽略（任何位置）
            continue
        if not stripped:
            if mode == "body":
                sections[current].append("")
            continue
        if mode == "header":
            match = _HDR_RE.match(stripped)
            if match:
                field = _header_field(match.group(1))
                if field is None:
                    raise _ParseError(
                        "第 %d 行：未知头部字段 [%s]。允许的键只有 "
                        "ID / 分类 / 中文名 / 英文名 / 作者。修法：改用白名单里的键，"
                        "不要自造字段" % (lineno, match.group(1).strip()),
                        "unknown_header")
                if field in header:
                    raise _ParseError("第 %d 行：头部字段重复（%s）。修法：每个头部键只写一次"
                                      % (lineno, match.group(1).strip()), "duplicate_header")
                header[field] = match.group(2).strip()
                continue
            sec = _SEC_RE.match(stripped)
            if not sec:
                raise _ParseError(
                    "第 %d 行：头部出现无法解析的内容 %r。只允许 `[键] 值` 行或 "
                    "`=== 区块 ===`，且头部必须在最前面。修法：把这一行改成 "
                    "`[字段] 值` 或 `=== 中文 ===` 这样的区块标题" % (lineno, stripped),
                    "bad_header_line")
            _open_section(sec, sections, lineno)
            mode = "body"
            current = _last_section(sections)
            continue
        sec = _SEC_RE.match(stripped)
        if sec:
            _open_section(sec, sections, lineno)
            current = _last_section(sections)
            continue
        sections[current].append(raw.rstrip())

    for key in ("zh", "en", "example", "see"):
        if key in sections:
            sections[key] = _strip_trailing_blanks(list(sections[key]))
    entry = {"header": header, "sections": sections}
    _validate_entry(entry)
    return entry


def _open_section(match, sections, lineno):
    field = _section_field(match.group(1))
    if field is None:
        raise _ParseError(
            "第 %d 行：未知区块 `=== %s ===`。允许的区块只有 中文 / English / 示例 / 见也。"
            "修法：把区块名改成这四个之一" % (lineno, match.group(1).strip()), "unknown_section")
    if field in sections:
        raise _ParseError("第 %d 行：区块重复（%s）。修法：每个区块只出现一次"
                          % (lineno, match.group(1).strip()),
                          "duplicate_section")
    sections[field] = []
    return field


def _last_section(sections):
    return list(sections)[-1]


def _body(sections, key):
    return "\n".join(sections.get(key) or ()).strip()


def _validate_entry(entry):
    header = entry["header"]
    item_id = (header.get("id") or "").strip()
    if not item_id:
        raise _ParseError("缺少必填字段 [ID]（格式 Kind:名字，例如 [ID] op:mul）。"
                          "修法：在文件头部加一行 `[ID] 类型:名字`",
                          "missing_id")
    head, sep, rest = item_id.partition(":")
    if not sep or not rest or head not in KINDS:
        raise _ParseError(
            "[ID] 格式不对：%r。正确格式是 `类型:名字`，类型只能是 %s。"
            "修法：把 [ID] 改成 `op:mul` 这种形式" % (item_id, " / ".join(KINDS)), "bad_id")
    if head == "plugin" and rest.count(":") < 1:
        raise _ParseError("[ID] 插件条目必须是 `plugin:包名:条目名`，收到 %r。"
                          "修法：改成 `plugin:<你的包名>:<条目名>`" % item_id,
                          "bad_id")
    entry["id"] = item_id
    entry["kind"] = head
    entry["name"] = rest.split(":")[-1]
    for field in ("category", "title_zh", "title_en", "author"):
        entry[field] = (header.get(field) or "").strip()
    for lang in ("zh", "en"):
        entry[lang] = _body(entry["sections"], lang)
    entry["example"] = _body(entry["sections"], "example")
    see = []
    for line in entry["sections"].get("see") or ():
        text = line.strip()
        if not text:
            continue
        found = _SEE_RE.findall(text)
        if not found:
            raise _ParseError("`=== 见也 ===` 里有一行无法解析：%r。一行写一个 ID（例如 op:matmul）。"
                              "修法：删掉普通文字，只保留 `类型:名字` 形式的引用" % text,
                              "bad_see")
        see.extend(found)
    entry["see_also"] = see
    # 记录"缺边"（页面显示 暂无xx，不阻断加载）
    entry["missing_langs"] = [lang for lang in ("zh", "en") if not entry[lang]]
    return entry


def parse_text(text, source="<text>"):
    """公开的单文件解析入口（只做协议校验，不做 L0 交叉校验）。"""
    try:
        return _parse_txt(text)
    except _ParseError as exc:
        err = {"file": source, "message": str(exc), "code": exc.code}
        raise WikiError(str(exc) + "。修法：按上面的提示改正该词条文件", [err]) from None


# -------------------------------------------------------------- _meta.txt
def _parse_meta(text, source="<meta>"):
    categories = []
    seen = {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for lineno, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _HDR_RE.match(stripped)
        if not match or _header_field(match.group(1)) != "category":
            raise _ParseError(
                "%s 第 %d 行：只允许 `[分类] id | 中文 | English | order`。"
                "修法：按要求补全 4 个字段，或整行用 # 注释掉" % (source, lineno),
                "meta_line")
        parts = [p.strip() for p in match.group(2).split("|")]
        if len(parts) != 4:
            raise _ParseError(
                "%s 第 %d 行：需要 4 个字段（id | 中文 | English | order），实际 %d 个。"
                "修法：补成 `[分类] tensor | 张量 | Tensor | 20` 这种格式"
                % (source, lineno, len(parts)), "meta_fields")
        cid, zh, en, order = parts
        if not cid:
            raise _ParseError("%s 第 %d 行：分类 id 为空。修法：在 `|` 前写英文分类 id"
                              % (source, lineno), "meta_fields")
        if cid in seen:
            raise _ParseError(
                "%s 重复分类 id：%r（第 %d 行与第 %d 行）。修法：删掉重复的一行"
                % (source, cid, seen[cid], lineno),
                "meta_duplicate")
        try:
            order_i = int(order)
        except ValueError:
            raise _ParseError("%s 第 %d 行：order 必须是整数，收到 %r。"
                              "修法：把最后一个字段改成数字（如 20）"
                              % (source, lineno, order), "meta_order")
        seen[cid] = lineno
        categories.append({"id": cid, "zh": zh or cid, "en": en or cid, "order": order_i})
    if "other" not in seen:
        categories.append({"id": "other", "zh": "其他", "en": "Other", "order": 999})
    categories.sort(key=lambda c: (c["order"], c["id"]))
    return categories


def parse_meta_text(text, source="<meta>"):
    """公开的 `_meta.txt` 解析入口。"""
    try:
        return _parse_meta(text, source)
    except _ParseError as exc:
        raise WikiError(str(exc) + "。修法：按上面的提示改正 `_meta.txt`",
                        [{"file": source, "message": str(exc),
                          "code": exc.code}]) from None


# ------------------------------------------------------------------ 目录发现
def search_dirs():
    """按优先级（低 → 高）返回 wiki 目录列表：包内/仓库根 → TG_WIKI_PATH → ~/.tgai/wiki。"""
    dirs = []
    seen = set()

    def add(path):
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(path) and path not in seen:
            dirs.append(path)
            seen.add(path)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # 1) 包内 wiki/（wheel 里）
    for cand in (os.path.join(pkg_dir, "wiki"),
                 os.path.join(os.path.dirname(pkg_dir), "wiki")):
        add(cand)
    # 2) 源码仓库根 wiki/（开发时最常用：向上找带 wiki/ 的 pyproject 目录）
    #    注意：仓库里的 python/ 子目录也可能有自己的 pyproject.toml，所以必须
    #    「pyproject 与 wiki/ 同时存在」才算命中，否则继续往上找。
    cur = pkg_dir
    for _ in range(8):
        nxt = os.path.dirname(cur)
        if not nxt or nxt == cur:
            break
        cur = nxt
        wiki_dir = os.path.join(cur, "wiki")
        if (os.path.isdir(wiki_dir)
                and (os.path.isfile(os.path.join(cur, "pyproject.toml"))
                     or os.path.isfile(os.path.join(wiki_dir, "_meta.txt")))):
            add(wiki_dir)
            break
    # 3) TG_WIKI_PATH / TG_WIKI_DIR（可多个）
    raw = os.environ.get("TG_WIKI_PATH") or os.environ.get("TG_WIKI_DIR") or ""
    for part in raw.split(os.pathsep):
        add(part.strip())
    # 4) 用户私有覆盖
    add(os.path.join(os.path.expanduser("~"), ".tgai", "wiki"))
    return dirs


# --------------------------------------------------------------- 扫描与缓存
def _scan(dirs):
    """返回 [(path, mtime_ns, size), ...]（含 _meta.txt 与词条 txt，排序稳定）。"""
    rows = []
    for directory in dirs:
        for root, dirnames, filenames in os.walk(directory):
            dirnames.sort()
            for filename in sorted(filenames):
                if not filename.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, filename)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                rows.append((path, int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                             int(st.st_size)))
    rows.sort()
    return rows


def _is_entry_file(path):
    name = os.path.basename(path)
    if name.lower() == "_meta.txt":
        return False
    if name.upper().startswith("README"):
        return False
    return True


def _is_meta_file(path):
    return os.path.basename(path).lower() == "_meta.txt"


def _mtime_key(path):
    try:
        st = os.stat(path)
        return (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))
    except OSError:
        return (0, 0)


def _read_cached(path):
    """读词条文件：缓存 (text, parsed, error)，按 (mtime_ns, size) 失效。"""
    key = _mtime_key(path)
    cached = _file_cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1:]
    try:
        text = _read_text(path)
        parsed = _parse_txt(text)
        payload = (text, parsed, None)
    except _ParseError as exc:
        payload = (None, None, exc)
    except WikiError as exc:                      # 理论上不会到
        payload = (None, None, _ParseError(str(exc)))
    _file_cache[path] = (key, payload[0], payload[1], payload[2])
    return payload


def _read_meta_cached(path):
    """读 `_meta.txt`：**不能**按词条解析（它没有 [ID]，字段是 [分类] 行）。"""
    key = _mtime_key(path)
    cached = _file_cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    try:
        text = _read_text(path)
        payload = (text, None)
    except _ParseError as exc:
        payload = (None, exc)
    _file_cache[path] = (key, payload[0], payload[1])
    return payload


# ------------------------------------------------------------------ 加载主流程
def _unknown_candidates(item_id):
    return help_l0.nearest(item_id, n=3)


def _entry_filename_expected(item_id):
    return item_id.replace(":", ".")


def _build(dirs, scan):
    l0_catalog = help_l0.catalog()
    l0_ids = set(item["id"] for item in l0_catalog)

    errors = []
    warnings = []
    meta_errors = []
    unknown_ids = []
    duplicate_ids = []
    dead_links = []
    unmapped = {}
    shadowed = {}

    # ---- 1. 逐目录解析（低 → 高） ----
    layer_entries = {}        # layer_index -> {id: entry}
    meta_categories = None
    meta_source = ""
    parsed_files = 0
    texts_for_hash = []

    for layer_index, directory in enumerate(dirs):
        layer = {}
        for path, _mtime, _size in scan:
            if not path.startswith(directory + os.sep) and os.path.dirname(path) != directory:
                continue
            rel = os.path.relpath(path, directory)
            if _is_meta_file(path):
                raw_text, err = _read_meta_cached(path)
                if err is not None:
                    meta_errors.append({"file": path, "message": str(err),
                                        "code": err.code})
                    continue
                try:
                    meta_categories = _parse_meta(raw_text, path)
                    meta_source = path
                    texts_for_hash.append((layer_index, rel, raw_text))
                except _ParseError as exc:
                    meta_errors.append({"file": path, "message": str(exc),
                                        "code": exc.code})
                continue
            if not _is_entry_file(path):
                continue
            raw_text, parsed, err = _read_cached(path)
            if err is not None:
                candidates = _unknown_candidates(
                    _entry_filename_expected(os.path.splitext(rel)[0]))
                errors.append({"file": path, "message": str(err), "code": err.code,
                               "candidates": candidates})
                continue
            parsed_files += 1
            texts_for_hash.append((layer_index, rel, raw_text))
            entry = dict(parsed)
            entry["source"] = path
            entry["layer_index"] = layer_index
            # 规则 5：[ID] 与文件名不一致 ⇒ 警告但加载（[ID] 为准）
            stem = os.path.splitext(os.path.basename(path))[0]
            expected = _entry_filename_expected(entry["id"])
            if stem != expected:
                warnings.append({
                    "file": path, "code": "filename_mismatch",
                    "message": "文件名 %r 与 [ID] %r 不一致（已按 [ID] 加载；建议改名 %s.txt）"
                               % (stem, entry["id"], expected),
                })
            if entry["id"] in layer:
                duplicate_ids.append({
                    "id": entry["id"], "file": path,
                    "other_file": layer[entry["id"]]["source"],
                    "message": "同一目录下重复 [ID]：%s" % entry["id"],
                })
                continue
            layer[entry["id"]] = entry
        layer_entries[layer_index] = layer

    if meta_categories is None:
        meta_categories = [{"id": "other", "zh": "其他", "en": "Other", "order": 999}]
    meta_ids = set(c["id"] for c in meta_categories)

    # ---- 2. 跨目录合并（后者覆盖前者） ----
    merged = {}
    for layer_index in sorted(layer_entries):
        for item_id, entry in layer_entries[layer_index].items():
            if item_id in merged:
                prev = merged[item_id]
                entry["shadowed_by"] = list(prev.get("shadowed_by") or ()) + [prev["source"]]
                shadowed[item_id] = entry["shadowed_by"]
            merged[item_id] = entry

    # ---- 3. 交叉校验（未知 ID / 死链 / 分类） ----
    all_wiki_ids = set(merged)
    for item_id in sorted(merged):
        entry = merged[item_id]
        if entry["kind"] != "plugin" and item_id not in l0_ids:
            unknown_ids.append({
                "id": item_id, "file": entry["source"],
                "candidates": _unknown_candidates(item_id),
                "message": "未知 ID：%s（引擎里没有这个条目；最接近的候选见 candidates）" % item_id,
            })
        category = (entry.get("category") or "").strip()
        if category and category not in meta_ids:
            unmapped.setdefault(category, []).append(item_id)
            entry["category"] = "other"
            entry["category_original"] = category
        elif not category:
            entry["category"] = "other"

    for item_id in sorted(merged):
        entry = merged[item_id]
        for target in entry.get("see_also") or ():
            if target not in l0_ids and target not in all_wiki_ids:
                dead_links.append({
                    "id": item_id, "file": entry["source"], "target": target,
                    "candidates": _unknown_candidates(target),
                    "message": "%s 的 `见也` 引用了不存在的 ID：%s" % (item_id, target),
                })

    invalid_ids = set(row["id"] for row in unknown_ids)

    # ---- 4. 组装条目（L0 + L2，缺词条也保留） ----
    items = []
    for l0_item in l0_catalog:
        wiki = merged.get(l0_item["id"])
        item = dict(l0_item)
        item.setdefault("subkind", "")
        if wiki is not None:
            item.update({
                "category": wiki.get("category") or "other",
                "documented": bool(wiki.get("zh") or wiki.get("en")),
                "title_zh": wiki.get("title_zh") or "",
                "title_en": wiki.get("title_en") or "",
                "zh": wiki.get("zh") or "",
                "en": wiki.get("en") or "",
                "wiki_example": wiki.get("example") or "",
                "see_also": list(wiki.get("see_also") or ()),
                "author": wiki.get("author") or "",
                "source": wiki.get("source") or "",
                "missing_langs": list(wiki.get("missing_langs") or ()),
                "category_original": wiki.get("category_original") or "",
                "shadowed_by": list(wiki.get("shadowed_by") or ()),
            })
        else:
            item.update({
                "category": "other",
                "documented": False,
                "title_zh": "", "title_en": "", "zh": "", "en": "",
                "wiki_example": "", "see_also": [], "author": "",
                "source": "", "missing_langs": ["zh", "en"],
                "category_original": "", "shadowed_by": [],
            })
        if item["id"] not in invalid_ids:
            items.append(item)

    # ---- 5. 插件条目（P2 自动发现；P1 只认插件自带 wiki 的声明） ----
    for item_id in sorted(merged):
        entry = merged[item_id]
        if entry["kind"] != "plugin" or item_id in invalid_ids:
            continue
        items.append({
            "id": item_id,
            "kind": "plugin",
            "name": entry["name"],
            "signature": "",
            "example": "T.plugin ...  # 插件条目（P2 自动发现）",
            "module": "plugin",
            "subkind": "",
            "category": entry.get("category") or "plugin",
            "documented": bool(entry.get("zh") or entry.get("en")),
            "title_zh": entry.get("title_zh") or "",
            "title_en": entry.get("title_en") or "",
            "zh": entry.get("zh") or "",
            "en": entry.get("en") or "",
            "wiki_example": entry.get("example") or "",
            "see_also": list(entry.get("see_also") or ()),
            "author": entry.get("author") or "",
            "source": entry.get("source") or "",
            "missing_langs": list(entry.get("missing_langs") or ()),
            "category_original": entry.get("category_original") or "",
            "shadowed_by": list(entry.get("shadowed_by") or ()),
            # 插件条目只能由插件自证；L0 不 import 插件，所以按"插件声明"计可达。
            "form": "",
            "form_kind": "plugin",
            "reachable": True,
            "reach_evidence": "插件自带词条声明（P1 不内省插件运行时）",
            "unreachable": "",
        })

    # ---- 6. 覆盖率 / 版本 / 缺边 ----
    by_id = {item["id"]: item for item in items}
    documented_ids = [i for i, it in by_id.items()
                      if it["kind"] != "plugin" and it["documented"]]
    l0_total = len([it for it in l0_catalog])
    percent = round(100.0 * len(documented_ids) / l0_total, 1) if l0_total else 0.0

    missing_zh = [it["id"] for it in items
                  if it["id"] in merged and not it["zh"]]
    missing_en = [it["id"] for it in items
                  if it["id"] in merged and not it["en"]]

    hasher = hashlib.sha256()
    for layer_index, rel, text in sorted(texts_for_hash):
        hasher.update(("%d|%s|" % (layer_index, rel.replace("\\", "/"))).encode("utf-8"))
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    wiki_version = hasher.hexdigest()[:8]

    registry = {
        "schema": _SCHEMA,
        "dirs": list(dirs),
        "categories": meta_categories,
        "meta_source": meta_source,
        "items": items,
        "by_id": by_id,
        "l0_total": l0_total,
        "parsed_files": parsed_files,
        "wiki_version": wiki_version,
        "coverage": {"documented": len(documented_ids), "total": l0_total,
                     "percent": percent},
        "errors": errors,
        "warnings": warnings,
        "meta_errors": meta_errors,
        "unknown_ids": unknown_ids,
        "duplicate_ids": duplicate_ids,
        "dead_links": dead_links,
        "unmapped_categories": [
            {"category": cat, "ids": sorted(ids)} for cat, ids in sorted(unmapped.items())
        ],
        "missing_zh": missing_zh,
        "missing_en": missing_en,
        "shadowed": shadowed,
    }
    return registry


def _load(force=False):
    global _cache
    dirs = search_dirs()
    scan = _scan(dirs)
    if (not force and _cache is not None
            and _cache["dirs"] == dirs and _cache["scan"] == scan):
        return _cache["registry"]
    registry = _build(dirs, scan)
    _cache = {"dirs": dirs, "scan": scan, "registry": registry}
    return registry


def load(strict=False):
    """加载全部词条，返回 registry（含 errors/warnings/coverage/wiki_version）。

    `strict=True` 时若有解析失败/未知 ID/重复 ID/死链，抛 `WikiError`（CI 用）。
    """
    registry = _load()
    if strict:
        blocking = []
        for key in ("errors", "meta_errors", "unknown_ids", "duplicate_ids", "dead_links"):
            for row in registry.get(key) or ():
                err = dict(row)
                err.setdefault("file", "")
                err.setdefault("message", str(row.get("id") or row.get("target") or ""))
                blocking.append(err)
        if blocking:
            raise WikiError("Wiki 词条有 %d 个错误（修完再提交）。"
                            "修法：按下面每个文件的提示改，或跑 T.help.check() 看明细"
                            % len(blocking), blocking)
    return registry


def load_strict():
    """`load(strict=True)` 的便捷别名（CI / 测试用）。"""
    return load(strict=True)


def refresh():
    """强制清空 L0/L2 缓存；返回刷新后的目录列表。"""
    global _cache
    _cache = None
    _file_cache.clear()
    help_l0.invalidate()
    return {"refreshed": True, "dirs": search_dirs()}


# ------------------------------------------------------------------ 查询
_LEGACY_KIND = {
    "kernels": "kernel", "kernel": "kernel", "内核": "kernel", "内核表": "kernel",
    "op": "op", "ops": "op", "operator": "op", "operators": "op",
    "算子": "op", "张量算子": "op",
    "api": "api", "apis": "api", "接口": "api", "运行时": "api", "backend": "api",
    "env": "env", "环境变量": "env",
    "cli": "cli", "命令行": "cli", "参数": "cli",
    "plugin": "plugin", "插件": "plugin",
}

#: kind 的界面标签：(中文, English)。网页按**当前界面语言只显示其一**，不再并排。
_KIND_LABEL = {
    "op": ("张量算子", "tensor ops"),
    "api": ("顶层 API", "top-level API"),
    "kernel": ("CUDA 内核", "kernels"),
    "cli": ("命令行参数", "CLI flags"),
    "env": ("环境变量", "env vars"),
    "plugin": ("插件条目", "plugin entries"),
    "": ("全部", "all"),
}


def _resolve_kind(kind):
    """把各类别名解析成 (kind, subkind, note)。

    - 新协议 kind（op/api/kernel/cli/env/plugin）与中英文别名直接映射；
    - 旧 `T.help(tag="opt")` → 顶层 api 里的优化器类（subkind=optimizer）；
    - 旧 `tag="mods"`（子模块）在新 Wiki 里没有对应 kind → note 提示。
    """
    if not kind:
        return None, None, ""
    raw = str(kind).strip()
    low = raw.lower()
    if low in ("opt", "optim", "optimizer", "optimizers", "优化器"):
        return "api", "optimizer", ""
    if low in ("mods", "mod", "module", "modules", "模块", "子模块"):
        return None, None, "mods"
    if low in _LEGACY_KIND:
        return _LEGACY_KIND[low], None, ""
    if raw in _LEGACY_KIND:
        return _LEGACY_KIND[raw], None, ""
    return "?", None, "unknown"


def _matches(item, needle):
    blob = " ".join(str(item.get(key) or "") for key in (
        "id", "kind", "name", "title_zh", "title_en", "zh", "en",
        "signature", "example", "category", "module", "wiki_example",
    ))
    blob += " " + " ".join(item.get("see_also") or ())
    return needle in blob.lower()


def _query_items(registry, kind=None, s=None):
    new_kind, subkind, note = _resolve_kind(kind)
    items = list(registry["items"])
    if new_kind == "?":
        # 旧 API 里不认识的 tag：按关键词搜（不要静默空输出）
        if not s and kind:
            s = str(kind)
    elif new_kind:
        if subkind:
            items = [it for it in items if it["kind"] == new_kind
                     and it.get("subkind") == subkind]
        else:
            items = [it for it in items if it["kind"] == new_kind]
    if s:
        needle = str(s).strip().lower()
        items = [it for it in items if needle and _matches(it, needle)]
    kind_order = {k: i for i, k in enumerate(KINDS)}
    items.sort(key=lambda it: (kind_order.get(it["kind"], 99),
                               0 if it["documented"] else 1,
                               it["name"]))
    return items, new_kind, note


def _public_item(item):
    return {
        "id": item["id"],
        "kind": item["kind"],
        "name": item["name"],
        "title_zh": item.get("title_zh") or "",
        "title_en": item.get("title_en") or "",
        "zh": item.get("zh") or "",
        "en": item.get("en") or "",
        "usage": item.get("signature") or "",
        "namespace": item.get("example") or "",
        "category": item.get("category") or "other",
        "documented": bool(item.get("documented")),
        "module": item.get("module") or "",
        "wiki_example": item.get("wiki_example") or "",
        "see_also": list(item.get("see_also") or ()),
        "author": item.get("author") or "",
        # ★ 可达性自证（L0 现算，见 help_l0 顶部说明）：别把不可调用的东西写成能调用的。
        "form": item.get("form") or "",
        "form_kind": item.get("form_kind") or "",
        "reachable": bool(item.get("reachable", True)),
        "reach_evidence": item.get("reach_evidence") or "",
        "unreachable": item.get("unreachable") or "",
    }


def _problems(registry):
    rows = []
    for key in ("errors", "meta_errors"):
        for row in registry.get(key) or ():
            rows.append({"file": row.get("file", ""), "code": row.get("code", ""),
                         "message": row.get("message", ""),
                         "candidates": list(row.get("candidates") or ())})
    for row in registry.get("unknown_ids") or ():
        rows.append({"file": row.get("file", ""), "code": "unknown_id",
                     "message": row.get("message", ""),
                     "candidates": list(row.get("candidates") or ())})
    for row in registry.get("duplicate_ids") or ():
        rows.append({"file": row.get("file", ""), "code": "duplicate_id",
                     "message": row.get("message", ""), "candidates": []})
    for row in registry.get("dead_links") or ():
        rows.append({"file": row.get("file", ""), "code": "dead_link",
                     "message": row.get("message", ""),
                     "candidates": list(row.get("candidates") or ())})
    return rows


def _no_match_hint(kind, query, registry):
    parts = []
    if query:
        parts.append("没有命中 %r / no match." % (query,))
    else:
        parts.append("没有命中 / no match.")
    parts.append("可执行 / try: T.help()（总览）; T.help(format=\"json\")（全量元数据）; "
                 "T.help.missing()（待写清单）; T.help.check()（词条健康度）.")
    if _resolve_kind(kind)[0] == "?":
        parts.append("kind 只支持 op/api/kernel/cli/env/plugin；旧 tag 可用 kernels/api/op/opt/mods.")
    if _problems(registry):
        parts.append("另有 %d 个词条文件问题，用 T.help.check() 查看." % len(_problems(registry)))
    return " ".join(parts)


def json_payload(kind=None, s=None):
    """`format="json"` 的 payload（纯 JSON 的 schema 契约见 docs/AI_HELP_PROTOCOL.md）。"""
    registry = load()
    items, new_kind, note = _query_items(registry, kind, s)
    payload = {
        "schema": _SCHEMA,
        "query": str(s) if s else "",
        "kind": new_kind or "",
        "count": len(items),
        "items": [_public_item(it) for it in items],
        "hint": "" if items else _no_match_hint(kind, s, registry),
        "wiki_version": registry["wiki_version"],
        "coverage": dict(registry["coverage"]),
        "errors": _problems(registry),
        "warnings": [{"file": w["file"], "message": w["message"]}
                     for w in registry.get("warnings") or ()],
    }
    if note == "mods":
        payload["note"] = "mods（子模块）不是 Wiki 的 kind；已返回全部条目"
    elif note == "unknown":
        payload["note"] = "kind 无法识别，已按关键词搜索：%r" % (str(kind),)
    return payload


def web_payload(kind=None, s=None):
    """`open="web"` 的渲染数据（分类顺序、kind 计数、覆盖率、版本都在里面）。"""
    registry = load()
    items, new_kind, note = _query_items(registry, kind, s)
    categories = []
    for cat in registry["categories"]:
        count = len([it for it in items if (it.get("category") or "other") == cat["id"]])
        categories.append({"id": cat["id"], "zh": cat["zh"], "en": cat["en"],
                           "order": cat["order"], "count": count})
    kind_counts = []
    for k in KINDS:
        count = len([it for it in items if it["kind"] == k])
        if count:
            label = _KIND_LABEL.get(k) or (k, k)
            kind_counts.append({"id": k, "count": count,
                                "label_zh": label[0], "label_en": label[1]})
    payload = {
        "schema": _SCHEMA,
        "query": str(s) if s else "",
        "kind": new_kind or "",
        "count": len(items),
        "items": [_public_item(it) for it in items],
        "hint": "" if items else _no_match_hint(kind, s, registry),
        "categories": categories,
        "kinds": kind_counts,
        "wiki_version": registry["wiki_version"],
        "coverage": dict(registry["coverage"]),
        "l0_total": registry["l0_total"],
        "errors": _problems(registry),
        "warnings_count": len(registry.get("warnings") or ()),
        "shadowed_count": len(registry.get("shadowed") or {}),
    }
    if note:
        payload["note"] = note
    return payload


def text_search(s, kind=None):
    """给终端文本搜索追加的 Wiki 片段（空串表示没有命中）。

    只列**已编写**的外部词条，这样 `T.help('mul')` 里能直接看到 wiki 文案，
    改 `wiki/*.txt` 的一个字在同一个进程内立刻可见。
    """
    try:
        registry = load()
        items, _new_kind, _note = _query_items(registry, kind, s)
    except Exception:
        return ""                     # 帮助系统永远不能因为 wiki 坏掉而报错
    items = [it for it in items if it["documented"] and it["kind"] != "plugin"][:8]
    if not items:
        return ""
    lines = ["[wiki] 外部词条 / external wiki entries — %d 条（wiki/*.txt，改文件即时生效）"
             % len(items)]
    for idx, item in enumerate(items, 1):
        title = item.get("title_zh") or item.get("title_en") or ""
        head = "  %d. %s" % (idx, item["id"])
        if title:
            head += " — " + title
        head += "  [%s]" % (item.get("category") or "other")
        lines.append(head)
        if item.get("example"):
            lines.append("     自动示例: %s" % item["example"])
        body = (item.get("zh") or item.get("en") or "").strip()
        if body:
            first = " ".join(body.splitlines()[0].split())
            if len(first) > 120:
                first = first[:119] + "…"
            lines.append("     正文: %s" % first)
        if item.get("missing_langs"):
            lines.append("     暂无: %s（T.help.contrib(%r) 可补）"
                         % (" / ".join(item["missing_langs"]), item["id"]))
    return "\n".join(lines)


# ------------------------------------------------------- missing / check / contrib
def _print(text):
    print(text)


def missing():
    """待编写清单（按 kind/category 分组；给朋友们的"工作看板"）。"""
    registry = load()
    rows = [it for it in registry["items"]
            if it["kind"] != "plugin" and not it["documented"]]
    rows.sort(key=lambda it: (KINDS.index(it["kind"]),
                              it.get("category") or "other", it["name"]))
    by_kind = {}
    by_category = {}
    for it in rows:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
        cat = it.get("category") or "other"
        by_category[cat] = by_category.get(cat, 0) + 1
    out = {
        "total": len(rows),
        "documented": registry["coverage"]["documented"],
        "l0_total": registry["l0_total"],
        "percent": registry["coverage"]["percent"],
        "by_kind": by_kind,
        "by_category": by_category,
        "items": [{"id": it["id"], "kind": it["kind"], "name": it["name"],
                   "category": it.get("category") or "other",
                   "signature": it.get("signature") or "",
                   "example": it.get("example") or "",
                   "module": it.get("module") or ""} for it in rows],
    }
    _print("待编写词条 / missing entries：%d 条未写 / %d 条引擎条目（覆盖率 %.1f%%）"
           % (len(rows), registry["l0_total"], registry["coverage"]["percent"]))
    if by_kind:
        _print("  按 kind: " + "，".join("%s %d" % (k, by_kind[k])
                                        for k in KINDS if k in by_kind))
    if by_category:
        _print("  按分类: " + "，".join("%s %d" % (k, by_category[k])
                                        for k in sorted(by_category)))
    for it in rows[:20]:
        _print("  %-28s 示例: %-34s 分类: %s" % (it["id"], it["example"] or "-",
                                                it.get("category") or "other"))
    if len(rows) > 20:
        _print("  ……还有 %d 条（返回值的 items 里有全部）" % (len(rows) - 20))
    _print("补词条：T.help.contrib('op:mul') 生成骨架；把 .txt 放进 wiki/ 即热生效。")
    return out


def check():
    """词条健康度（CI 用）：解析失败、未知 ID、中英缺边、死链、未映射分类、覆盖率。"""
    registry = load()
    missing_translations = []
    for item in registry["items"]:
        if item["id"] not in registry["by_id"]:
            continue
        if item["kind"] == "plugin" and not item["documented"]:
            continue
        if item.get("source") == "":
            continue
        for lang in item.get("missing_langs") or ():
            missing_translations.append({"id": item["id"], "lang": lang})
    blocking = (len(registry["errors"]) + len(registry["meta_errors"])
                + len(registry["unknown_ids"]) + len(registry["duplicate_ids"])
                + len(registry["dead_links"]))
    result = {
        "ok": blocking == 0,
        "wiki_version": registry["wiki_version"],
        "coverage": dict(registry["coverage"]),
        "counts": {
            "l0_total": registry["l0_total"],
            "wiki_entries": len([it for it in registry["items"]
                                 if it.get("source")]),
            "parsed_files": registry["parsed_files"],
            "documented": registry["coverage"]["documented"],
            "missing": registry["l0_total"] - registry["coverage"]["documented"],
            "errors": len(registry["errors"]) + len(registry["meta_errors"]),
            "unknown_ids": len(registry["unknown_ids"]),
            "duplicate_ids": len(registry["duplicate_ids"]),
            "dead_links": len(registry["dead_links"]),
            "shadowed": len(registry["shadowed"]),
            "missing_zh": len(registry["missing_zh"]),
            "missing_en": len(registry["missing_en"]),
        },
        "dirs": registry["dirs"],
        "errors": _problems(registry),
        "warnings": [{"file": w["file"], "code": w.get("code", ""),
                      "message": w["message"]}
                     for w in registry.get("warnings") or ()],
        "unknown_ids": registry["unknown_ids"],
        "duplicate_ids": registry["duplicate_ids"],
        "dead_links": registry["dead_links"],
        "shadowed_by": registry["shadowed"],
        "unmapped_categories": registry["unmapped_categories"],
        "missing_translations": missing_translations,
    }
    _print("T.help.check() — Wiki 词条健康度 / wiki health")
    _print("  wiki 版本: %s   目录: %d 个" % (registry["wiki_version"],
                                             len(registry["dirs"])))
    _print("  覆盖率: %.1f%%（已写 %d / 引擎条目 %d，缺 %d）"
           % (registry["coverage"]["percent"], registry["coverage"]["documented"],
              registry["l0_total"],
              registry["l0_total"] - registry["coverage"]["documented"]))
    _print("  解析: %d 个词条文件；错误 %d；未知 ID %d；重复 ID %d；死链 %d；路径遮蔽 %d"
           % (registry["parsed_files"], result["counts"]["errors"],
              result["counts"]["unknown_ids"], result["counts"]["duplicate_ids"],
              result["counts"]["dead_links"], result["counts"]["shadowed"]))
    if result["counts"]["missing_zh"] or result["counts"]["missing_en"]:
        _print("  中英缺边: 缺中文 %d 条，缺英文 %d 条（不算错，页面显示「暂无」）"
               % (result["counts"]["missing_zh"], result["counts"]["missing_en"]))
    if registry["unmapped_categories"]:
        _print("  分类不在 _meta.txt: " + "，".join(
            "%s(%d)" % (row["category"], len(row["ids"]))
            for row in registry["unmapped_categories"]) + "（已归入 other）")
    for label, key in (("解析/结构错误", "errors"), ("未知 ID", "unknown_ids"),
                       ("重复 ID", "duplicate_ids"), ("死链", "dead_links")):
        rows = result[key]
        for row in rows[:10]:
            line = "  [%s] %s: %s" % (label, row.get("file", "?"),
                                      row.get("message", ""))
            if row.get("candidates"):
                line += "  候选: " + ", ".join(row["candidates"])
            _print(line)
        if len(rows) > 10:
            _print("  [%s] ……还有 %d 条" % (label, len(rows) - 10))
    for row in registry["warnings"][:10]:
        _print("  [警告] %s: %s" % (row["file"], row["message"]))
    if len(registry["warnings"]) > 10:
        _print("  [警告] ……还有 %d 条" % (len(registry["warnings"]) - 10))
    _print("  ok = %s（解析/校验无阻断性错误）" % result["ok"])
    return result


def _sanitize_filename(item_id):
    name = item_id.replace(":", ".")
    for ch in '<>:"/\\|?*\n\r\t':
        name = name.replace(ch, "_")
    return name + ".txt"


def _contrib_target_dir():
    raw = os.environ.get("TG_WIKI_PATH") or os.environ.get("TG_WIKI_DIR") or ""
    if raw:
        for part in raw.split(os.pathsep):
            part = part.strip()
            if part:
                return os.path.abspath(os.path.expanduser(part))
    return os.path.join(os.path.expanduser("~"), ".tgai", "wiki")


def contrib(item_id=None):
    """有词条⇒打印路径；没有⇒生成 TXT 骨架（front matter/分类/中英区块/示例都填好）。"""
    if item_id is None or not str(item_id).strip():
        raise ValueError("T.help.contrib(id) 需要条目 ID，例如 T.help.contrib('op:mul')。"
                         "修法：先跑 T.help.missing() 看可写清单，把列表里的 id 传进来")
    wiki_id = str(item_id).strip()
    l0_item = help_l0.entry(wiki_id)
    registry = load()
    existing = registry["by_id"].get(wiki_id)
    if existing is not None and existing.get("source"):
        paths = [existing["source"]]
        for path in existing.get("shadowed_by") or ():
            if path not in paths:
                paths.append(path)
        _print("该条目已有词条 / already documented:")
        for path in paths:
            _print("  " + path)
        _print("修改后保存即可热生效（同一进程内无需重启）。")
        return {"id": wiki_id, "existing": paths, "created": []}
    if l0_item is None:
        cands = help_l0.nearest(wiki_id)
        raise ValueError("没有这个条目：%r。最接近的候选：%s。"
                         "修法：从候选里选一个，或先跑 T.help.missing() 看全部清单"
                         % (wiki_id, ", ".join(cands) if cands else "（无）"))
    kind = l0_item["kind"]
    category = help_l0.default_category(kind)
    signature = l0_item.get("signature") or ""
    example = l0_item.get("example") or ""
    target_dir = _contrib_target_dir()
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, _sanitize_filename(wiki_id))
    if os.path.exists(path):
        _print("骨架文件已存在 / skeleton already exists: %s" % path)
        return {"id": wiki_id, "existing": [path], "created": []}
    text = (
        "# 由 T.help.contrib(%r) 生成。填文字即可；[ID] 与区块名请别改。\n"
        "[ID] %s\n"
        "[分类] %s\n"
        "[中文名] 【待填：中文短名】\n"
        "[英文名] 【待填：English short name】\n"
        "[作者] \n"
        "\n"
        "=== 中文 ===\n"
        "【待填：中文说明。可以多段，`反引号` 是代码样式，**加粗** 也可以。】\n"
        "\n"
        "=== English ===\n"
        "【待填：English description. Multiple paragraphs are fine.】\n"
        "\n"
        "=== 示例 ===\n"
        "# 引擎按真实签名自动生成的调用示例（这里只是补充，不必重复）：\n"
        "%s\n"
        "# 签名: %s\n"
        "\n"
        "=== 见也 ===\n"
        "# 一行一个 ID，例如 op:matmul；引用了不存在的 ID 会在 T.help.check() 里报死链\n"
        % (wiki_id, wiki_id, category, example, signature)
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    refresh()
    _print("已生成骨架 / skeleton created:")
    _print("  " + path)
    _print("下一步 / next:")
    _print("  1. 打开上面的文件，填 [中文名] / [英文名] 与 中文 / English 区块（只写中文也行）。")
    _print("  2. 保存后不用重启：下一次 T.help() 立刻生效（同一进程也算）。")
    _print("  3. 跑 T.help.check() 看覆盖率、死链、缺边；要提交回仓库就复制到 <repo>/wiki/。")
    return {"id": wiki_id, "existing": [], "created": [path]}
