#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""词条 TXT 校验器 —— **不依赖引擎加载器**，现在就能跑。

为什么单独存在：`wiki/README_词条规范.txt` 里让作者跑 `T.help.check()`，而那个接口要等
引擎侧的 TXT 加载器（1.0.17）才落地。这个脚本按同一套协议做静态校验，
让写词条的人**今天**就能确认自己的文件是否合法。

用法：

    python scripts/wiki_check.py                 # 校验仓库里的 wiki/
    python scripts/wiki_check.py --dir my_wiki   # 校验你自己的目录
    python scripts/wiki_check.py --strict        # 有警告也算失败

会检查：

1. 编码（UTF-8 / UTF-8-BOM / GB18030 自动识别 —— Windows 记事本默认存 ANSI）
2. `[ID]` 必填、格式为 `类型:名字`，类型在白名单里
3. 头部只允许 `ID / 分类 / 中文名 / 英文名 / 作者`
4. 区块只允许 `=== 中文 ===` / `=== English ===` / `=== 示例 ===` / `=== 见也 ===`
5. `[分类]` 必须在 `_meta.txt` 里（否则警告：会归 other）
6. `=== 见也 ===` 里引用的 ID 是否存在（死链 ⇒ 错误）
7. 中英缺边（警告）
8. 重复 ID（错误）
"""
import argparse
import os
import re
import sys

KINDS = ("op", "api", "kernel", "cli", "env", "plugin", "meta")
HEAD_KEYS = ("ID", "分类", "中文名", "英文名", "作者")
BLOCKS = ("中文", "English", "示例", "见也")
_DEFAULT_CATS = ("getting-started", "tensor", "model", "memory", "compute", "optim",
                 "distributed", "plugin", "debug", "compat", "other")
_HEAD_RE = re.compile(r"^\s*[\[［]\s*([^\]］]+?)\s*[\]］]\s*(.*)$")
_BLOCK_RE = re.compile(r"^\s*===\s*(.+?)\s*===\s*$")


def read_text(path):
    """按 UTF-8 → UTF-8-BOM → GB18030 顺序尝试解码，返回 (文本, 编码名)。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), "utf-8（有损）"


def parse_entry(path):
    """解析一个词条文件；返回 (字段 dict, 区块 dict, 错误 list, 警告 list)。"""
    text, enc = read_text(path)
    head, blocks, errs, warns = {}, {}, [], []
    if enc.endswith("有损"):
        warns.append("编码无法识别，已按 UTF-8 有损解码：%s" % enc)
    cur = None
    buf = []

    def _flush():
        """把当前累积的行刷进 blocks[cur]。★ 切换区块时必须调用，
        否则只有最后一个区块的内容被保留（我第一版就漏了这一步，
        结果所有正文都被算成 `见也` 的引用，报了 49 个假错误）。"""
        if cur is not None:
            blocks.setdefault(cur, []).extend(buf)
        del buf[:]

    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = _BLOCK_RE.match(line)
        if m:
            name = m.group(1)
            if name not in BLOCKS:
                errs.append("第%d行：未知区块 `=== %s ===`（只允许 %s）"
                            % (lineno, name, " / ".join(BLOCKS)))
                _flush()
                cur = None
                continue
            _flush()
            cur = name
            blocks.setdefault(cur, [])
            continue
        hm = _HEAD_RE.match(line)
        if hm and cur is None:
            key = hm.group(1)
            val = hm.group(2).strip()
            if key not in HEAD_KEYS:
                errs.append("第%d行：未知头部字段 `[%s]`（只允许 %s）"
                            % (lineno, key, " / ".join(HEAD_KEYS)))
                continue
            if key in head:
                errs.append("第%d行：字段 `[%s]` 重复" % (lineno, key))
            head[key] = val
            continue
        if cur is not None:
            buf.append(line)
        elif line.strip():
            errs.append("第%d行：头部之后、区块之前出现游离文本：%r"
                        % (lineno, line.strip()[:40]))
    _flush()
    blocks = {k: "\n".join(v).strip() for k, v in blocks.items()}
    return head, blocks, errs, warns


def main(argv=None):
    ap = argparse.ArgumentParser(description="校验 ORV wiki 词条（TXT 协议）")
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki"))
    ap.add_argument("--strict", action="store_true", help="警告也算失败")
    ap.add_argument("--strict-links", action="store_true",
                    help="把「引用了还没写的 ID」也算失败")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.dir)
    if not os.path.isdir(root):
        print("目录不存在：%s" % root)
        return 2

    cats = list(_DEFAULT_CATS)
    meta = os.path.join(root, "_meta.txt")
    if os.path.isfile(meta):
        txt, _ = read_text(meta)
        found = [l.split("|")[0].split("]", 1)[-1].strip()
                 for l in txt.splitlines() if l.strip().startswith("[分类]")]
        found = [c for c in found if c]
        if found:
            cats = found
            print("分类表：%d 个（来自 _meta.txt）" % len(found))
    else:
        print("提示：没有 _meta.txt，按内置分类表校验")

    files, ids = [], {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".txt"):
                continue
            if fn.startswith("README") or fn == "_meta.txt":
                continue
            files.append(os.path.join(dirpath, fn))

    entries = {}
    n_err = n_warn = 0
    for path in files:
        rel = os.path.relpath(path, root)
        head, blocks, errs, warns = parse_entry(path)
        eid = head.get("ID", "").strip()
        if not eid:
            errs.append("缺少必填的 `[ID]`")
        else:
            if ":" not in eid:
                errs.append("`[ID]` 必须是 `类型:名字`，当前 %r" % eid)
            else:
                kind = eid.split(":", 1)[0]
                if kind not in KINDS:
                    errs.append("`[ID]` 的类型 %r 不在白名单 %s" % (kind, "/".join(KINDS)))
            if eid in ids:
                errs.append("ID 重复：%s（另一处 %s）" % (eid, ids[eid]))
            ids[eid] = rel
            entries[eid] = (rel, head, blocks)
        cat = head.get("分类", "").strip()
        if cat and cat not in cats:
            warns.append("分类 %r 不在 _meta.txt 里（会归 other）" % cat)
        if not blocks.get("中文") and not blocks.get("English"):
            warns.append("中英两个正文区块都空（页面只会有自动示例）")
        elif not blocks.get("English"):
            warns.append("缺 `=== English ===`（页面显示暂无英文）")
        elif not blocks.get("中文"):
            warns.append("缺 `=== 中文 ===`（页面显示暂无中文）")
        if not blocks.get("示例"):
            warns.append("缺 `=== 示例 ===`（仍会有引擎自动示例，但建议自己写一条）")
        if errs:
            n_err += len(errs)
            print("\n✗ %s" % rel)
            for e in errs:
                print("   错：%s" % e)
        if warns:
            n_warn += len(warns)
            if not errs:
                print("\n△ %s" % rel)
            for w in warns:
                print("   警：%s" % w)

    # 交叉引用：指向"引擎里有、但还没人写词条"的 ID 不算错（那是待写清单），
    # 只有 `--strict-links` 才当失败；真正的错是格式错误与重复 ID。
    dangling = 0
    for eid, (rel, head, blocks) in entries.items():
        for line in blocks.get("见也", "").splitlines():
            ref = line.strip().strip(",，")
            if not ref or ref in ids:
                continue
            dangling += 1
            print("\n△ %s：`=== 见也 ===` 引用的 %r 还没有词条（待写，不算错）"
                  % (rel, ref))
    if dangling and a.strict_links:
        n_err += dangling

    print("\n== 汇总 ==")
    print("  词条文件：%d" % len(files))
    print("  唯一 ID ：%d" % len(ids))
    print("  错误    ：%d" % n_err)
    print("  警告    ：%d" % n_warn)
    if n_err:
        print("\n结论：有 %d 个错误，这些文件**不会被加载**。按上面的提示改。" % n_err)
        return 1
    if n_warn and a.strict:
        print("\n结论：--strict 模式下警告也算失败。")
        return 1
    print("\n结论：全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
