# -*- coding: utf-8 -*-
"""核对语料标记是否匹配**模型自带的 V2.7 对话模板**。

为什么必须核：语料文件名里的 `V2.6`/`V2.7` 不可信，而标记模板换过版本。
喂错格式的训练对比等于在测"格式不匹配"，不是测性能。

做法：从 `.TG` 里读内嵌的 chat-meta（`render_mode` / `template_turn` / `turn_markers` /
`stop_strings`），再到语料首行里**逐个查这些标记是否真的出现**。

用法：`python check_markers.py <bench_cfg.json> [语料路径]`
"""
import io
import json
import sys


def main():
    cfg_path = sys.argv[1]
    with io.open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    corpus = sys.argv[2] if len(sys.argv) > 2 else cfg["corpus"]

    import tgai_go
    cm = tgai_go.chat_meta_of(cfg["model_tg"])
    print("=== 模型内嵌 chat-meta ===")
    if not cm.get("present"):
        print("(该模型没有内嵌对话配置)")
        return 1
    for k in ("render_mode", "chat_template", "template_turn", "system_prompt",
              "special_tokens", "turn_markers", "stop_strings", "gen_defaults"):
        if k in cm:
            print("%-16s %s" % (k, cm[k]))

    # 收集模板里的"标记"文本（去重、去掉过短的）
    marks = set()
    for key in ("turn_markers", "special_tokens"):
        v = cm.get(key)
        if isinstance(v, str):
            for part in v.replace(";", ",").split(","):
                p = part.strip().strip('"').strip("'")
                if 1 <= len(p) <= 24:
                    marks.add(p)
        elif isinstance(v, (list, tuple, dict)):
            items = v if isinstance(v, (list, tuple)) else v.values()
            for it in items:
                s = str(it).strip()
                if 1 <= len(s) <= 24:
                    marks.add(s)
    for key in ("template_turn", "chat_template"):
        v = cm.get(key)
        if isinstance(v, str):
            for tok in ("用户", "TGAI", "系统", "思考过程", "回答", "记忆", "工具"):
                if tok in v:
                    marks.add(tok)
    marks = sorted(marks)
    print("\n模板标记候选（%d 个）: %s" % (len(marks), marks))

    print("\n=== 语料首 2 条 ===")
    lines = []
    with io.open(corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= 2:
                break
    texts = []
    for line in lines:
        try:
            rec = json.loads(line)
            texts.append(rec.get("text") or rec.get("content") or line)
        except ValueError:
            texts.append(line)
    for i, t in enumerate(texts):
        print("  [%d] %s" % (i, t.replace("\n", " ")[:200]))

    print("\n=== 标记命中情况（在首 2 条里）===")
    joined = "\n".join(texts)
    hit = [m for m in marks if m in joined]
    miss = [m for m in marks if m not in joined]
    print("命中 %d/%d: %s" % (len(hit), len(marks), hit))
    if miss:
        print("未出现: %s" % miss)
    print("\n判读:", "标记基本匹配，可用这份语料" if len(hit) >= max(1, len(marks) // 2)
          else "**标记不匹配**，别用这份语料做 V2.7 的 SFT 对比")
    return 0


if __name__ == "__main__":
    sys.exit(main())
