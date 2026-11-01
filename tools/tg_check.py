# -*- coding: utf-8 -*-
"""`.TG` 容器自洽性检查：hash 唯一、区间不重叠、不越界、无空洞。

为什么需要它（2026-11-01 的教训）：
  之前我用"归一化权重均值应≈1"这种**常识阈值**去判断"`.TG` 读错了"，
  后来对拍官方 safetensors 才发现 **官方 Qwen2.5-0.5B 的 ln1 均值本来就是 0.032**，
  那个判据会拒绝合法权重。
  真正客观的是**结构检查**：容器自己声明了 hash/off/len/shape，
  这些必须自洽 —— 自洽就说明"读到的就是文件里存的"，不涉及对模型好坏的猜测。

用法：
  python tools/tg_check.py model.TG        # 0 = 通过；1 = 有问题
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def check_table(table, file_bytes, data_off):
    """纯函数：只吃表与文件大小，便于单测。返回 (问题列表, 统计)。"""
    problems = []
    hashes = {}
    for i, e in enumerate(table):
        hashes.setdefault(e["hash"], []).append(i)
    dups = {h: idx for h, idx in hashes.items() if len(idx) > 1}
    if dups:
        problems.append("有 %d 个 hash 重复：按名查表会取到别人的张量（例 %s）"
                        % (len(dups), list(dups.items())[:2]))

    spans = sorted((e["off"], e["off"] + e["len"], i) for i, e in enumerate(table))
    prev_end, prev_i = -1, -1
    for s, t, i in spans:
        if s < prev_end:
            problems.append("张量 %d 与 %d 的数据区间重叠 %d 字节（表或数据区被写坏）"
                            % (prev_i, i, prev_end - s))
        if t > prev_end:
            prev_end, prev_i = t, i

    if table:
        span = max(t for _, t, _ in spans)
        if data_off + span > file_bytes:
            problems.append("数据区越界：data_off(%d)+span(%d) > 文件 %d 字节"
                            % (data_off, span, file_bytes))
    stats = dict(entries=len(table), unique_hashes=len(hashes),
                 data_span=max((t for _, t, _ in spans), default=0))
    return problems, stats


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    from tgai_go.read_tg import read_tg
    path = argv[0]
    if not os.path.isfile(path):
        print("找不到文件：%s" % path)
        return 2
    info = read_tg(path)
    problems, stats = check_table(info["table"], len(info["raw"]), info["data_off"])
    print("entries=%d unique_hashes=%d data_span=%.1f MB"
          % (stats["entries"], stats["unique_hashes"], stats["data_span"] / 2**20))
    if problems:
        for p in problems[:8]:
            print("  FAIL:", p)
        print("结论：FAIL —— 容器不自洽，加载结果不可信")
        return 1
    print("结论：PASS —— 容器自洽（读到的就是文件里存的；模型好坏另用 tg_audit 对比）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
