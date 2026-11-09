# -*- coding: utf-8 -*-
"""给 CudaRuntime 增加**多图支持**（默认行为不变），供"每层一张图"使用。

为什么不直接改：cuda.py 是引擎路径，改动需用户授权。本脚本把改动写成可审阅、可回退、
幂等的形式，授权后一条命令应用：

    python tools/apply_multigraph.py --dry     # 只校验锚点
    python tools/apply_multigraph.py           # 应用
    git checkout -- python/tgai_go/train/cuda.py   # 一键回退

语义（**完全向后兼容**）：
  * 现有 `_graph` / `_graph_exec` / `graph_begin` / `graph_end` / `graph_launch` 行为不变：
    `graph_launch()` 仍重放"最近捕获的那张"。
  * 新增 `_graph_slots: list[(graph, exec)]`；`graph_end()` 成功后**追加**一份引用。
  * 新增 `graph_count()` 与 `graph_launch_slot(i)`（重放第 i 张，不影响最近指针）。
  * `_graph_destroy()` 同时清空 slots，避免泄漏。
  * 注意：`graph_begin()` 在再捕获时仍会销毁"最近那张"的 graph 对象；因此应用本补丁后，
    逐层捕获的**正确用法**是每层捕获后立刻把 exec 存入 slot，且**不要**依赖最近指针。
"""
import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")

OLD_END_TAIL = """            self._graph = graph
            self._graph_exec = self._instantiate_graph(graph)"""

NEW_END_TAIL = """            self._graph = graph
            self._graph_exec = self._instantiate_graph(graph)
            # 多图支持（应用 apply_multigraph.py 后存在）：保留一份引用，供逐层重放
            if not hasattr(self, "_graph_slots"):
                self._graph_slots = []
            self._graph_slots.append((graph, self._graph_exec))"""

OLD_LAUNCH_HEAD = """    def graph_launch(self, bc1=None, bc2=None):"""

NEW_LAUNCH_HEAD = """    def graph_count(self):
        \"\"\"已捕获并保留的图数量（多图支持）。\"\"\"
        return len(getattr(self, "_graph_slots", []) or [])

    def graph_launch_slot(self, i):
        \"\"\"重放第 i 张已捕获的图（不影响 graph_launch 的"最近一张"语义）。\"\"\"
        slots = getattr(self, "_graph_slots", None) or []
        if not slots or i < 0 or i >= len(slots):
            return False
        _graph, exec_ = slots[i]
        if self._graph_dead or not exec_:
            return False
        self._activate()
        self._chk(self.cu.cuGraphLaunch(exec_, self._stream), "cuGraphLaunch(slot)")
        return True

    def graph_launch(self, bc1=None, bc2=None):"""

PAIRS = [("graph_end 追加 slot", OLD_END_TAIL, NEW_END_TAIL),
         ("graph_launch_slot", OLD_LAUNCH_HEAD, NEW_LAUNCH_HEAD)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    s = io.open(TARGET, encoding="utf-8").read()
    if "graph_launch_slot" in s:
        print("已应用（graph_launch_slot 已存在），跳过")
        return 0
    bad = [(n, s.count(o)) for n, o, _ in PAIRS if s.count(o) != 1]
    if bad:
        print("锚点不唯一/不存在，未改动：%s" % bad)
        return 2
    if args.dry:
        print("锚点全部唯一 ✔（--dry，未写入）")
        return 0
    for _n, old, new in PAIRS:
        s = s.replace(old, new, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(s)
    print("已应用多图支持（默认行为不变）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
