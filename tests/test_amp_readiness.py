# -*- coding: utf-8 -*-
"""真 AMP 差距清单工具的自检：能扫到内核、分类齐全、统计自洽。

这条测试的作用不是"验证 bf16 实现"，而是**防止清单本身失真**：
这份清单是 L3「真 AMP」那一项的唯一量化依据，如果扫出来是 0 个内核、
或者分类丢了一半，计划就会建立错的前提上。
"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "tools", "amp_readiness.py")
    spec = importlib.util.spec_from_file_location("amp_readiness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scan_finds_kernels_and_classifies():
    ar = _load()
    ks = ar.scan()
    assert len(ks) >= 50, "只扫到 %d 个内核，明显不对（cuda.py 里应有 80 上下）" % len(ks)
    # 已知一定存在的内核
    names = {k["name"] for k in ks}
    for must in ("ewise_add", "rmsnorm_fwd", "ce_fwd_rows"):
        assert must in names, "清单里缺少 %s，扫描正则可能失效了" % must
    # 每个内核都要能落到一个类别
    for k in ks:
        assert ar.classify(k["name"]), k["name"]


def test_counts_are_self_consistent():
    ar = _load()
    ks = ar.scan()
    n_bf16 = sum(1 for k in ks if k["bf16"])
    assert n_bf16 <= len(ks)
    # 目前（未实现 bf16 内核变体）应当全是 float-only；
    # 将来做了 S1 之后这条会变，届时**主动改测试**并更新 docs/AMP_PLAN.md
    assert n_bf16 == 0, ("已有 %d 个内核支持 bf16 —— 说明 AMP 计划推进了，"
                         "请更新 docs/AMP_PLAN.md 与 README 的状态表" % n_bf16)


def test_plan_doc_exists():
    p = os.path.join(REPO, "docs", "AMP_PLAN.md")
    assert os.path.isfile(p), "缺 docs/AMP_PLAN.md（L3 计划的依据）"
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    assert "amp_readiness" in txt, "计划文档里应指回工具，保证数据可复现"
