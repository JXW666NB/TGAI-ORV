# -*- coding: utf-8 -*-
"""NumPy 参考后端的**覆盖度棘轮**：只许进步，不许退步。

先说清定位（`docs/BACKEND_PORTING.md` 里写明）：
`numpy_ref` 是**接口模板**，不是产品后端；产品后端是 CUDA。
它当前实现 36/79（文档里写作"核心 36/54"），缺的部分是 MoE / 优化器 /
融合核 / 注意力反向等 —— 这些在模板里没有实现是**预期行为**，不是 bug。

那么这条测试要守什么？
1. **不倒退**：已实现的算子数不能少于文档里写的数字（谁删了实现就红）；
2. **已满的组保持满**：形状/布局、归约、GEMM、逐元素四组当前 100%，
   以后也不许掉（它们是模板里最有价值的示范）；
3. **文档与代码一致**：文档里的数字要能从代码复现，避免"文档写着 36，实际 30"。

注意：**不要**把这条测试写成"必须 79/79" —— 那会逼着人去写一堆没人用的模板实现，
或者干脆改测试数字蒙过去。
"""
import io
import os
import re

import numpy as np
import pytest

from tgai_go.train import backend
from tgai_go.train.backends import numpy_ref

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: 当前 100% 的四组 —— 掉一个都算退步
FULL_GROUPS = ("形状/布局", "归约", "GEMM", "逐元素")


@pytest.fixture(scope="module")
def rt():
    return numpy_ref.make()


def _coverage(rt):
    per = {}
    for group, ops in backend.GROUPS.items():
        per[group] = (sum(1 for o in ops if callable(getattr(rt, o, None))), len(ops))
    return per


def _documented_core_count():
    """从 BACKEND_PORTING.md 里读出"核心 N/54"里的 N。"""
    p = os.path.join(REPO, "docs", "BACKEND_PORTING.md")
    if not os.path.isfile(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    m = re.search(r"核心\s*\*\*(\d+)\s*/\s*(\d+)\*\*", txt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def test_full_groups_stay_full(rt):
    per = _coverage(rt)
    bad = []
    for g in FULL_GROUPS:
        if g not in per:
            continue
        got, total = per[g]
        if got != total:
            bad.append("%s: %d/%d" % (g, got, total))
    assert not bad, "这些组原来 100% 覆盖，现在掉了：%s" % bad


def test_coverage_does_not_regress_below_documented(rt):
    """已实现总数不能少于文档里写的数字（文档是承诺，代码是事实）。"""
    per = _coverage(rt)
    impl = sum(g for g, _ in per.values())
    doc = _documented_core_count()
    if doc is None:
        pytest.skip("BACKEND_PORTING.md 里没有『核心 N/54』的记录，跳过对比")
    n_doc = doc[0]
    # 文档写的是"核心"子集；这里比的是**总数**，所以下界就是那个数
    assert impl >= n_doc, (
        "numpy_ref 现在只实现 %d 个算子，但文档写着核心 %d 个 —— "
        "要么有人删了实现，要么文档没跟着改。逐组：%s"
        % (impl, n_doc, {k: "%d/%d" % v for k, v in per.items()}))


def test_missing_ops_are_the_expected_template_gaps(rt):
    """缺的算子应当集中在"模板本来就不做"的组里（MoE/优化器/融合核/注意力反向）。

    如果哪天"逐元素"或 "GEMM" 这种基础组里出现缺失，说明是新引入的洞，
    必须显式列出来让人看见，而不是悄悄少一个。
    """
    per = _coverage(rt)
    holes = {g: "%d/%d" % v for g, v in per.items() if v[0] < v[1]}
    basic = [g for g in ("形状/布局", "归约", "GEMM", "逐元素") if g in holes]
    assert not basic, "基础组不该有缺失：%s（全部缺失情况：%s）" % (basic, holes)


def test_smoke_on_implemented_ops_is_finite(rt):
    """已实现的算子跑一遍轻量冒烟，输出不能有 NaN/Inf。"""
    x = rt.h2d(np.random.rand(2, 4).astype(np.float32) + 0.5)
    probes = {
        "exp": lambda: rt.exp(x),
        "sumsum": lambda: rt.sum_all(x),
        "clone": lambda: rt.clone(x),
        "scale": lambda: rt.scale(x, 2.0),
        # transpose2d 的签名与 CUDA 侧不同（模板实现只吃 buf），
        # 已由 test_numpy_backend.py::test_reshape_permute_swap_broadcast 覆盖，这里不重复。
    }
    for name, fn in probes.items():
        out = fn()
        v = np.asarray(getattr(out, "arr", out), np.float32)
        assert np.isfinite(v).all(), "%s 输出含 NaN/Inf" % name
