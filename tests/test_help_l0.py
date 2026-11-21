# -*- coding: utf-8 -*-
"""L0 清单的**可达性自证**测试（2026-11-21）。

### 为什么有这个文件

`T.help('op')` 曾经写着「调用方式是 `t.<名字>(...)`，例如 `t.mul(x)`」，
而 Tensor 上**根本没有** `.mul()` —— 照文档写会 AttributeError。文档列出一个
调不到的入口，比不写更坏：用户会以为是自己写错了。本文件把"清单里的每个名字
都真的可达"钉成 CI 断言：

1. 每条 L0 条目必须声明 `reachable` / `form` / `form_kind` / `reach_evidence`；
   `reachable=False` 的必须带 `unreachable` 原因，且**不许**再留假调用示例。
2. **op**：方法形式必须真的 `getattr(Tensor, 名字)` 可调用；6 个运算符条目
   必须**没有**同名方法（这正是它们单列的原因），dunder 存在且表达式真的能算。
3. **api**：`getattr(T, 名字)` 真的存在且可调用（类=构造调用）。
4. **kernel**：名字必须出现在 NVRTC 真正编译的 `cuda.py::_device_source()` 里
   （本测试独立 `import` cuda 复核，顺便验证 help_l0 的静态重建与真实函数一致）。
5. **cli**：flag 必须来自真实 `add_argument(...)`，且模块有 `__main__` 入口。
6. **env**：可达条目必须有真实读取点（`os.environ` / 项目内 env 助手 / 别名常量的
   真实使用）；只在注释和报错文案里出现的"幽灵变量"必须标 `reachable=False`。
7. `T.help('op')` 的终端文案不许再出现不存在的调用形式，且列表里每个名字都验证可得。
"""

import ast
import importlib
import json
import os
import re
import sys

import pytest

import tgai_go.train as T
from tgai_go.train import help_l0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "python", "tgai_go", "train")

ENTRY_FIELDS = ("id", "kind", "name", "reachable", "form", "form_kind",
                "reach_evidence", "unreachable")
FORM_KINDS = {"method", "operator", "class", "function", "kernel", "cli",
              "env", "plugin"}
OPERATOR_FORMS = {
    "add": "a + b",
    "sub": "a - b",
    "mul": "a * b",
    "div": "a / b",
    "matmul": "a @ b",
    "pow": "a ** 2",
}
OPERATOR_DUNDERS = {
    "add": "__add__", "sub": "__sub__", "mul": "__mul__",
    "div": "__truediv__", "matmul": "__matmul__", "pow": "__pow__",
}


def _catalog():
    return help_l0.catalog()


def _by_kind(kind):
    return [e for e in _catalog() if e["kind"] == kind]


# --------------------------------------------------- 1. 每条都必须自证（不许静默）
def test_every_entry_declares_reachability_fields():
    entries = _catalog()
    assert len(entries) > 300
    for e in entries:
        for field in ENTRY_FIELDS:
            assert field in e, "%s 缺字段 %r" % (e.get("id"), field)
        assert isinstance(e["reachable"], bool), e["id"]
        assert e["form_kind"] in FORM_KINDS, e["id"]
        if e["reachable"]:
            # 可达就必须给出"真实写法"；唯一例外是插件条目（形式由插件决定）
            assert e["form"] or e["kind"] == "plugin", e["id"]
            assert not e["unreachable"], e["id"]
        else:
            assert e["unreachable"], "%s 不可调用却没写原因" % e["id"]
            assert not e["example"], "%s 不可调用却仍有假示例 %r" % (e["id"], e["example"])


# ------------------------------------------------------------------- 2. op 轴
def test_op_methods_really_exist_and_are_callable():
    rows = [e for e in _by_kind("op") if e["form_kind"] == "method"]
    assert len(rows) >= 35
    for e in rows:
        fn = getattr(T.Tensor, e["name"], None)
        assert callable(fn), "op:%s 声明是方法形式，但 Tensor 上拿不到" % e["name"]
        assert e["form"].startswith("x.") and e["reachable"]


def test_op_operators_are_not_fake_methods_and_really_work():
    operators = [e for e in _by_kind("op") if e["form_kind"] == "operator"]
    assert {e["name"] for e in operators} == set(OPERATOR_FORMS)
    a = T.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = T.Tensor([[5.0, 6.0], [7.0, 8.0]])
    results = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b,
               "matmul": a @ b, "pow": a ** 2}
    for e in operators:
        name = e["name"]
        # 它们**不是**方法 —— 这正是"单列 + form 写运算符"的原因
        assert not hasattr(T.Tensor, name), \
            "op:%s 若真有方法形式，就不该再用运算符写法" % name
        assert hasattr(T.Tensor, OPERATOR_DUNDERS[name]), name
        assert e["form"] == OPERATOR_FORMS[name], name
        out = results[name]
        assert tuple(out.shape) == (2, 2), name
        # 运算符写出来的就是"真实可用"，而不是一个名字
        assert not e["example"].endswith(")")


def test_op_mul_trap_documented_in_operator_entry():
    """全仓最有价值的新人坑：`*` 是逐元素，矩阵乘要 `@` / `T.matmul`。"""
    mul = [e for e in _by_kind("op") if e["name"] == "mul"][0]
    matmul = [e for e in _by_kind("op") if e["name"] == "matmul"][0]
    assert "逐元素" in mul["note"] and "没有 .mul()" in mul["note"]
    assert matmul["form"] == "a @ b"
    assert "T.matmul" in matmul["note"]


# ------------------------------------------------------------------ 3. api 轴
def test_api_entries_are_getattr_able_and_callable():
    rows = _by_kind("api")
    assert len(rows) >= 100
    for e in rows:
        obj = getattr(T, e["name"], None)
        assert obj is not None, "api:%s 在 T 上拿不到" % e["name"]
        assert callable(obj), "api:%s 存在但不可调用" % e["name"]
        expected = "class" if isinstance(obj, type) else "function"
        assert e["form_kind"] == expected, e["id"]
        assert e["form"].startswith("T." + e["name"])


# --------------------------------------------------------------- 4. kernel 轴
def _compiled_kernel_names():
    """独立复核：import 真正的 cuda 模块，拿 `_device_source()`（NVRTC 编译的源码）。"""
    cuda = importlib.import_module("tgai_go.train.cuda")
    source = cuda._device_source()
    return set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', source))


def test_kernel_entries_are_in_nvrtc_device_source():
    compiled = _compiled_kernel_names()
    assert len(compiled) > 100
    kernels = _by_kind("kernel")
    assert len(kernels) == len(compiled)
    for e in kernels:
        assert e["reachable"], e["id"]
        assert e["name"] in compiled, "kernel:%s 不在 _device_source() 里" % e["name"]
        assert e["form"] == 'rt.launch("%s", grid, block, [...])' % e["name"]
    # help_l0 里的**静态重建**必须与真实函数完全一致，否则自证就是假的
    assert help_l0.device_source_kernel_names() == compiled


# ------------------------------------------------------------------ 5. cli 轴
def test_cli_entries_have_real_add_argument_and_main_guard():
    rows = _by_kind("cli")
    assert len(rows) >= 30
    for e in rows:
        assert e["reachable"], e["id"]
        assert e["form"].startswith("python -m ")
        filename = e["module"].rsplit(".", 1)[-1] + ".py"
        with open(os.path.join(TRAIN, filename), encoding="utf-8") as fh:
            text = fh.read()
        assert re.search(r"add_argument\(\s*[\"']%s[\"']" % re.escape(e["name"]), text), \
            "cli:%s 在 %s 里找不到 add_argument" % (e["name"], filename)
        assert "__main__" in text and "__name__" in text


# ------------------------------------------------------------------ 6. env 轴
#: 只在注释/报错文案里出现、代码从不读取的"幽灵环境变量"（必须标不可达）。
GHOST_ENV_NAMES = {"TG_DEBUG", "TGAI_ATTN_FLASH", "TG_CUDA_GRAPH"}


def test_env_entries_have_real_read_points():
    rows = _by_kind("env")
    assert len(rows) >= 60
    reachable = [e for e in rows if e["reachable"]]
    unreachable = [e for e in rows if not e["reachable"]]
    assert len(reachable) >= len(rows) - 5
    # 幽灵变量必须被显式标出来，而不是混在正常变量里骗人
    assert {e["name"] for e in unreachable} <= GHOST_ENV_NAMES
    for e in reachable:
        assert ":" in e["reach_evidence"], e["id"]
        filename, rest = e["reach_evidence"].split(":", 1)
        assert os.path.isfile(os.path.join(TRAIN, filename)), e["reach_evidence"]
        lineno = int(rest.split(" ", 1)[0])
        assert lineno >= 1
    for e in unreachable:
        assert "没有读取点" in e["unreachable"], e["id"]
        assert e["form"] == "" and e["example"] == ""


def test_env_alias_and_helper_reads_are_detected():
    """别名常量（AMP_ENV = "TG_AMP"）与 env 助手（_env_int("TG_X", ...)）也算读取。"""
    by_id = {e["id"]: e for e in _by_kind("env")}
    for name in ("TG_AMP", "TG_FP16", "TG_DP_BUCKET_MB", "TG_ZERO_CHUNK_MB"):
        assert by_id["env:" + name]["reachable"], name
        ev = by_id["env:" + name]["reach_evidence"]
        assert "别名" in ev or "助手" in ev, ev


# ------------------------------------------- 7. 终端文案不再教不存在的调用方式
def test_help_op_text_does_not_claim_nonexistent_forms():
    text = T.help("op", print=False)
    assert "t.mul(x)" not in text
    assert "t.<名字>" not in text
    # 必须写清真实形态
    assert "a * b" in text and "a @ b" in text and "T.matmul" in text
    assert ".mul()" in text and "AttributeError" in text


def test_help_op_list_contains_only_real_tensor_methods():
    text = T.help("op", print=False)
    names = re.findall(r"^\s+\d+\. (\w+)$", text, re.M)
    assert len(names) >= 30
    for name in names:
        assert callable(getattr(T.Tensor, name, None)), \
            "T.help('op') 列出了不存在的 Tensor 方法: %s" % name


def test_json_exposes_reachability_and_flags_ghosts(capsys):
    """`env:TG_DEBUG` 以前是"幽灵变量"（192 处文案让用户设它，却没人读）。

    2026-11-21 把它**修实**了（`CudaError` 现在真的读它并附诊断），
    所以判据从"必须报不可达"翻转为"必须报可达且给出真实形式"。
    另外两个幽灵（`TGAI_ATTN_FLASH` / `TG_CUDA_GRAPH`）的**假名字已从源码删除**，
    换成了真正会被读的 `TG_NO_FUSED` / "调用 graph_arm()" 的说明，因此它们不再出现。

    这条测试的真正职责没变：**L0 里任何一条 env，都必须真的有读取点**。
    """
    T.help("TG_DEBUG", format="json")
    payload = json.loads(capsys.readouterr().out)
    row = [it for it in payload["items"] if it["id"] == "env:TG_DEBUG"]
    assert row, "env:TG_DEBUG 应该出现在清单里"
    assert row[0]["reachable"] is True, "TG_DEBUG 现在有真实读取点，必须报可达"
    assert row[0].get("form"), "可达条目必须给出真实使用形式"

    # 全局不变量：清单里不许再有任何"不可达"的 env
    bad = [it["id"] for it in payload["items"]
           if it.get("kind") == "env" and it.get("reachable") is False]
    assert bad == [], "env 里还有幽灵变量：%s" % (bad,)


def test_reachability_summary_counts_are_sane():
    """给报告用的汇总数字（也防止将来某类整体变成不可达而没人发现）。"""
    summary = {}
    for kind in ("op", "api", "kernel", "cli", "env"):
        rows = _by_kind(kind)
        summary[kind] = (len(rows), sum(1 for e in rows if e["reachable"]))
    assert summary["op"][1] == summary["op"][0]
    assert summary["api"][1] == summary["api"][0]
    assert summary["kernel"][1] == summary["kernel"][0]
    assert summary["cli"][1] == summary["cli"][0]
    # 2026-11-21：三个幽灵 env 已修实/删名，env 现在应当 **100% 可达**。
    # 保留"必须全可达"这条断言比"恰好有 3 个不可达"更有价值：
    # 以后谁再往文案里写一个没人读的变量，这里会红。
    assert summary["env"][1] == summary["env"][0], (
        "env 里出现了不可达条目（文案提到但没人读）：%s"
        % ([e["id"] for e in _by_kind("env") if not e["reachable"]],))
