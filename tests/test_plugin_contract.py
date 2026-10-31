# -*- coding: utf-8 -*-
"""插件生态的地基：契约校验 + 来源指纹 + 严格模式 + 数值自检。

生态能不能起来，取决于**别人写插件时会不会踩坑没人管**。这条测试守住四件事：
1. 契约写错（缺字段/api 版本不匹配/声明了没注册的算子）必须被**指出来**；
2. 加载插件必须记录 **sha256**（出问题能追到具体文件）；
3. `TGAI_OPS_STRICT=1` 下，没有契约的插件必须被拒（给 CI/团队用）；
4. 检验器能抓到**写错的反向**（否则这个工具就是摆设）。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T
from tgai_go.train import ops_plugin as P


def _load_checker():
    path = os.path.join(REPO, "tools", "check_plugin.py")
    spec = importlib.util.spec_from_file_location("check_plugin_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


@pytest.fixture()
def cleanup():
    """测试结束后把注册过的算子清掉，别污染其它用例。"""
    before = set(P.registered_names())
    yield
    for name in set(P.registered_names()) - before:
        try:
            P.unregister(name)
        except Exception:                       # noqa: BLE001
            pass


def test_contract_problems_are_reported():
    ok = {"name": "x", "version": "0.1.0", "api_version": P.API_VERSION, "ops": {}}
    assert P.check_contract(ok, ()) == []

    assert any("缺少 name" in p for p in P.check_contract({"version": "1", "api_version": 1}))
    assert any("api_version" in p for p in
               P.check_contract({"name": "x", "version": "1", "api_version": 999}))
    # 声明了没注册的算子
    probs = P.check_contract({"name": "x", "version": "1", "api_version": P.API_VERSION,
                              "ops": {"nope": {}}}, added_ops=["other"])
    assert any("没有注册" in p for p in probs), probs
    # test_inputs 类型不对
    probs = P.check_contract({"name": "x", "version": "1", "api_version": P.API_VERSION,
                              "ops": {"o": {"test_inputs": "不是列表"}}}, added_ops=["o"])
    assert any("test_inputs" in p for p in probs), probs


def test_manifest_records_digest_and_ops(cleanup):
    path = os.path.join(REPO, "examples", "plugins", "swiglu.py")
    P.load_plugin_file(path)
    rec = P.manifest()[-1]
    assert rec["file"].endswith("swiglu.py")
    assert len(rec["sha256"]) == 64 and all(c in "0123456789abcdef" for c in rec["sha256"])
    assert "swiglu" in rec["ops"]
    assert rec["api_version"] == P.API_VERSION
    assert not rec["contract_problems"]


def test_strict_mode_rejects_plugin_without_contract(tmp_path, cleanup, monkeypatch):
    p = tmp_path / "no_contract.py"
    p.write_text(
        "import tgai_go.train as T\n"
        "from tgai_go.train import ops_plugin\n"
        "ops_plugin.register('no_contract_op', lambda x: x * 2.0, doc='无契约')\n",
        encoding="utf-8")
    # 默认宽松：能加载，但契约问题要记下来
    P.load_plugin_file(str(p))
    assert P.manifest()[-1]["contract_problems"], "没有契约时应该记录问题"
    assert "no_contract_op" in P.registered_names()
    P.unregister("no_contract_op")            # 给严格模式让路（同名会先撞"已注册"）

    monkeypatch.setenv("TGAI_OPS_STRICT", "1")
    with pytest.raises(ValueError) as ei:
        P.load_plugin_file(str(p))
    assert "PLUGIN" in str(ei.value) and "修法" in str(ei.value)
    # 关键：被拒绝的插件不能留下任何算子（否则"拒绝"是假的）
    assert "no_contract_op" not in P.registered_names(), "拒绝后应回滚已注册的算子"


def test_example_plugins_pass_numeric_check():
    """两个示例插件都要能通过检验器（前向 + 中心差分对拍梯度）。"""
    tool = _load_checker()
    for name in ("squared_relu.py", "swiglu.py"):
        path = os.path.join(REPO, "examples", "plugins", name)
        mod = P.load_plugin_file(path)
        info = mod.PLUGIN
        for op, spec in info["ops"].items():
            fn = getattr(T.functional, op)
            ok, detail = tool.check_op(op, fn, spec["test_inputs"])
            assert ok, "%s: %s" % (op, detail)
        for op in info["ops"]:
            if op in P.registered_names():
                P.unregister(op)


def test_checker_catches_wrong_backward(tmp_path):
    """反向少乘一个系数必须被抓到 —— 否则检验器就是摆设。"""
    tool = _load_checker()
    p = tmp_path / "bad.py"
    p.write_text(
        "import numpy as np\n"
        "from tgai_go.train import ops_plugin\n"
        "def fwd(x):\n"
        "    return np.square(np.maximum(x, 0.0))\n"
        "def bwd(g, out, xs):\n"
        "    x = np.asarray(xs[0], dtype=np.float32)\n"
        "    return [np.asarray(g, np.float32) * np.maximum(x, 0.0)]\n"
        "ops_plugin.register_numpy('bad_act', fwd, bwd, doc='故意错')\n"
        "PLUGIN = {'name': 'bad_act', 'version': '0.1.0',\n"
        "          'api_version': ops_plugin.API_VERSION,\n"
        "          'ops': {'bad_act': {'test_inputs': [\n"
        "              {'shape': [2, 4], 'low': -2.0, 'high': 2.0, 'seed': 1}]}}}\n",
        encoding="utf-8")
    mod = P.load_plugin_file(str(p))
    fn = getattr(T.functional, "bad_act")
    ok, detail = tool.check_op("bad_act", fn, mod.PLUGIN["ops"]["bad_act"]["test_inputs"])
    assert not ok, "写错的反向竟然通过了检查"
    assert "差分" in detail
    P.unregister("bad_act")


def test_api_version_is_exposed():
    assert isinstance(P.API_VERSION, int) and P.API_VERSION >= 1
