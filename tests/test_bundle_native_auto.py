# -*- coding: utf-8 -*-
"""插件包：原生算子免写反向（`form: native` + `fallback`）与「一条命令装完环境」。

这两件事是「轮椅化」的核心：
* 作者只写 .cu/.dll 快路径 + 一份用已有算子写的等价实现 ⇒ 反向由 ORV 自动导出；
* 用户一行 `T.plugin.install_all("bundles/")` 装完整个环境。
"""
import json
import os

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import plugin_meta


IMPL = '''
import numpy as np


def fast(x):
    """假装这是原生 kernel 的包装（真实项目里这里会调 .dll/.so）。"""
    return (x * 2.0).astype(np.float32)


def fallback(x):
    """与 fast 数学等价、用 ORV 已有算子写 ⇒ 反向由它自动导出。"""
    return x * 2.0


def bad_fallback(x):
    return x * 3.0
'''


def _write_bundle(root, name, native_op=True, version="0.1.0", bad=False):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "impl.py"), "w", encoding="utf-8") as f:
        f.write(IMPL)
    if native_op:
        op = {
            "name": "nat_double",
            "arity": 1,
            "form": "native",
            "impl": {"module": "impl.py", "python": "fast",
                     "fallback": "bad_fallback" if bad else "fallback"},
        }
    else:
        op = {
            "name": "cmp_double",
            "arity": 1,
            "form": "composed",
            "impl": {"module": "impl.py", "python": "fallback"},
        }
    manifest = {"schema": 1, "name": name, "version": version, "api_version": "1.0",
                "namespace": "demo", "backends": ["cpu", "cuda"], "ops": [op]}
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def test_native_op_with_fallback_gets_automatic_backward(tmp_path):
    root = tmp_path / "auto-native"
    _write_bundle(str(root), "auto-native")
    meta = T.plugin.install(str(root), replace=True)
    assert meta["name"] == "auto-native"

    x = T.from_numpy(np.array([1.0, 2.0, 3.0], np.float32))
    x.requires_grad = True
    y = T.demo.nat_double(x)
    assert np.allclose(y.to_numpy(), np.array([2.0, 4.0, 6.0], np.float32))
    y.sum().backward()
    g = x.grad
    g = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float32)
    assert np.allclose(g, np.array([2.0, 2.0, 2.0], np.float32)), \
        "反向应由 fallback 自动导出（d(2x)/dx = 2）"


def test_mismatched_fallback_is_caught_on_first_backward(tmp_path):
    root = tmp_path / "bad-native"
    _write_bundle(str(root), "bad-native", bad=True)
    T.plugin.install(str(root), replace=True)
    x = T.from_numpy(np.array([1.0, 2.0], np.float32))
    x.requires_grad = True
    y = T.demo.nat_double(x)
    with pytest.raises(ValueError) as ei:
        y.sum().backward()
    assert "不一致" in str(ei.value)


def test_install_all_installs_every_bundle_in_a_directory(tmp_path):
    base = tmp_path / "env"
    _write_bundle(str(base / "pkg-a"), "pkg-a")
    _write_bundle(str(base / "pkg-b"), "pkg-b", native_op=False, version="0.2.0")
    metas = T.plugin.install_all(str(base), replace=True)
    names = sorted(m["name"] for m in metas)
    assert names == ["pkg-a", "pkg-b"], names
    installed = {p["name"] for p in T.plugin.list_installed()}
    assert {"pkg-a", "pkg-b"} <= installed


def test_install_all_reports_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        T.plugin.install_all(str(empty))
    assert ".tgop" in str(ei.value)


def test_install_all_surfaces_which_package_failed(tmp_path):
    base = tmp_path / "env2"
    _write_bundle(str(base / "good"), "good")
    broken = base / "broken"
    broken.mkdir()
    with open(str(broken / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "name": "broken", "version": "1.0.0",
                   "api_version": "1.0", "backends": ["cpu"], "ops": []}, f)
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        T.plugin.install_all(str(base))
    assert "broken" in str(ei.value), "要指出是哪个包坏了"
