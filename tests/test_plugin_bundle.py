# -*- coding: utf-8 -*-
"""插件包（`.tgop`）与元信息规范的回归测试。

覆盖三件事：
1. 元信息校验（名字 / 语义化版本 / 后端 / C 档必须有回退）
2. 从**目录**一行安装整个包，并拿到 `T.<namespace>`
3. 从**zip（.tgop）**一行安装同样有效

不需要 CUDA：包里的算子用 numpy 实现即可，这里只验装载链路。
"""
import json
import os
import zipfile

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import plugin_meta


def _write_bundle(root, name="demo-ops", version="0.1.0"):
    os.makedirs(root, exist_ok=True)
    impl = '''
import numpy as np


def double(x):
    """把一个张量乘 2（纯组合式，梯度由 autograd 给）。"""
    return x * 2.0


def shift(x, k=1.0):
    return x + k
'''
    with open(os.path.join(root, "impl.py"), "w", encoding="utf-8") as f:
        f.write(impl)
    manifest = {
        "schema": 1,
        "name": name,
        "version": version,
        "api_version": "1.0",
        "namespace": "demo",
        "summary": "测试用插件包",
        "backends": ["cpu", "cuda"],
        "ops": [
            {"name": "double", "arity": 1, "form": "composed",
             "impl": {"module": "impl.py", "python": "double"}},
            {"name": "shift", "arity": 1, "form": "composed",
             "impl": {"module": "impl.py", "python": "shift"}},
        ],
    }
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


# ---------------- 元信息校验 ----------------

def test_validate_accepts_minimal_meta():
    m = plugin_meta.default_meta("ok-name", version="1.2.3", backends=["cpu"])
    # 规范要求至少声明一个算子（空包几乎肯定是漏写）
    m["ops"] = [{"name": "an_op", "form": "composed",
                 "impl": {"module": "impl.py", "python": "an_op"}}]
    out = plugin_meta.validate(m)
    assert out["name"] == "ok-name" and out["version"] == "1.2.3"


def test_validate_rejects_empty_ops():
    m = plugin_meta.default_meta("ok-name", version="1.0.0", backends=["cpu"])
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        plugin_meta.validate(m)
    assert "算子" in str(ei.value)


@pytest.mark.parametrize("bad,field", [
    ({"name": "Bad_Name"}, "name"),
    ({"name": "x"}, "name"),
    ({"version": "1.0"}, "version"),
    ({"version": "v1.2.3"}, "version"),
    ({"backends": []}, "backends"),
    ({"backends": ["quantum"]}, "backends"),
    ({"schema": 99}, "schema"),
])
def test_validate_rejects_bad_fields(bad, field):
    m = plugin_meta.default_meta("ok-name", version="1.0.0", backends=["cpu"])
    m.update(bad)
    with pytest.raises(plugin_meta.PluginMetaError):
        plugin_meta.validate(m)


def test_validate_requires_fallback_for_native_only():
    m = plugin_meta.default_meta("native-only", version="1.0.0", backends=["cuda"])
    m["ops"] = [{"name": "kk", "form": "native", "impl": {"backends": {"cuda": {"source": "k.cu"}}}}]
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        plugin_meta.validate(m)
    assert "回退" in str(ei.value)


def test_validate_allows_native_with_fallback():
    m = plugin_meta.default_meta("mixed", version="1.0.0", backends=["cpu", "cuda"])
    m["ops"] = [
        {"name": "kk", "form": "native", "impl": {"backends": {"cuda": {"source": "k.cu"}}}},
        {"name": "k_py", "form": "composed", "impl": {"module": "impl.py", "python": "double"}},
    ]
    out = plugin_meta.validate(m)
    assert len(out["ops"]) == 2


# ---------------- 一行安装 ----------------

def test_install_from_directory(tmp_path):
    root = tmp_path / "demo-ops"
    _write_bundle(str(root))
    meta = T.plugin.install(str(root), replace=True)
    assert meta["name"] == "demo-ops"
    assert hasattr(T, "demo"), "manifest 里声明了 namespace，应装出 T.demo"
    assert T.demo.double is not None
    x = T.from_numpy(np.array([[1.0, 2.0]], np.float32))
    assert np.allclose(T.demo.double(x).to_numpy(), [[2.0, 4.0]])


def test_install_from_zip(tmp_path):
    root = tmp_path / "demo-ops"
    _write_bundle(str(root), name="zip-ops", version="0.2.0")
    zpath = tmp_path / "zip-ops-0.2.0.tgop"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("manifest.json", "impl.py"):
            z.write(os.path.join(str(root), f), f)
    meta = T.plugin.install(str(zpath), replace=True)
    assert meta["name"] == "zip-ops" and meta["version"] == "0.2.0"
    assert hasattr(T, "demo")


def test_install_rejects_missing_manifest(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        T.plugin.install(str(empty))
    assert "manifest.json" in str(ei.value)


def test_install_rejects_missing_impl_file(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    manifest = {
        "schema": 1, "name": "broken", "version": "1.0.0", "api_version": "1.0",
        "backends": ["cpu"],
        "ops": [{"name": "ff", "form": "composed",
                 "impl": {"module": "nope.py", "python": "ff"}}],
    }
    with open(os.path.join(str(root), "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(plugin_meta.PluginMetaError) as ei:
        T.plugin.install(str(root))
    assert "找不到实现文件" in str(ei.value) or "实现文件" in str(ei.value)
