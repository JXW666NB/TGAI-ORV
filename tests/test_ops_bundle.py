# -*- coding: utf-8 -*-
"""`.tgop` 通用插件包：打包 → 检查 → 按后端加载（纯 CPU 可测）。"""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import ops_bundle  # noqa: E402

MANIFEST = {
    "schema": 1, "name": "tgop_test_relu", "arity": 1,
    "backends": {
        "cuda": {"source": "relu.cu", "fwd": "relu_fwd", "bwd": "relu_bwd"},
        "hip": {"source": "relu.cu"},
        "vulkan": {"spirv": "relu.spv"},
        "python": {"module": "relu_py.py", "callable": "relu"},
    },
}
PY_IMPL = "def relu(x):\n    return x * ((x > 0.0).astype('float32'))\n"
CU_SRC = 'extern "C" __global__ void relu_fwd(const float* a, float* o, int n) {}\n'


def _make(tmp_path, manifest=None, files=None):
    d = tmp_path / "plug"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(manifest or MANIFEST), encoding="utf-8")
    for name, body in (files or {"relu.cu": CU_SRC, "relu_py.py": PY_IMPL}).items():
        (d / name).write_text(body, encoding="utf-8")
    out = str(tmp_path / "x.tgop")
    with zipfile.ZipFile(out, "w") as zf:
        zf.write(str(d / "manifest.json"), "manifest.json")
        for name in (files or {"relu.cu": CU_SRC, "relu_py.py": PY_IMPL}):
            zf.write(str(d / name), name)
    return out


def test_inspect_lists_backends(tmp_path):
    p = _make(tmp_path)
    info = ops_bundle.inspect(p)
    assert info["name"] == "tgop_test_relu"
    assert "python" in info["backends"] and "vulkan" in info["backends"]


def test_load_picks_python_on_cpu(tmp_path):
    p = _make(tmp_path)
    name, chosen = ops_bundle.load(p, verbose=False)
    assert name == "tgop_test_relu"
    if "cuda" in ops_bundle.os.environ.get("TGAI_TEST_BACKENDS", "python"):
        pass
    assert chosen in ("cuda", "python")          # 有卡选 cuda，无卡选 python


def test_manifest_without_python_fallback_rejected(tmp_path):
    m = dict(MANIFEST)
    m["backends"] = {"vulkan": {"spirv": "relu.spv"}}
    p = _make(tmp_path, manifest=m)
    with pytest.raises(ops_bundle.BundleError) as ei:
        ops_bundle.load(p, verbose=False)
    assert "python" in str(ei.value) and "修法" in str(ei.value)


def test_missing_manifest_message(tmp_path):
    out = str(tmp_path / "empty.tgop")
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(ops_bundle.BundleError) as ei:
        ops_bundle.inspect(out)
    assert "manifest.json" in str(ei.value) and "修法" in str(ei.value)


def test_bad_schema_rejected(tmp_path):
    m = dict(MANIFEST, schema=99)
    p = _make(tmp_path, manifest=m)
    with pytest.raises(ops_bundle.BundleError) as ei:
        ops_bundle.inspect(p)
    assert "schema" in str(ei.value)


def test_bom_tolerated(tmp_path):
    d = tmp_path / "bomplug"
    d.mkdir()
    bom_manifest = json.loads(json.dumps(MANIFEST))
    bom_manifest["name"] = "tgop_bom_test"     # 避免与其它用例重名
    (d / "manifest.json").write_text("\ufeff" + json.dumps(bom_manifest), encoding="utf-8")
    (d / "relu_py.py").write_text("\ufeff" + PY_IMPL, encoding="utf-8")
    out = str(tmp_path / "bom.tgop")
    with zipfile.ZipFile(out, "w") as zf:
        zf.write(str(d / "manifest.json"), "manifest.json")
        zf.write(str(d / "relu_py.py"), "relu_py.py")
    # 目标：不因 BOM 报错（后端选择无关紧要）
    name, chosen = ops_bundle.load(out, verbose=False)
    assert name == "tgop_bom_test" and chosen in ("cuda", "python")


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        ops_bundle.load(str(tmp_path / "nope.tgop"))
    assert "修法" in str(ei.value)
