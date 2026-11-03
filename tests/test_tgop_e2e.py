# -*- coding: utf-8 -*-
"""`.tgop` 插件包端到端：打包 → 加载 → **走 CPU C 后端**跑前向+反向。

这条是插件故事的关键闭环：一个 `.tgop` 里同时放 CUDA/C 源码与 python 兜底，
引擎按本机能力挑一个。CI 上（linux + gcc）会真的编译 C 并跑；没有编译器时跳过。
"""
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import ops_bundle, ops_cpu  # noqa: E402

C_SRC = r"""
void tgop_dbl_fwd(const float* a, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * 2.0f;
}
void tgop_dbl_bwd(const float* g, const float* a, float* da, int n) {
    for (int i = 0; i < n; ++i) da[i] = g[i] * 2.0f;
}
"""

CU_SRC = r"""
extern "C" __global__ void tgop_dbl_fwd_cu(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * 2.0f;
}
"""

PY_IMPL = "def dbl(x):\n    return x * 2.0\n"


def _build(tmp_path, with_c=True):
    d = tmp_path / "plug"
    d.mkdir()
    manifest = {
        "schema": 1, "name": "tgop_e2e_dbl", "arity": 1,
        "backends": {
            "cuda": {"source": "dbl.cu", "fwd": "tgop_dbl_fwd_cu", "bwd": "tgop_dbl_fwd_cu"},
            "python": {"module": "dbl_py.py", "callable": "dbl"},
        },
    }
    files = {"dbl.cu": CU_SRC, "dbl_py.py": PY_IMPL}
    if with_c:
        manifest["backends"]["cpu"] = {"c": "dbl.c", "fwd": "tgop_dbl_fwd",
                                       "bwd": "tgop_dbl_bwd"}
        files["dbl.c"] = C_SRC
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, body in files.items():
        (d / name).write_text(body, encoding="utf-8")
    out = str(tmp_path / "dbl.tgop")
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(REPO, "tools", "tgop_pack.py"),
                        str(d), "-o", out], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    return out


def _has_cc():
    cc, _ = ops_cpu.find_compiler()
    return cc is not None


def test_pack_reports_all_backends(tmp_path):
    p = _build(tmp_path)
    info = ops_bundle.inspect(p)
    assert info["name"] == "tgop_e2e_dbl"
    assert set(info["backends"]) == {"cuda", "cpu", "python"}


def test_pack_rejects_missing_referenced_file(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "schema": 1, "name": "x", "arity": 1,
        "backends": {"python": {"module": "missing.py", "callable": "f"}}}), encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(REPO, "tools", "tgop_pack.py"), str(d)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode != 0 and "不存在" in ((r.stdout or "") + (r.stderr or ""))


def test_cpu_backend_runs_forward_and_backward(tmp_path):
    if not _has_cc():
        pytest.skip("本机没有 C 编译器")
    import tgai_go.train as T
    p = _build(tmp_path)
    name, chosen = ops_bundle.load(p, verbose=False)
    assert name == "tgop_e2e_dbl"
    if chosen != "cpu":
        pytest.skip("本机选中了 %s 后端（有 CUDA/GPU），这条专测 CPU C 后端" % chosen)
    x = np.random.default_rng(0).normal(size=(64,)).astype(np.float32)
    t = T.tensor(x.copy(), requires_grad=True)
    y = getattr(T.functional, "tgop_e2e_dbl")(t)
    assert np.allclose(np.asarray(y.to_numpy()), x * 2.0)
    (y * T.tensor(np.ones(64, np.float32))).sum().backward()
    assert np.allclose(np.asarray(t.grad), np.full(64, 2.0, np.float32))
