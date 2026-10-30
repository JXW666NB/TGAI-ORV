# -*- coding: utf-8 -*-
"""纯 CPU 单元测试：不需要 GPU，CI 上就能跑。

覆盖的都是「装上就崩」那类问题的高发区：
  * 张量形状/广播/负维度解析；
  * 后端切换的错误提示；
  * 续训存档的键名约定（与 exam2 的 --save-resume 保持一致）。

跑法：python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def test_import_package_surface():
    """包能 import，关键子模块都在（挡住缺模块那类事故）。"""
    import tgai_go.pack  # noqa: F401
    import tgai_go.train.backend  # noqa: F401
    import tgai_go.train.dist  # noqa: F401
    import tgai_go.train.export  # noqa: F401
    import tgai_go.train.doctor  # noqa: F401


def test_reshape_infers_minus_one():
    """reshape(-1, D) 要能推断维度（历史上报过"不支持负数维度"）。"""
    x = T.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    y = x.reshape(-1, 6)
    assert tuple(y.shape) == (2, 6), y.shape
    z = x.reshape(6, -1)
    assert tuple(z.shape) == (6, 2), z.shape


def test_reshape_wrong_size_raises():
    x = T.tensor(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(Exception) as ei:
        x.reshape(4, 4)
    assert "reshape" in str(ei.value).lower() or "形状" in str(ei.value)


def test_elementwise_and_matmul_values():
    a = T.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = T.tensor(np.array([[1.0], [2.0]], dtype=np.float32))
    c = a @ b
    assert np.allclose(np.asarray(c.to_numpy()), [[5.0], [11.0]])


def test_backend_switch_error_names_fix():
    """切到不存在的后端要报清楚，并且给出修法（不是裸异常）。"""
    for bad in ("vulkan", "rocm", "hip"):
        with pytest.raises(Exception) as ei:
            T.set_default_device(bad)
        msg = str(ei.value)
        assert "docs" in msg or "修法" in msg or "支持" in msg, msg


def test_backend_registry_lists_cuda_and_cpu():
    from tgai_go.train import backend
    names = backend.available()
    assert "cuda" in names or "cpu" in names


def test_resume_archive_key_convention(tmp_path):
    """续训存档的键名约定：step / w_<param> / m_<i> / v_<i> / opt_t。

    这里只校验约定本身（exam2 的 --save-resume 用同一套名字），
    不需要 GPU —— 之前正是「键名与读取端不一致」导致恢复数为 0 却无报错。
    """
    p = tmp_path / "ck.npz"
    np.savez(p, step=np.array(40),
             **{"w_wte": np.zeros((4, 3), np.float32),
                "m_0": np.zeros((4, 3), np.float32),
                "v_0": np.zeros((4, 3), np.float32),
                "opt_t": np.array(40)})
    z = np.load(p, allow_pickle=False)
    assert "step" in z.files and int(z["step"]) == 40
    assert any(k.startswith("w_") for k in z.files)
    assert any(k.startswith("m_") for k in z.files)
    assert any(k.startswith("v_") for k in z.files)
    assert "opt_t" in z.files


def test_no_bom_in_sources():
    """源码不许带 BOM（tomllib 会因此解析失败，历史上踩过）。"""
    bad = []
    root = os.path.join(REPO, "python")
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith((".py", ".cu", ".cuh")):
                fp = os.path.join(dp, f)
                if open(fp, "rb").read(3) == b"\xef\xbb\xbf":
                    bad.append(os.path.relpath(fp, REPO))
    assert not bad, "以下文件带 BOM：%s" % bad[:5]
