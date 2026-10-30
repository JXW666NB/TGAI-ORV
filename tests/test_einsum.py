# -*- coding: utf-8 -*-
"""einsum 的 PyTorch 对照测试（受限形式）。"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import nn_ops as O

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


CASES = [
    ("ij,jk->ik", [(4, 5), (5, 3)]),
    ("bij,bjk->bik", [(2, 4, 5), (2, 5, 3)]),
    ("bhid,bhjd->bhij", [(2, 2, 4, 8), (2, 2, 6, 8)]),
    ("i,i->", [(5,), (5,)]),
    ("ij->ji", [(3, 4)]),
    ("ij->", [(3, 4)]),
]


@pytest.mark.parametrize("sub,shapes", CASES)
def test_einsum_matches_torch(sub, shapes):
    rng = np.random.default_rng(11)
    arrs = [rng.normal(size=s).astype(np.float32) for s in shapes]
    ours = [T.tensor(a.copy(), requires_grad=True) for a in arrs]
    out = O.einsum(sub, *ours)
    out.sum().backward()

    ref = [torch.from_numpy(a.copy()).requires_grad_(True) for a in arrs]
    tout = torch.einsum(sub, *ref)
    tout.sum().backward()

    assert np.allclose(out.to_numpy(), tout.detach().numpy(), rtol=1e-5, atol=1e-6), sub
    for o, r in zip(ours, ref):
        assert np.allclose(o.grad, r.grad.numpy(), rtol=1e-5, atol=1e-6), sub


def test_einsum_repeated_index_rejected():
    x = T.tensor(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError) as ei:
        O.einsum("ii->i", x)
    assert "重复下标" in str(ei.value) and "修法" in str(ei.value)


def test_einsum_too_many_operands_rejected():
    a = T.tensor(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError) as ei:
        O.einsum("ij,jk,kl->il", a, a, a)
    assert "1~2 个操作数" in str(ei.value)


def test_einsum_rank_mismatch_rejected():
    a = T.tensor(np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError) as ei:
        O.einsum("ijk->k", a)
    assert "维度" in str(ei.value) or "字母" in str(ei.value)


def test_einsum_no_arrow_infers_output():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    b = rng.normal(size=(3,)).astype(np.float32)
    out = O.einsum("ij,j", T.tensor(a), T.tensor(b)).to_numpy()
    want = torch.einsum("ij,j", torch.from_numpy(a), torch.from_numpy(b)).numpy()
    assert np.allclose(out, want, rtol=1e-6, atol=1e-6)
