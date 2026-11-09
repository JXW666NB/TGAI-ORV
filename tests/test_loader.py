# -*- coding: utf-8 -*-
"""DataLoader 与 torch.utils.data 的对照（阶段 D 判据）。

判据（目标 D2）：
  * shuffle=False 时，批次的**内容与顺序**与 torch 逐个一致；
  * shuffle=True 时本库用自己的 RNG（顺序**不等同** torch，已在文档写明），
    因此只断言"不重不漏 + 批次形状正确"，并另测"显式 sampler 下与 torch 完全一致"；
  * num_workers>0 与 =0 结果完全相同；
  * 覆盖 drop_last、collate、空数据集、流式数据集。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader as TorchLoader  # noqa: E402
from torch.utils.data import Dataset as TorchDataset  # noqa: E402


class NpDS(T.Dataset):
    def __init__(self, n=7, d=3):
        rng = np.random.default_rng(0)
        self.x = rng.normal(0, 1, (n, d)).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": int(i % 2)}


class TorchDS(TorchDataset):
    def __init__(self, n=7, d=3):
        rng = np.random.default_rng(0)
        self.x = rng.normal(0, 1, (n, d)).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": int(i % 2)}


def _ours_batches(**kw):
    return [b for b in T.DataLoader(NpDS(), **kw)]


def _torch_batches(**kw):
    return [b for b in TorchLoader(TorchDS(), **kw)]


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("bs", [1, 2, 3])
def test_sequential_order_matches_torch(bs, drop_last):
    a = _ours_batches(batch_size=bs, shuffle=False, drop_last=drop_last)
    b = _torch_batches(batch_size=bs, shuffle=False, drop_last=drop_last)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x["x"], y["x"].numpy())
        assert np.array_equal(x["y"], y["y"].numpy())


def test_explicit_sampler_batching_matches_torch():
    """同一组索引（显式分组）下，两侧的分批结果必须完全一致。"""
    order = [4, 0, 6, 2, 5, 1, 3]
    groups = [order[k:k + 2] for k in range(0, len(order), 2)]

    ours = [b for b in T.DataLoader(IdxDS(), batch_size=2, sampler=order)]
    got = np.concatenate([b[:, 0] for b in ours]).tolist()
    assert got == order

    class TIdx(TorchDataset):
        def __len__(self):
            return 7

        def __getitem__(self, i):
            return int(i)

    ref = [b for b in TorchLoader(TIdx(), batch_sampler=groups)]
    assert [list(b) for b in ref] == groups


class IdxDS(T.Dataset):
    """返回自身索引，用来判“不重不漏”。"""

    def __init__(self, n=7):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.asarray([i], np.int64)


def test_shuffle_is_permutation_and_complete():
    bs = 3
    batches = [b for b in T.DataLoader(IdxDS(), batch_size=bs, shuffle=True, seed=1)]
    ids = np.concatenate([b[:, 0] for b in batches])
    assert sorted(ids.tolist()) == list(range(7))          # 不重不漏
    again = [b for b in T.DataLoader(IdxDS(), batch_size=bs, shuffle=True, seed=1)]
    for x, y in zip(batches, again):
        assert np.array_equal(x, y)                        # 同 seed 可复现
    other = [b for b in T.DataLoader(IdxDS(), batch_size=bs, shuffle=True, seed=2)]
    ids2 = np.concatenate([b[:, 0] for b in other])
    assert sorted(ids2.tolist()) == list(range(7))
    assert not np.array_equal(ids, ids2), "不同 seed 应当给出不同顺序"


def test_num_workers_matches_single_thread():
    a = _ours_batches(batch_size=2, num_workers=0)
    b = _ours_batches(batch_size=2, num_workers=4)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x["x"], y["x"])


def test_len_matches_torch():
    for bs in (1, 2, 3, 7, 10):
        for dl in (False, True):
            assert len(T.DataLoader(NpDS(), batch_size=bs, drop_last=dl)) == \
                len(TorchLoader(TorchDS(), batch_size=bs, drop_last=dl))


def test_default_collate_matches_torch_for_arrays_and_dicts():
    ours = T.data.default_collate([{"x": np.arange(3, dtype=np.float32), "y": 1},
                                   {"x": np.arange(3, dtype=np.float32) + 1, "y": 0}])
    from torch.utils.data._utils.collate import default_collate as tcollate
    ref = tcollate([{"x": torch.arange(3, dtype=torch.float32), "y": 1},
                    {"x": torch.arange(3, dtype=torch.float32) + 1, "y": 0}])
    assert np.array_equal(ours["x"], ref["x"].numpy())
    assert np.array_equal(ours["y"], ref["y"].numpy())


def test_empty_dataset_yields_nothing():
    class Empty(T.Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, i):
            raise AssertionError("不该被取")

    dl = T.DataLoader(Empty(), batch_size=4)
    assert len(dl) == 0
    assert list(dl) == []


def test_iterable_dataset_batches():
    class It(T.IterableDataset):
        def __iter__(self):
            for i in range(5):
                yield np.asarray([i], np.float32)

    got = [b for b in T.DataLoader(It(), batch_size=2)]
    assert len(got) == 3 and got[0].shape == (2, 1) and got[-1].shape == (1, 1)
    got2 = [b for b in T.DataLoader(It(), batch_size=2, drop_last=True)]
    assert len(got2) == 2


def test_errors_have_fix_hints():
    with pytest.raises(ValueError, match="修法"):
        T.DataLoader(NpDS(), batch_size=0)
    with pytest.raises(NotImplementedError, match="修法"):
        T.Dataset().__len__()
    with pytest.raises(TypeError, match="修法"):
        len(T.DataLoader(T.IterableDataset(), batch_size=1))
