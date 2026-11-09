# -*- coding: utf-8 -*-
"""数据加载：`Dataset` / `IterableDataset` / `Sampler` / `DataLoader`（对齐 torch 的常用面）。

设计取舍（如实说明）
* `shuffle=True` 时用**本库自己的** `np.random.Generator` 打乱，因此同一 seed 下的顺序
  **不等于** torch 的顺序（两边 RNG 算法不同）。想复现 torch 的顺序，就传 `sampler=`；
  只要求"打乱均匀、不重不漏"时用 `shuffle=True` 即可。
* `num_workers>0` 用**线程池**（不是进程）：受 GIL 限制，纯 Python 预处理不会真的并行；
  好处是零拷贝、无序列化、顺序易保证。真要并行，请把重活放在 numpy/CUDA 里。
* 顺序：即使 num_workers>0，也**按索引顺序**产出批次（与 torch 一致）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["Dataset", "IterableDataset", "Sampler", "SequentialSampler", "RandomSampler",
           "BatchSampler", "default_collate", "DataLoader"]


class Dataset(object):
    """映射式数据集：实现 `__len__` 与 `__getitem__`。"""

    def __len__(self):
        raise NotImplementedError("Dataset 子类必须实现 __len__。修法：返回样本数")

    def __getitem__(self, index):
        raise NotImplementedError("Dataset 子类必须实现 __getitem__。修法：按索引返回一个样本")


class IterableDataset(object):
    """流式数据集：实现 `__iter__`（可无限，配合 take/steps 用）。"""

    def __iter__(self):
        raise NotImplementedError("IterableDataset 子类必须实现 __iter__。修法：yield 样本")


class Sampler(object):
    """产出索引序列。"""

    def __iter__(self):
        raise NotImplementedError("Sampler 子类必须实现 __iter__。修法：yield 索引")

    def __len__(self):
        raise NotImplementedError("Sampler 子类必须实现 __len__。修法：返回索引个数")


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.n = len(data_source)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


class RandomSampler(Sampler):
    """有放回/无放回随机索引；用独立的 Generator，便于复现。"""

    def __init__(self, data_source, seed=None, generator=None, replacement=False):
        self.n = len(data_source)
        self.gen = generator if generator is not None else np.random.default_rng(seed)
        self.replacement = bool(replacement)

    def __iter__(self):
        if self.replacement:
            return iter(self.gen.integers(0, self.n, self.n).tolist())
        return iter(self.gen.permutation(self.n).tolist())

    def __len__(self):
        return self.n


class BatchSampler(Sampler):
    """把索引序列切成批次（drop_last 决定是否丢掉最后一小批）。"""

    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正，收到 %d。修法：传 >=1 的整数" % self.batch_size)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


def default_collate(batch):
    """把一批样本拼成批次（对标 torch.utils.data.default_collate 的常用类型）。

    支持：numpy 数组（堆叠）、数字/布尔（np 数组）、dict（按 key 递归）、list/tuple（转置后递归）。
    字符串等其它类型原样返回列表。
    """
    first = batch[0]
    if isinstance(first, dict):
        return {k: default_collate([b[k] for b in batch]) for k in first}
    if isinstance(first, (list, tuple)):
        return [default_collate(list(x)) for x in zip(*batch)]
    if isinstance(first, np.ndarray):
        return np.stack(batch, axis=0)
    if isinstance(first, (int, float, bool, np.number, np.bool_)):
        return np.asarray(batch)
    return list(batch)


class DataLoader(object):
    """批次迭代器。

    参数
      dataset     : Dataset（映射式）或 IterableDataset（流式）
      batch_size  : 每批样本数（流式时对"取到的样本"切批）
      shuffle     : 是否打乱（仅映射式；用本库自己的 RNG，见模块说明）
      sampler     : 显式索引序列（给了就忽略 shuffle）
      drop_last   : 是否丢掉最后不足一批的样本
      collate_fn  : 默认 `default_collate`
      num_workers : 线程数（>0 时并行取样本，顺序仍按索引）
      seed        : shuffle 的随机种子
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, drop_last=False,
                 collate_fn=None, num_workers=0, seed=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正，收到 %d。修法：传 >=1 的整数" % self.batch_size)
        self.shuffle = bool(shuffle)
        self.sampler = sampler
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn or default_collate
        self.num_workers = int(num_workers)
        self.seed = seed
        self._iterable = isinstance(dataset, IterableDataset)

    def _index_sampler(self):
        if self.sampler is not None:
            return self.sampler if hasattr(self.sampler, "__len__") else list(self.sampler)
        if self.shuffle:
            return RandomSampler(self.dataset, seed=self.seed)
        return SequentialSampler(self.dataset)

    def __len__(self):
        if self._iterable:
            raise TypeError("流式数据集没有长度。修法：改用 steps/计数控制训练轮数")
        return len(BatchSampler(self._index_sampler(), self.batch_size, self.drop_last))

    def _fetch(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        if self._iterable:
            batch = []
            for i, sample in enumerate(self.dataset):
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self.collate_fn(batch)
            return
        batches = BatchSampler(self._index_sampler(), self.batch_size, self.drop_last)
        if self.num_workers <= 0:
            for idxs in batches:
                yield self.collate_fn([self._fetch(i) for i in idxs])
            return
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for idxs in batches:
                samples = list(ex.map(self._fetch, idxs))     # map 保序
                yield self.collate_fn(samples)
