# -*- coding: utf-8 -*-
"""V3 **真实预训练**入口：真语料 + LR 调度 + 梯度裁剪 + checkpoint / 续训。

`tgai_go.train.v3_train_cli` 只能跑"随机 ids 冒烟"（无数据、无调度、无存档）。
本文件补上**真正能拿去预训练**的那一层，模型与前向完全复用 `v3_train_cli`
（`build` / `params_of` / `layer_kinds`），**不复制第二份模型实现**：

    python -m tgai_go.train.v3_pretrain \\
        --data D:\\TGAI-V3-1.5B\\01_corpus\\packed512\\s512 \\
        --d 256 --layers 4 --E 2 --seq 256 --vocab 96000 \\
        --steps 60 --warmup-steps 5 --out D:\\orv_runs\\v3p_small

## 数据：照 `data_v3.py` / `train_v3.py` 的语义，不自己发明

真 V3 训练用的打包产物是 `part-XXXXX.npz`，每个分片三个数组（实测
`D:\\TGAI-V3-1.5B\\01_corpus\\packed512\\s512\\part-00000.npz`：`(4001, 512)`）：

| 键 | dtype | 语义 |
| --- | --- | --- |
| `input_ids` | int32 `(N, T)` | 打包后的 token 序列（多篇文档首尾相接 + `<pad>` 补齐）|
| `labels` | int32 `(N, T)` | `labels[t] == input_ids[t]`（实测 98.6~99.2%）；文档首 token 与 pad 位 = `-100` |
| `seg_id` | int16 `(N, T)` | 每个 token 的文档号，`<0` = padding |

三条关键语义（都是"写错了不报错、loss 照样降"的那种）：

1. **shift 在模型里**：`loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)`
   （`model_v3.py:1508-1515`）。`labels[t] == ids[t]` 不是"抄输入"，因为错了一位。
2. **块对角掩码**：`seg_id` → `mask[i,j] = (j<=i) & (seg[j]==seg[i]) & (seg[i]>=0)`，
   跨文档**互不可见**（`data_v3.make_seg_mask` / `v3_masks.make_seg_mask`）。
   只做因果不做块对角会让文档互相泄漏，**训练不报错**（参考实现记录的 P0）。
3. **CSA 也要 seg**：压缩池化按文档边界切 chunk（`v3_csa.make_plan(..., seg=...)`），
   否则会把两篇文档的 token 池进同一个 chunk。

`input_ids >= vocab` 会静默算错 embedding，所以**每个窗口**都做一次 `max(id) < vocab`
校验（成本可忽略）；开训前还会先扫第一个分片的头部，早失败。

## 数据位置与续训

分片级 + 分片内双重打乱（照 `train_v3.py::PackedDataset.order` 的做法），
种子只由 `(seed, epoch, 物理分片号)` 决定 ⇒ **只存 4 个整数**（epoch / rank / offset /
consumed）就能精确回到断点。为什么不走"全局 300 万位置 permutation"：那样每步跳到随机
分片，读盘是随机的（`train_v3.py` 实测 0.985 s/步、占总时间 76%）。

## checkpoint：为什么不用 `ckpt.save_checkpoint`

`tgai_go.train.ckpt` 的 `save_checkpoint` / `load_checkpoint` 是**为 V2.7 训练器写死的**：

* `ckpt.iter_named_tensors(tr, st, names)` 按 `.TG` 张量名分派，依赖 `tr.layers[i]["attn"]["q"]`
  这种"分头后的模块列表"结构 —— V3 的参数是**扁平张量 dict**（`build()` 的产物），没有这层；
* `save_checkpoint` 开头 `read_tok_blob(src_tg)` **必须有带分词段的源 `.TG`**；V3 从零预训练
  时没有这个文件；
* 它把权重写 **TG4 / bf16**（`write_tg4_stream(..., weight_dtype=DT_BF16)`）⇒ 续训会损失精度，
  无法做到"续训后 loss 与不中断一致"。

所以本文件**复用** `ckpt` 里与 V2.7 无关的部分（`checkpoint_dir` / `list_checkpoints` /
`rotate` / `CKPT_META` 约定），权重与优化器状态按 **fp32 逐张量 `.npy`** 落盘：
`to_numpy()` → `np.save` → `copy_h2d_into(buf)` 原样写回**已有设备缓冲**（不新分配，
续训峰值显存与正常训练相同）。参数名逐个校验，cfg / seed / LR 计划不匹配直接报错。

## 续训一致性（本文件的验收点）

`--steps` 是 LR 余弦的 **horizon**（`T_max = steps - warmup`），`--stop-at` 才是"跑到第几步停"。
所以"制造中断"的写法是 `--steps 60 --stop-at 30`，续训仍用 `--steps 60` ——
两条曲线共用同一套 LR 计划，差异只可能来自"状态没恢复干净"。
checkpoint 里存了 `step / 调度器状态 / AdamW.t / m / v / 数据位置`，**缺一项都会漂移**。

## 与 `v3_train_cli` 的关系

* 不修改 `v3_train_cli`；`build` / `params_of` / `layer_kinds` 直接 import 复用；
* 前向多了两样东西：`seg`（块对角掩码 + CSA 分段）与 **shift 后的 ignore_index CE**；
* 少了一样东西（**已知差距**）：参考实现的 MoE 辅助损失（balance + router z-loss）
  与 QK-Clip 未实现 —— 本文件只做"主干 + 数据 + 调度 + 存档"，
  长跑前需要单独补 aux loss，否则 MoE 可能塌缩。

## 已知限制

* **只支持单卡**（ORV 现状）；`--batch > 1` 只是把 B 条序列并成一批，不是数据并行。
* **`.bin` 必须显式 `--bin-dtype`**（uint16/uint32）：猜错 dtype 不会报错，只会把 token 读成垃圾。
* `--seq` 小于分片宽度时按连续窗口切（512 → 2×256）；窗口首 token 丢了上文，
  属于"裁窗"而不是"重新打包"，跨文档隔离仍由 `seg_id` 保证。
* 续训必须有 `--out`（checkpoint 写到 `--out/step_%06d`）。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import struct
import subprocess
import threading
import time
import zipfile

import numpy as np

from tgai_go import train as T
from tgai_go.train import ckpt
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_masks as MK
from tgai_go.train import v3_train_cli as CLI

__all__ = ["PackedCorpus", "build_arg_parser", "main",
           "param_slots", "save_params", "load_params",
           "save_optimizer", "load_optimizer", "forward_loss",
           "save_run", "resolve_resume"]

SCHEMA = 1                     # checkpoint 格式版本（改字段必须 +1）
NPY_MAGIC = b"\x93NUMPY"
IGNORE_INDEX = -100            # 与 data_v3 / model_v3 的忽略值一致


# ═════════════════════════════════════════════════════════════════════════
# 小工具
# ═════════════════════════════════════════════════════════════════════════
def _now():
    """计时统一用 perf_counter（Windows 上 time.time() 粒度 15.6 ms，不可用）。"""
    return time.perf_counter()


def _scalar(t):
    """0 维张量 → float（CUDA 上 `to_numpy()` 返回 0 维数组）。"""
    v = t.to_numpy() if hasattr(t, "to_numpy") else t
    if isinstance(v, (int, float)):
        return float(v)
    return float(np.asarray(v).reshape(-1)[0])


def _gib(nbytes):
    return float(nbytes) / float(1 << 30)


def _is_finite(x):
    """NaN / Inf 判定（`x` 可以是 float、numpy 标量或 None）。"""
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _jsonable(obj):
    """把 cfg 变成可 json 的 dict（`csa_cfg` / `_csa_mod` 是对象，落盘前剔除）。"""
    out = {}
    for k, v in dict(obj).items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
    return out


def _npy_header(f):
    """读 `.npy` 头，返回 `(shape, dtype)`；只读约 1 KB，不加载数据。

    自己解析而不用 `np.lib.format.read_array_header_*` / `_read_array_header`：
    后者在 numpy 2.x 被改名/移除（实测远端 numpy 2.4.6 取不到 `_read_array_header`），
    而 `.npy` 头格式本身是**稳定**的（magic + version + 2 或 4 字节头长 + 字典 repr）。
    """
    magic = f.read(6)
    if magic != NPY_MAGIC:
        raise ValueError("不是 .npy 数据（magic=%r）。修法：确认文件没被截断/改后缀" % (magic,))
    ver = f.read(2)
    if len(ver) != 2:
        raise ValueError("npy 头被截断（version 缺失）。修法：重新导出该分片")
    major = ver[0]
    if major == 1:
        raw = f.read(2)
        if len(raw) != 2:
            raise ValueError("npy v1 头被截断。修法：重新导出该分片")
        hlen = struct.unpack("<H", raw)[0]
    elif major in (2, 3):
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError("npy v%d 头被截断。修法：重新导出该分片" % (major,))
        hlen = struct.unpack("<I", raw)[0]
    else:
        raise ValueError("不支持的 npy 版本 %r。修法：用 numpy<3 重新导出" % (ver,))
    text = f.read(hlen).decode("latin1")
    descr = ast.literal_eval(text)
    return tuple(int(x) for x in descr["shape"]), np.dtype(descr["descr"])


def _npz_members(path):
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def _npz_array_header(path, member):
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as f:
            return _npy_header(f)


_BIN_DTYPES = {"uint16": np.uint16, "int16": np.int16,
               "uint32": np.uint32, "int32": np.int32, "int64": np.int64}


# ═════════════════════════════════════════════════════════════════════════
# 数据层：分片探测 + 双重打乱 + 断点位置
# ═════════════════════════════════════════════════════════════════════════
class ShardInfo(object):
    """一个分片的元信息（**只读头部**，不加载数据）。"""

    __slots__ = ("path", "fmt", "n_rows", "width", "n_tokens", "n_windows", "has_labels")

    def __init__(self, path, fmt, n_rows=None, width=None, n_tokens=None,
                 n_windows=None, has_labels=False):
        self.path = path
        self.fmt = fmt
        self.n_rows = n_rows
        self.width = width
        self.n_tokens = n_tokens
        self.n_windows = int(n_windows or 0)
        self.has_labels = bool(has_labels)

    def __repr__(self):
        return ("ShardInfo(%s, fmt=%s, rows=%s, width=%s, tokens=%s, windows=%d)"
                % (os.path.basename(self.path), self.fmt, self.n_rows, self.width,
                   self.n_tokens, self.n_windows))


def expand_data_paths(specs):
    """把 `--data` 展开成**分片文件列表**（目录 / 通配符 / 单文件都接受）。

    目录只取**顶层**分片（与 `train_v3.PackedDataset` 一致）；要递归请写
    `dir/**/*.npz` 这种显式通配符（`**` 走 `glob(recursive=True)`）。
    """
    out, seen = [], set()
    for spec in specs or []:
        spec = os.path.expanduser(str(spec))
        got = []
        if os.path.isdir(spec):
            for ext in ("*.npz", "*.npy", "*.bin"):
                got.extend(glob.glob(os.path.join(spec, ext)))
        elif os.path.isfile(spec):
            got.append(spec)
        else:
            got.extend(glob.glob(spec, recursive=True))
        got = [p for p in got if os.path.isfile(p)
               and os.path.splitext(p)[1].lower() in (".npz", ".npy", ".bin")]
        got.sort()
        for p in got:
            key = os.path.normcase(os.path.abspath(p))
            if key in seen:
                continue
            seen.add(key)
            out.append(os.path.abspath(p))
    if not out:
        raise SystemExit(
            "--data 没有匹配到任何分片（.npz / .npy / .bin）：%r\n"
            "修法：① 指到真正的分片目录（真 V3 语料是 "
            "D:\\TGAI-V3-1.5B\\01_corpus\\packed512\\s512 这类 part-*.npz 目录）；"
            "② 或用通配符 D:\\path\\**\\*.npz。" % (list(specs or []),))
    return out


def probe_shard(path, seq, bin_dtype=None):
    """探测一个分片的格式与规模（**只读头部**，20~40 MB 的分片不会被读进来）。"""
    low = path.lower()
    ext = os.path.splitext(low)[1]
    seq = int(seq)
    if ext == ".npz":
        members = _npz_members(path)
        if "input_ids.npy" not in members:
            raise SystemExit(
                "%s 里没有 input_ids.npy（成员：%s）。修法：这不是 data_v3.py 产出的打包分片；"
                "要么换目录，要么用 --data 指到正确的分片" % (path, members[:6]))
        shape, _dtype = _npz_array_header(path, "input_ids.npy")
        if len(shape) != 2:
            raise SystemExit("input_ids 形状应为 (N,T)，实际 %s：%s。修法：重新打包"
                             % (shape, path))
        n_rows, width = int(shape[0]), int(shape[1])
        return ShardInfo(path, "npz", n_rows=n_rows, width=width,
                         n_tokens=n_rows * width, n_windows=n_rows * (width // seq),
                         has_labels=("labels.npy" in members))
    if ext == ".npy":
        with open(path, "rb") as f:
            shape, _dtype = _npy_header(f)
        if len(shape) == 2:
            n_rows, width = int(shape[0]), int(shape[1])
            return ShardInfo(path, "npy2d", n_rows=n_rows, width=width,
                             n_tokens=n_rows * width, n_windows=n_rows * (width // seq))
        if len(shape) == 1:
            n_tok = int(shape[0])
            return ShardInfo(path, "stream", n_tokens=n_tok, n_windows=n_tok // seq)
        raise SystemExit("npy 张量维度应为 1 或 2，实际 %s：%s。修法：导出成 (N,T) 或扁平流"
                         % (shape, path))
    # .bin
    if not bin_dtype:
        raise SystemExit(
            "%s 是裸 .bin，但没给 --bin-dtype。修法：显式给 --bin-dtype uint16|uint32"
            "（**不能猜**：猜错 dtype 不会报错，只会把 token 读成垃圾）。"
            "真 V3 语料都是 .npz，不需要这个参数。" % (path,))
    dt = _BIN_DTYPES.get(str(bin_dtype))
    if dt is None:
        raise SystemExit("--bin-dtype 只支持 %s，收到 %r" % (sorted(_BIN_DTYPES), bin_dtype))
    n_tok = os.path.getsize(path) // int(np.dtype(dt).itemsize)
    return ShardInfo(path, "bin", n_tokens=n_tok, n_windows=n_tok // seq)


class PackedCorpus(object):
    """打包分片语料：分片级 + 分片内双重打乱，位置可精确续训。

    状态只有 4 个数：`epoch` / `rank`（第几个分片，按打乱后的序）/ `offset`（分片内第几个
    窗口）/ `consumed`（累计取过多少窗口）。`take()` 一次推进窗口。

    `seq` 不能整除分片宽度时，每行尾部余数被丢弃（启动时打印 WARN，不静默）。
    """

    CACHE_MAX = 2                       # 常驻几个分片（每个 20~40 MB）
    ORDER_CACHE = 16                    # 分片内顺序表缓存条数

    def __init__(self, paths, seq, seed=42, bin_dtype=None, vocab=None, verbose=True):
        self.seq = int(seq)
        self.seed = int(seed)
        self.bin_dtype = bin_dtype
        self.vocab = None if vocab is None else int(vocab)
        self.infos = []
        t0 = _now()
        for p in paths:
            info = probe_shard(p, self.seq, bin_dtype=bin_dtype)
            if info.n_windows <= 0:
                if verbose:
                    print("[data] SKIP %s (windows=0, width=%s < seq=%d)"
                          % (os.path.basename(p), info.width, self.seq), flush=True)
                continue
            if info.fmt in ("npz", "npy2d") and (info.width % self.seq) and verbose:
                print("[data] WARN %s width=%d not divisible by seq=%d; "
                      "tail %d tokens/row dropped"
                      % (os.path.basename(p), info.width, self.seq,
                         info.width % self.seq), flush=True)
            self.infos.append(info)
        if not self.infos:
            raise SystemExit("所有分片都切不出 seq=%d 的窗口。修法：检查 --seq 与分片宽度"
                             "（真 V3 分片是 512/1024/2048）" % (self.seq,))
        self.scan_sec = _now() - t0
        self.n_shards = len(self.infos)
        self._wins = [i.n_windows for i in self.infos]
        self.total_windows = int(sum(self._wins))
        self.total_tokens = int(sum(i.n_tokens for i in self.infos))
        self.n_no_labels = sum(1 for i in self.infos if not i.has_labels
                               and i.fmt == "npz")
        self._cache = {}
        self._orders = {}
        self._shard_orders = {}
        self._max_id_seen = -1
        self.epoch = 0
        self.rank = 0
        self.offset = 0
        self.consumed = 0

    # ---- 顺序（只由 seed / epoch / 物理分片号 决定 ⇒ 可精确复现）----
    def shard_order(self, epoch):
        got = self._shard_orders.get(int(epoch))
        if got is None:
            rng = np.random.default_rng(self.seed + int(epoch) * 100003)
            got = rng.permutation(self.n_shards).astype(np.int64)
            if len(self._shard_orders) > 4:
                self._shard_orders.clear()
            self._shard_orders[int(epoch)] = got
        return got

    def window_order(self, epoch, rank):
        key = (int(epoch), int(rank))
        got = self._orders.get(key)
        if got is None:
            si = int(self.shard_order(epoch)[rank])
            n = self._wins[si]
            # 与 train_v3.py 同族：seed + epoch*100003，再掺物理分片号
            mix = (self.seed * 1000003 + int(epoch) * 100003 + si * 7919) % (2 ** 62)
            got = np.random.default_rng(int(mix)).permutation(n).astype(np.int64)
            if len(self._orders) > self.ORDER_CACHE:
                self._orders.clear()
            self._orders[key] = got
        return got

    # ---- 取数据 ----
    def _load(self, si):
        got = self._cache.get(si)
        if got is not None:
            return got
        info = self.infos[si]
        if info.fmt == "npz":
            with np.load(info.path) as z:
                ids = np.asarray(z["input_ids"])
                lab = np.asarray(z["labels"]) if "labels" in z.files else None
                seg = np.asarray(z["seg_id"]) if "seg_id" in z.files else None
        elif info.fmt == "npy2d":
            ids = np.load(info.path, mmap_mode="r")
            lab, seg = None, None
        elif info.fmt == "bin":
            ids = np.memmap(info.path, dtype=_BIN_DTYPES[str(self.bin_dtype)], mode="r")
            lab, seg = None, None
        else:                                   # pragma: no cover
            raise SystemExit("未知分片格式 %r" % (info.fmt,))
        got = (ids, lab, seg)
        if len(self._cache) >= self.CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[si] = got
        return got

    def _window(self, si, k):
        info = self.infos[si]
        ids, lab, seg = self._load(si)
        Tlen = self.seq
        if info.fmt in ("npz", "npy2d"):
            n_win = info.width // Tlen
            row = int(k) // n_win
            off = (int(k) % n_win) * Tlen
            sl = slice(off, off + Tlen)
            x = np.asarray(ids[row, sl], np.int64)
            y = (np.asarray(lab[row, sl], np.int64) if lab is not None else x.copy())
            s = (np.asarray(seg[row, sl], np.int64) if seg is not None
                 else np.zeros(Tlen, np.int64))
        else:                                   # 扁平 token 流
            off = int(k) * Tlen
            x = np.asarray(ids[off:off + Tlen], np.int64)
            y = x.copy()
            s = np.zeros(Tlen, np.int64)
        if x.size != Tlen:
            raise SystemExit("%s 第 %d 个窗口不足 %d 个 token。修法：分片被截断，重新导出"
                             % (os.path.basename(info.path), k, Tlen))
        lo, hi = int(x.min()), int(x.max())
        if hi > self._max_id_seen:
            self._max_id_seen = hi
        if lo < 0:
            raise SystemExit("input_ids 出现负值（min=%d）：%s。修法：分片被写坏，重新打包"
                             % (lo, info.path))
        if self.vocab is not None and hi >= self.vocab:
            raise SystemExit(
                "token id %d >= --vocab %d（%s）。修法：词表必须与分片一致"
                "（真 V3 是 vocab=96000，分片 id 在 0..95999）。"
                "传错不会影响任何 shape 检查，只会把 embedding 查错，所以这里直接拦下。"
                % (hi, self.vocab, os.path.basename(info.path)))
        return x, y, s

    def _next(self):
        while True:
            if self.rank >= self.n_shards:
                self.epoch += 1
                self.rank = 0
                self.offset = 0
            si = int(self.shard_order(self.epoch)[self.rank])
            n = self._wins[si]
            if n <= 0 or self.offset >= n:
                self.rank += 1
                self.offset = 0
                continue
            k = int(self.window_order(self.epoch, self.rank)[self.offset])
            self.offset += 1
            self.consumed += 1
            return self._window(si, k)

    def take(self, batch=1):
        """取 `batch` 个窗口 → `(ids (B,T) int64, labels (B,T) int64, seg (B,T) int64)`。"""
        xs, ys, ss = [], [], []
        for _ in range(int(batch)):
            x, y, s = self._next()
            xs.append(x)
            ys.append(y)
            ss.append(s)
        return np.stack(xs), np.stack(ys), np.stack(ss)

    def check_vocab(self, n_rows=512, verbose=True):
        """开训前先扫第一个分片的头部：max(token id) 必须 < vocab（早失败）。"""
        ids, _lab, _seg = self._load(0)
        info = self.infos[0]
        if info.fmt in ("npz", "npy2d"):
            n = min(int(info.n_rows), int(n_rows))
            block = np.asarray(ids[:n])
        else:
            n = min(int(info.n_tokens), int(n_rows) * self.seq)
            block = np.asarray(ids[:n])
        hi = int(block.max()) if block.size else -1
        lo = int(block.min()) if block.size else 0
        if verbose:
            print("[data] vocab check: first %d rows of %s -> id range [%d, %d]"
                  % (n, os.path.basename(info.path), lo, hi), flush=True)
        if lo < 0:
            raise SystemExit("input_ids 出现负值（min=%d）：%s。修法：分片被写坏，重新打包"
                             % (lo, info.path))
        if self.vocab is not None and hi >= self.vocab:
            raise SystemExit(
                "token id %d >= --vocab %d（%s）。修法：词表必须与分片一致"
                "（真 V3 是 vocab=96000，分片 id 在 0..95999）"
                % (hi, self.vocab, os.path.basename(info.path)))
        return hi

    # ---- 断点 ----
    def state_dict(self):
        return {"epoch": int(self.epoch), "rank": int(self.rank),
                "offset": int(self.offset), "consumed": int(self.consumed),
                "n_shards": int(self.n_shards), "seq": int(self.seq),
                "seed": int(self.seed), "windows_per_epoch": int(self.total_windows)}

    def load_state_dict(self, st):
        if int(st.get("seq", -1)) != self.seq:
            raise SystemExit("数据位置与当前 --seq 不一致：checkpoint seq=%s，本次 seq=%d。"
                             "修法：续训要保持 --seq 不变（改 seq 等于换了数据切法）"
                             % (st.get("seq"), self.seq))
        if int(st.get("n_shards", -1)) != self.n_shards:
            raise SystemExit("数据分片数不一致：checkpoint %s，本次 %d。"
                             "修法：续训要用同一批分片（--data 指回原目录）"
                             % (st.get("n_shards"), self.n_shards))
        self.epoch = int(st["epoch"])
        self.rank = int(st["rank"])
        self.offset = int(st["offset"])
        self.consumed = int(st["consumed"])
        return self

    def stats(self):
        return {"shards": self.n_shards, "windows": self.total_windows,
                "tokens": self.total_tokens, "seq": self.seq,
                "tokens_per_epoch": self.total_windows * self.seq,
                "shards_without_labels": int(self.n_no_labels),
                "max_id_seen": int(self._max_id_seen),
                "scan_sec": round(self.scan_sec, 3)}


# ═════════════════════════════════════════════════════════════════════════
# 参数遍历 / 存档（与 ckpt 的目录约定兼容，但 fp32 逐张量落盘）
# ═════════════════════════════════════════════════════════════════════════
def param_slots(P):
    """按 `v3_train_cli.params_of` **完全相同**的顺序产出 `(name, container, key)`。

    名字只用于存档自检（续训时逐个校验），所以取 `layers.3.csa.wq` / `layers.0.W1.2`
    这种可读形式。
    """
    yield "emb", P, "emb"
    yield "ln_f", P, "ln_f"
    yield "head", P, "head"
    for i, L in enumerate(P["layers"]):
        for k, v in L.items():
            if isinstance(v, dict):
                for kk, t in v.items():
                    if t is not None:
                        yield "layers.%d.%s.%s" % (i, k, kk), v, kk
            elif isinstance(v, list):
                for j in range(len(v)):
                    yield "layers.%d.%s.%d" % (i, k, j), v, j
            else:
                yield "layers.%d.%s" % (i, k), L, k


def param_names(P):
    return [n for n, _c, _k in param_slots(P)]


def _to_np(t):
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t, np.float32)
    return np.ascontiguousarray(np.asarray(t.to_numpy()), np.float32)


def save_params(dirpath, P):
    """把每个参数存成 `params/<序号>_<名字>.npy`（fp32，逐张量落盘，峰值=单张量）。"""
    os.makedirs(dirpath, exist_ok=True)
    n = 0
    for i, (name, cont, key) in enumerate(param_slots(P)):
        np.save(os.path.join(dirpath, "%06d_%s" % (i, name)), _to_np(cont[key]))
        n += 1
    return n


def load_params(dirpath, P):
    """把 `save_params` 的产物**原地**写回已有张量（CUDA 走 `copy_h2d_into`，不新分配）。"""
    slots = list(param_slots(P))
    files = sorted(glob.glob(os.path.join(dirpath, "*.npy")))
    if len(files) != len(slots):
        raise SystemExit(
            "参数个数不一致：checkpoint %d 个，当前模型 %d 个。"
            "修法：续训必须用同一套 --d/--layers/--E/--n_q/--d_ff/--vocab/--csa"
            % (len(files), len(slots)))
    rt = None
    for i, ((name, cont, key), path) in enumerate(zip(slots, files)):
        base = os.path.basename(path)
        want = "%06d_%s.npy" % (i, name)
        if base != want:
            raise SystemExit("参数名不一致：checkpoint 里是 %s，当前模型期望 %s。"
                             "修法：模型结构与 checkpoint 不同"
                             "（--d/--layers/--E/--n_q/--d_ff/--vocab/--csa 必须一致）"
                             % (base, want))
        arr = np.ascontiguousarray(np.load(path), np.float32)
        t = cont[key]
        if tuple(int(s) for s in t.shape) != tuple(arr.shape):
            raise SystemExit("参数 %s 形状不一致：checkpoint %s，当前 %s。"
                             "修法：模型结构必须与 checkpoint 完全一致"
                             % (name, tuple(arr.shape), tuple(t.shape)))
        if getattr(t, "device", "cpu") == "cuda":
            if rt is None:
                rt = T._cuda()
            rt.copy_h2d_into(t._gpu, arr)
        else:
            np.copyto(t.data, arr)
    return len(slots)


def _buffer_to_np(buf):
    if isinstance(buf, np.ndarray):
        return np.ascontiguousarray(buf, np.float32)
    return np.ascontiguousarray(T._cuda().d2h(buf), np.float32)


def _load_into_buffer(buf, arr):
    arr = np.ascontiguousarray(arr, np.float32)
    if isinstance(buf, np.ndarray):
        if tuple(buf.shape) != tuple(arr.shape):
            raise SystemExit("优化器状态形状不一致：%s vs %s。修法：同一模型才能续训"
                             % (tuple(buf.shape), tuple(arr.shape)))
        np.copyto(buf, arr)
    else:
        T._cuda().copy_h2d_into(buf, arr)


def _buffer_to_np_raw(buf):
    """保留原 dtype 的 d2h。

    ★ 绝不能用 `_buffer_to_np`：它强制 `astype(np.float32)`，
    而 `Fp16MomAdamW(state="bf16")` 的 m16/v16 是 **uint16 存 bf16 位模式**，
    按数值转 fp32 会把位模式当数字解释，读回来是垃圾 ⇒ 续训静默出错。
    """
    if isinstance(buf, np.ndarray):
        return np.ascontiguousarray(buf)
    return np.ascontiguousarray(T._cuda().d2h(buf))


def _load_into_buffer_raw(buf, arr):
    """按原 dtype 灌回；**dtype 不符必须立刻报错**，不许静默转换。"""
    arr = np.ascontiguousarray(arr)
    if tuple(buf.shape) != tuple(arr.shape):
        raise SystemExit("优化器状态形状不一致：%s vs %s。修法：同一模型才能续训"
                         % (tuple(buf.shape), tuple(arr.shape)))
    if buf.dtype != arr.dtype:
        raise SystemExit(
            "优化器状态 dtype 不一致：checkpoint %s vs 当前 %s。"
            "修法：用存盘时相同的 --opt-state（fp16/bf16）续训"
            % (arr.dtype, buf.dtype))
    if isinstance(buf, np.ndarray):
        np.copyto(buf, arr)
    else:
        T._cuda().copy_h2d_into(buf, arr)


def _save_optimizer_m16(dirpath, opt):
    """`Fp16MomAdamW` 的 `(t, m16/v16 逐张量)` 落盘（保留 fp16 / uint16-bf16 原 dtype）。"""
    os.makedirs(dirpath, exist_ok=True)
    n = len(opt.params)
    if len(opt.m16) != n or len(opt.v16) != n:
        raise SystemExit("fp16mom 状态长度与参数数不符（%d/%d vs %d）。修法：优化器被换过？"
                         % (len(opt.m16), len(opt.v16), n))
    dt = None
    for i in range(n):
        m = _buffer_to_np_raw(opt.m16[i])
        v = _buffer_to_np_raw(opt.v16[i])
        if dt is None:
            dt = str(m.dtype)
        elif str(m.dtype) != dt:
            raise SystemExit("m16 各张量 dtype 不一致（%s vs %s）" % (dt, m.dtype))
        if str(v.dtype) != dt:
            raise SystemExit("v16 dtype 与 m16 不一致（%s vs %s）" % (v.dtype, dt))
        np.save(os.path.join(dirpath, "%06d_m" % i), m)
        np.save(os.path.join(dirpath, "%06d_v" % i), v)
    with open(os.path.join(dirpath, "t.txt"), "w", encoding="utf-8") as f:
        f.write("%d\n" % int(getattr(opt, "t", 0)))
    with open(os.path.join(dirpath, "dtype.txt"), "w", encoding="utf-8") as f:
        f.write("%s\n" % dt)
    return n


def _load_optimizer_m16(dirpath, opt):
    n = len(opt.params)
    mfiles = sorted(glob.glob(os.path.join(dirpath, "*_m.npy")))
    vfiles = sorted(glob.glob(os.path.join(dirpath, "*_v.npy")))
    if len(mfiles) != n or len(vfiles) != n:
        raise SystemExit("fp16mom 状态个数不一致：checkpoint %d/%d，当前参数 %d。"
                         "修法：同一模型才能续训" % (len(mfiles), len(vfiles), n))
    for i in range(n):
        _load_into_buffer_raw(opt.m16[i], np.load(mfiles[i]))
        _load_into_buffer_raw(opt.v16[i], np.load(vfiles[i]))
    tp = os.path.join(dirpath, "t.txt")
    opt.t = int(open(tp, encoding="utf-8").read().strip()) if os.path.exists(tp) else 0
    return n


def save_optimizer(dirpath, opt, half=True):
    """AdamW 的 `(t, m/v 逐张量)` 落盘（目录名与 `ckpt` 的 `opt/` 约定一致）。

    ★ 2026-11-16：默认按 **fp16** 存 m/v。原因（实测）：1.5B 模型一个 checkpoint
      是 params 6 GB + m 6 GB + v 6 GB = **18 GB**，三次滚动就 54 GB，
      把 V100 的 D 盘（剩余 46 GB）写满并报 `OSError: ... 0 written`。
      Adam 的一阶/二阶矩降到 fp16 是业界常规做法（相对误差 ~5e-4，远小于 lr 本身），
      换来 checkpoint 从 18 GB 降到 **12 GB**。需要逐位复现时用 `half=False`。
    """
    if hasattr(opt, "m16") and hasattr(opt, "v16"):
        # Fp16MomAdamW：状态在 m16/v16（fp16 或 uint16-bf16），走保留 dtype 的实现
        return _save_optimizer_m16(dirpath, opt)
    os.makedirs(dirpath, exist_ok=True)
    n = len(opt.params)
    if len(opt.m) != n or len(opt.v) != n:
        raise SystemExit("优化器状态长度与参数数不符（%d/%d vs %d）。修法：优化器被换过？"
                         % (len(opt.m), len(opt.v), n))
    for i in range(n):
        m = _buffer_to_np(opt.m[i])
        v = _buffer_to_np(opt.v[i])
        if half:
            m = m.astype(np.float16)
            v = v.astype(np.float16)
        np.save(os.path.join(dirpath, "%06d_m" % i), m)
        np.save(os.path.join(dirpath, "%06d_v" % i), v)
    with open(os.path.join(dirpath, "t.txt"), "w", encoding="utf-8") as f:
        f.write("%d\n" % int(getattr(opt, "t", 0)))
    with open(os.path.join(dirpath, "dtype.txt"), "w", encoding="utf-8") as f:
        f.write("float16\n" if half else "float32\n")
    return n


def load_optimizer(dirpath, opt):
    if hasattr(opt, "m16") and hasattr(opt, "v16"):
        return _load_optimizer_m16(dirpath, opt)
    n = len(opt.params)
    mfiles = sorted(glob.glob(os.path.join(dirpath, "*_m.npy")))
    vfiles = sorted(glob.glob(os.path.join(dirpath, "*_v.npy")))
    if len(mfiles) != n or len(vfiles) != n:
        raise SystemExit("优化器状态个数不一致：checkpoint %d/%d，当前参数 %d。"
                         "修法：同一模型才能续训" % (len(mfiles), len(vfiles), n))
    for i in range(n):
        # 兼容 fp16 / fp32 两种落盘：统一升回 fp32 再灌进设备缓冲
        _load_into_buffer(opt.m[i], np.load(mfiles[i]).astype(np.float32))
        _load_into_buffer(opt.v[i], np.load(vfiles[i]).astype(np.float32))
    tp = os.path.join(dirpath, "t.txt")
    opt.t = int(open(tp, encoding="utf-8").read().strip()) if os.path.exists(tp) else 0
    return n


# ═════════════════════════════════════════════════════════════════════════
# 前向 + 损失（块对角掩码 / CSA 分段 / shift + ignore_index）
# ═════════════════════════════════════════════════════════════════════════
def _attention_mask(seg):
    """`seg` `(B,T)` → 加性掩码 `(B,1,T,T)` 的 ORV 张量（0 可见 / -inf 不可见）。"""
    return T.tensor(MK.seg_additive(MK.make_seg_mask(seg)))


def forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                 top_k=2, shared=True, use_seg=True):
    """一次前向 + shift 后的 CE。返回 `(loss, meta)`，`loss` 是 0 维张量。

    与参考实现（`model_v3.py:1368-1515`）对齐的三点：
      ① `seg_id` → 块对角因果掩码（`causal=False` 交给掩码，避免叠两次）；
      ② CSA 层把 `seg` 传给 `make_plan`（按文档边界池化）；
      ③ `CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)`。
    """
    B, Tlen = int(ids.shape[0]), int(ids.shape[1])
    d = int(cfg["d"])
    n_q = int(cfg["n_q"])
    n_kv_attn = int(cfg.get("n_kv_attn", cfg["n_kv"]))
    C = cfg.get("_csa_mod")
    csac = cfg.get("csa_cfg")
    n_csa = sum(1 for k in kinds if k == "csa")

    am = _attention_mask(seg) if use_seg else None
    plan = None
    if n_csa:
        plan = C.make_plan(csac, B, Tlen, seg=(seg if use_seg else None))

    x = P["emb"][ids.reshape(-1)].reshape(B, Tlen, d)
    counts = {"csa": 0, "global": 0}
    # ★ 2026-11-17：**预训练有自己的层循环**（不走 `CLI.forward_logits`）。
    #   重算开关必须在**这里**生效，否则等于没开（实测内核数 6898 完全不变）。
    use_ckpt = (os.environ.get("TG_GRAD_CKPT", "") not in ("", "0", "false", "False")
                or bool((cfg or {}).get("grad_ckpt")))
    for Li, L in enumerate(P["layers"]):
        kind = kinds[Li]
        # 计数放在循环里做一次（重算会再跑一遍层体，不能在里面计数，否则翻倍）
        counts["csa" if kind == "csa" else "global"] += 1

        def _layer(x, Li=Li, L=L, kind=kind):
            xn = V.rmsnorm_orv(x, L["ln1"])
            if kind == "csa":
                attn = C.csa_forward(xn, L["csa"], plan, cos, sin)
            else:
                attn = V.attention_orv(xn, L["wq"], L["wk"], L["wv"], L["wo"], cos, sin,
                                       d // n_q, n_q, n_kv_attn,
                                       causal=(am is None), attn_mask=am)
            h = x + attn
            hn = V.rmsnorm_orv(h, L["ln2"]).reshape(B * Tlen, d)
            if int(top_k) == 1:
                eids, g, _s = MOE.route_top1(hn, L["router"])
                mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
            else:
                eids, gates, _s = MOE.route_topk(hn, L["router"], int(top_k))
                mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
            if shared:
                mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
            return h + mo.reshape(B, Tlen, d)

        x = CLI._checkpoint_layer(_layer, x) if use_ckpt else _layer(x)

    xh = V.rmsnorm_orv(x, P["ln_f"])[:, :Tlen - 1, :].reshape(B * (Tlen - 1), d)
    logits = xh @ P["head"].transpose(0, 1)
    target = np.ascontiguousarray(labels[:, 1:]).reshape(-1)
    loss = T.functional.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    meta = {"counts": counts, "n_csa": n_csa, "n_global": len(kinds) - n_csa,
            "n_valid": int((target != IGNORE_INDEX).sum())}
    return loss, meta


# ═════════════════════════════════════════════════════════════════════════
# 存档 / 续训
# ═════════════════════════════════════════════════════════════════════════
def save_run(out_root, step, corpus, P, opt, sched, state_extra=None, extra_meta=None):
    """写一个完整 checkpoint（`params/` + `opt/` + `train_state.json` + `meta.json`）。"""
    d = ckpt.checkpoint_dir(out_root, step)
    os.makedirs(d, exist_ok=True)
    npar = save_params(os.path.join(d, "params"), P)
    save_optimizer(os.path.join(d, "opt"), opt)
    st = {
        "schema": SCHEMA,
        "step": int(step),
        "kind": "ckpt",
        "data": corpus.state_dict(),
        "schedule_state": sched.state_dict() if sched is not None else None,
        "optimizer_t": int(getattr(opt, "t", 0)),
    }
    if state_extra:
        st.update(state_extra)
    with open(os.path.join(d, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    meta = {"step": int(step), "kind": "ckpt", "tensors": int(npar),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(d, ckpt.CKPT_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return d


def resolve_resume(path):
    """`--resume` 可以是具体 checkpoint 目录，也可以是 `--out` 根目录（取最新的）。"""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isfile(os.path.join(path, "train_state.json")):
        return path
    cks = ckpt.list_checkpoints(path)
    if not cks:
        raise SystemExit("--resume %s 下没有任何 checkpoint（要找 meta.json + train_state.json）。"
                         "修法：指到 step_000030 这种具体目录，或指 --out 根目录" % (path,))
    return cks[-1][1]


def _close(a, b, rel=1e-12):
    """浮点"相等"：给 LR / eta_min 用相对容差，避免 3e-05 与 2.9999999999999997e-05 误报。"""
    a, b = float(a), float(b)
    if a == b:
        return True
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def check_resume_state(st, cfg, sched_cfg, seed):
    """续训前校验：模型配置 / 随机种子 / LR 计划必须一致（否则曲线必然漂）。"""
    if int(st.get("schema", -1)) != SCHEMA:
        raise SystemExit("checkpoint schema=%r，本版本要求 %d。修法：用同版本代码续训"
                         % (st.get("schema"), SCHEMA))
    old = st.get("model")
    if old is not None:
        new = _jsonable(cfg)
        keys = set(old) | set(new)
        diff = [k for k in sorted(keys) if old.get(k) != new.get(k)]
        if diff:
            raise SystemExit(
                "模型配置与 checkpoint 不一致：%s。修法：续训要用同一套模型参数"
                % (", ".join("%s(%r->%r)" % (k, old.get(k), new.get(k)) for k in diff),))
    old_seed = st.get("seed")
    if old_seed is not None and int(old_seed) != int(seed):
        raise SystemExit("seed 不一致：checkpoint %s，本次 %s。"
                         "修法：数据顺序由 seed 决定，续训必须用同一个 seed"
                         "（不给 --seed 时会自动沿用 checkpoint 里的值）"
                         % (old_seed, seed))
    old_s = st.get("schedule") or {}
    diff = [k for k in ("base_lr", "eta_min", "warmup_steps", "t_max")
            if k in old_s and not _close(old_s[k], sched_cfg[k])]
    if diff:
        raise SystemExit(
            "LR 计划和 checkpoint 不一致：%s。修法：① 续训用与原来相同的 "
            "--lr/--steps/--warmup-steps/--min-lr-ratio；② 确实要换计划就加 --reschedule"
            "（曲线会不连续，仅用于有意续训）"
            % (", ".join("%s(%r->%r)" % (k, old_s.get(k), sched_cfg[k]) for k in diff),))
    return True


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="V3 真实预训练（真语料 + LR 调度 + 梯度裁剪 + checkpoint/续训）。"
                    "远端：python -m tgai_go.train.v3_pretrain --data <dir> ...")
    # ---- 模型（数值与 v3_train_cli 同名同义）----
    g = ap.add_argument_group("model")
    g.add_argument("--d", type=int, default=256)
    g.add_argument("--layers", type=int, default=4)
    g.add_argument("--E", type=int, default=4)
    g.add_argument("--n_q", type=int, default=4)
    g.add_argument("--d_ff", type=int, default=0, help="0 = 4*d（真 V3 是 2816）")
    g.add_argument("--vocab", type=int, default=96000,
                   help="词表大小，必须 > 分片里的最大 token id（真 V3 是 96000）")
    g.add_argument("--top_k", type=int, default=2)
    g.add_argument("--csa", "--attn-global-every", dest="csa", type=int, default=0,
                   metavar="N", help="真 V3 的 3:1 混合注意力（每 N 层 1 层 Global），0=关闭")
    g.add_argument("--no-shared", action="store_true", help="关闭共享专家")
    # ---- 数据 ----
    g = ap.add_argument_group("data")
    g.add_argument("--data", nargs="+", required=True,
                   help="分片目录 / 通配符 / 单文件（.npz/.npy/.bin 混排也行）")
    g.add_argument("--seq", type=int, default=256)
    g.add_argument("--bin-dtype", default=None, choices=sorted(_BIN_DTYPES),
                   help="裸 .bin 的 dtype（**必须显式**，猜错不会报错）")
    g.add_argument("--no-seg", action="store_true",
                   help="忽略 seg_id（纯因果、不分段）。只用于对照实验，生产不要开")
    # ---- 训练 ----
    g = ap.add_argument_group("train")
    g.add_argument("--steps", type=int, default=100,
                   help="训练到第几步（同时也是 LR 余弦的 horizon）")
    g.add_argument("--stop-at", type=int, default=0,
                   help="跑到第几步就停（0 = 用 --steps）。制造中断用：--steps 60 --stop-at 30")
    g.add_argument("--batch", type=int, default=1, help="每步几条序列（真 V3 用梯度累积）")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="eta_min = lr * ratio（参考实现默认 0.1）")
    g.add_argument("--warmup-steps", type=int, default=0,
                   help=">0 时用这个步数；=0 时用 --warmup-ratio 乘 --steps")
    g.add_argument("--warmup-ratio", type=float, default=0.01)
    g.add_argument("--wd", type=float, default=0.1, help="AdamW weight decay（参考实现 0.1）")
    g.add_argument("--opt", default="adamw",
                   choices=["adamw", "muon", "fp16mom", "adafactor", "int8"],
                   help="优化器。状态显存（每参数）：adamw 8B / muon(fp16 动量) 2B / fp16mom 4B "
                        "/ adafactor 因子化 O(n+m) / int8 约 1-2B。"
                        "★ 注意：非 adamw 时 checkpoint 的优化器状态暂不支持续训（会明确拒绝）")
    g.add_argument("--opt-state", dest="opt_state", default="fp16",
                   choices=["fp16", "bf16", "fp32"],
                   help="--opt fp16mom 的动量存储精度。实测 fp16 会让 Adam 的二阶动量 v"
                        "（常 1e-8 量级，fp16 最小正规数 6.1e-5）下溢成 0，"
                        "第 2 步 loss 就从 11.7 炸到 135；bf16 指数范围与 fp32 相同不会下溢，"
                        "V100 上可用（存储 bf16、计算 fp32）")
    g.add_argument("--amp", default="off", choices=["off", "gemm", "mlp", "both"],
                   help="混合精度。gemm=仅 GEMM 输入 fp16（参数/激活/梯度仍 fp32）；"
                        "mlp=MLP 内激活存 fp16（省带宽，带梯度缩放）；both=两者都开。"
                        "★ 实测：V100 上 fp16 张量核峰值 125 TFLOPS vs fp32 15.7，"
                        "但小配置下 cast 开销可能盖过收益，必须在目标配置上实测。")
    g.add_argument("--amp-no-cache", dest="amp_no_cache", action="store_true",
                   help="AMP 时不启用 fp16 权重影子缓存。1.5B 模型影子要 3 GB，"
                        "开 --amp gemm 可能 OOM；关掉则每次重新 cast（慢一点但省 3 GB）。")
    g.add_argument("--muon-lr", type=float, default=0.02,
                   help="Muon 自己的学习率（默认 0.02，与 AdamW 的 1e-3 不同量级；用 --opt muon 时生效）")
    g.add_argument("--muon-momentum", type=float, default=0.95)
    g.add_argument("--muon-fp16-mom", action="store_true", default=True,
                   help="Muon 动量存 fp16（默认开，每参数 2B；关掉则 fp32 动量）")
    g.add_argument("--muon-fp32-mom", dest="muon_fp16_mom", action="store_false",
                   help="Muon 动量存 fp32（更准，但每参数 4B）")
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument("--grad-clip", type=float, default=1.0, help="<=0 关闭裁剪")
    g.add_argument("--seed", type=int, default=None,
                   help="不给则 42；续训时自动沿用 checkpoint 里的 seed")
    g.add_argument("--policy", default=None,
                   choices=["standard", "speed", "extreme", "memory"],
                   help="ORV 执行档位（显存/速度取舍，见 T.set_policy）")
    g.add_argument("--fp16-gemm", action="store_true",
                   help="打开 fp16 GEMM（V100 实测慢 16.7%% 且多占 3GB，仅复现用）")
    g.add_argument("--log-every", type=int, default=10)
    # ---- 存档 ----
    g = ap.add_argument_group("checkpoint")
    g.add_argument("--out", default=None,
                   help="checkpoint 根目录（写 step_%%06d 子目录）；不给则只训练不存档")
    g.add_argument("--ckpt-every", type=int, default=0, help="每 N 步存一次（0=只在结束时存）")
    g.add_argument("--keep", type=int, default=2, help="只保留最近 K 个 checkpoint")
    g.add_argument("--resume", default=None,
                   help="checkpoint 目录（或 --out 根目录，取最新的）")
    g.add_argument("--no-final-save", action="store_true", help="结束时不再存（大模型省盘）")
    g.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true",
                   help="梯度重算：每层前向不建图、反向重跑（省激活显存，多花约一层的前向时间）")
    g.add_argument("--reschedule", action="store_true",
                   help="允许续训时改 LR 计划（默认严格校验一致）")
    # ---- 其它 ----
    ap.add_argument("--json-out", default=None, help="把 RESULT 额外写到这个 json")
    ap.add_argument("--dry-run", action="store_true",
                    help="只探测数据 + 建模型 + 打印统计，不训练")
    return ap


def _sched_cfg(a, steps):
    steps = int(steps)
    warmup = int(a.warmup_steps) if int(a.warmup_steps) > 0 else \
        int(round(float(a.warmup_ratio) * steps))
    warmup = max(0, min(warmup, steps - 1)) if steps > 1 else 0
    return {"base_lr": float(a.lr), "eta_min": float(a.lr) * float(a.min_lr_ratio),
            "warmup_steps": int(warmup),
            "t_max": int(max(1, steps - warmup)), "horizon": steps}


def _nvidia_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _vram_peak(stop, base, out):
    """后台采样 nvidia-smi 的显存峰值（池高水位之外的独立证据）。"""
    peak = base if base is not None else 0
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    out["peak_mib"] = peak


def _expand_cfg(a):
    cfg = dict(d=a.d, d_ff=(a.d_ff or 4 * a.d), E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=a.vocab, seq=a.seq)
    if getattr(a, "grad_ckpt", False):
        # 梯度重算：用算力换显存（每层前向不建图，反向重跑）。
        # 原型实测梯度逐位等价；默认关闭，只有显式 --grad-ckpt 才开。
        cfg["grad_ckpt"] = True
    if int(a.csa) > 0:
        cfg.update(csa=int(a.csa), n_kv_attn=a.n_q, rope_theta=500000.0,
                   csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    return cfg


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    if int(a.steps) < 1:
        raise SystemExit("--steps 必须 >= 1")
    if int(a.seq) < 2:
        raise SystemExit("--seq 必须 >= 2（shift 后至少要剩 1 个目标 token）")
    stop_step = int(a.stop_at) if int(a.stop_at) > 0 else int(a.steps)
    if stop_step > int(a.steps):
        raise SystemExit("--stop-at %d 超过 --steps %d。修法：--stop-at <= --steps"
                         "（--steps 是 LR horizon，--stop-at 只是提前收工）"
                         % (stop_step, int(a.steps)))
    if a.csa and (a.d // a.n_q) % 2 != 0:
        raise SystemExit("--csa 要求 d/n_q 为偶数：d=%d, n_q=%d ⇒ d_k=%d"
                         % (a.d, a.n_q, a.d // a.n_q))
    if a.csa and int(a.d // a.n_q) < 2:
        raise SystemExit("--csa 要求 d/n_q >= 2（d=%d, n_q=%d）" % (a.d, a.n_q))

    # ---- 续训目录 / seed 解析（都要在建语料之前定下来：数据顺序由 seed 决定）----
    resume_dir = resolve_resume(a.resume) if a.resume else None
    saved = None
    if resume_dir:
        with open(os.path.join(resume_dir, "train_state.json"), encoding="utf-8") as f:
            saved = json.load(f)
    seed = int(a.seed) if a.seed is not None else 42
    if a.seed is None and saved is not None and saved.get("seed") is not None:
        seed = int(saved["seed"])

    # ---- 数据（先探测：格式/规模/词表在占 GPU 之前就能判断）----
    paths = expand_data_paths(a.data)
    corpus = PackedCorpus(paths, seq=a.seq, seed=seed, bin_dtype=a.bin_dtype,
                          vocab=a.vocab)
    fmts = {}
    for info in corpus.infos:
        fmts[info.fmt] = fmts.get(info.fmt, 0) + 1
    print("[data] %d shards %s  seq=%d  windows=%d  tokens=%d  scan=%.2fs"
          % (corpus.n_shards, fmts, corpus.seq, corpus.total_windows,
             corpus.total_tokens, corpus.scan_sec), flush=True)
    print("[data] 1 epoch = %d windows = %d tokens; steps/epoch @B=%d = %d"
          % (corpus.total_windows, corpus.total_windows * corpus.seq, int(a.batch),
             corpus.total_windows // max(1, int(a.batch))), flush=True)
    if corpus.n_no_labels:
        print("[data] WARN %d shards have no 'labels' array -> labels=ids"
              " (doc-start tokens stay supervised)" % (corpus.n_no_labels,), flush=True)
    corpus.check_vocab()

    # ---- 设备 ----
    T.set_default_device("cuda")
    if a.policy:
        T.set_policy(a.policy)
    if a.fp16_gemm:
        T._cuda().fp16_gemm = True
        print("[dev] rt.fp16_gemm=True (V100: slower + 3GB, reproduce only)", flush=True)
    rt = T._cuda()
    base_mib = _nvidia_used_mib()

    # ---- 模型 ----
    cfg = _expand_cfg(a)
    kinds = CLI.layer_kinds(cfg)
    n_csa = sum(1 for k in kinds if k == "csa")
    if n_csa:
        from tgai_go.train import v3_csa as _C
        cfg["_csa_mod"] = _C
    rng = np.random.default_rng(seed)
    t_build0 = _now()
    P = CLI.build(cfg, rng)                      # csa>0 时它会写回 cfg["csa_cfg"]
    params = CLI.params_of(P)
    nparam = int(sum(int(t.numel()) for t in params))
    print("[model] d=%d layers=%d E=%d d_ff=%d vocab=%d n_q=%d top_k=%d csa=%d "
          "(csa=%d global=%d) tensors=%d params=%.3fM build=%.2fs seed=%d"
          % (cfg["d"], cfg["layers"], cfg["E"], cfg["d_ff"], cfg["vocab"], cfg["n_q"],
             int(a.top_k), int(a.csa), n_csa, len(kinds) - n_csa, len(params),
             nparam / 1e6, _now() - t_build0, seed), flush=True)

    # 位置编码表（与 v3_train_cli 同源）
    if n_csa:
        C = cfg["_csa_mod"]
        cos, sin = C.rope_tables(cfg["csa_cfg"], seq_len=a.seq)
    else:
        cos, sin = V.rope_cos_sin(a.seq, cfg["d"] // cfg["n_q"], 10000.0)
        cos, sin = T.tensor(cos), T.tensor(sin)

    # ---- 优化器 / 调度器 ----
    if a.opt == "adamw":
        opt = T.optim.AdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),
                            eps=float(a.eps), weight_decay=float(a.wd))
    elif a.opt == "muon":
        # Muon：Newton-Schulz 正交化动量。lr 是它自己的量级（0.02，不是 AdamW 的 1e-3），
        # 所以走 --muon-lr；--lr 对它不生效（避免误用 AdamW 的量级把权重打飞）。
        opt = T.optim.Muon(params, lr=float(a.muon_lr), momentum=float(a.muon_momentum),
                           weight_decay=float(a.wd), fp16_mom=bool(a.muon_fp16_mom))
    elif a.opt == "fp16mom":
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        opt = Fp16MomAdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),
                           eps=float(a.eps), weight_decay=float(a.wd),
                           state=str(getattr(a, "opt_state", "fp16")))
    elif a.opt == "adafactor":
        opt = T.optim.Adafactor(params, lr=float(a.lr), weight_decay=float(a.wd))
    elif a.opt == "int8":
        from tgai_go.train.int8_adamw import Int8AdamW
        opt = Int8AdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),
                        eps=float(a.eps), weight_decay=float(a.wd))
    else:
        raise SystemExit("未知的 --opt=%r。修法：用 adamw / fp16mom / adafactor / int8"
                         % (a.opt,))
    # checkpoint 的优化器状态目前只实现了 AdamW 的 (t, m/v) 落盘。
    # 别的优化器不是"存不下"，而是存了也不能正确续训 —— 必须明确拒绝，不能静默错。
    if a.opt not in ("adamw", "fp16mom") and (a.ckpt_every > 0 or a.resume):
        # ★ 2026-11-17：`fp16mom` 已支持（按原 dtype 存 m16/v16，见 _save_optimizer_m16）。
        #   其余优化器（muon/adafactor/int8）仍然拒绝 —— 存了也不能正确续训，
        #   必须明确拒绝，不能静默错。
        raise SystemExit(
            "--opt=%s 目前不支持 checkpoint 续训：优化器状态落盘只实现了 AdamW 的 (t, m/v)。"
            "修法：要么用 --opt adamw 跑长训练；要么去掉 --ckpt-every/--resume "
            "只做显存与吞吐验证。" % (a.opt,))
    # ---- 混合精度（AMP）----
    # gemm: 只把 GEMM 输入 cast 成 fp16（参数/激活/梯度仍 fp32，不需要 GradScaler）
    # mlp : MLP 块内激活存 fp16（省带宽，自带梯度缩放）
    # both: 两者都开
    _AMP = {"gemm": a.amp in ("gemm", "both"), "mlp": a.amp in ("mlp", "both")}
    if _AMP["gemm"]:
        T.amp_on(dtype="fp16", weight_cache=not bool(a.amp_no_cache))
        # ★ 只把**参数**地址登记给 fp16 影子缓存：参数整步不变，缓存安全；
        #   激活/中间量不登记 ⇒ 每次现 cast，不可能读到池子复用留下的旧影子。
        try:
            ptrs = []
            for p_ in params:
                g = getattr(p_, "_gpu", None) or getattr(p_, "gpu", None)
                pv = int(getattr(g, "ptr", 0) or 0)
                if pv:
                    ptrs.append(pv)
            n_pin = rt.f16_cache_pin(ptrs) if ptrs else 0
            print("[amp] 已登记 %d 个参数地址给 fp16 影子缓存" % n_pin, flush=True)
        except Exception as _e:
            print("[amp] 参数登记失败（缓存将不生效，只影响速度）：%r" % (_e,), flush=True)
    if _AMP["mlp"]:
        from tgai_go.train import amp_mlp
        amp_mlp.enable()
    if a.amp != "off":
        print("[amp] mode=%s state=%r" % (a.amp, T.amp_state()), flush=True)

    sc = _sched_cfg(a, int(a.steps))
    sched = T.optim.lr_scheduler.CosineLR(
        opt, T_max=sc["t_max"], eta_min=sc["eta_min"],
        warmup_steps=sc["warmup_steps"], warmup_start=0.0)
    print("[optim] AdamW lr=%.3e wd=%.3g betas=(%.3g,%.3g) eps=%.1e clip=%.3g"
          % (a.lr, a.wd, a.beta1, a.beta2, a.eps, a.grad_clip), flush=True)
    print("[sched] CosineLR warmup=%d t_max=%d eta_min=%.3e horizon=%d"
          % (sc["warmup_steps"], sc["t_max"], sc["eta_min"], sc["horizon"]), flush=True)

    # ---- 续训：恢复参数 / 优化器 / 调度器 / 数据位置 ----
    start_step = 0
    if saved is not None:
        if not a.reschedule:
            check_resume_state(saved, cfg,
                               {"base_lr": sc["base_lr"], "eta_min": sc["eta_min"],
                                "warmup_steps": sc["warmup_steps"], "t_max": sc["t_max"]},
                               seed)
        else:
            print("[resume] --reschedule: LR plan may jump (intended)", flush=True)
        t_ld = _now()
        n_par = load_params(os.path.join(resume_dir, "params"), P)
        n_opt = load_optimizer(os.path.join(resume_dir, "opt"), opt)
        if saved.get("schedule_state"):
            sched.load_state_dict(saved["schedule_state"])
        corpus.load_state_dict(saved["data"])
        start_step = int(saved["step"])
        print("[resume] %s step=%d params=%d opt=%d data(epoch,rank,offset)=%s "
              "loaded=%.2fs"
              % (resume_dir, start_step, n_par, n_opt,
                 (corpus.epoch, corpus.rank, corpus.offset), _now() - t_ld), flush=True)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
    if a.dry_run:
        print("[dry-run] data/model/optim OK; no step executed. pool=%s"
              % (rt.pool_stats(),), flush=True)
        return 0
    if start_step >= stop_step:
        print("[done] checkpoint 已经到 step %d >= 目标 %d，无需再训"
              % (start_step, stop_step), flush=True)
        return 0

    # ---- 训练 ----
    log_path = os.path.join(a.out, "train_log.jsonl") if a.out else None
    log_f = open(log_path, "a", encoding="utf-8") if log_path else None
    stop_ev = threading.Event()
    prof = {}
    th = threading.Thread(target=_vram_peak, args=(stop_ev, base_mib, prof), daemon=True)
    th.start()

    hist_tail = []
    losses, secs = [], []
    t_run0 = _now()
    step = start_step - 1
    aborted = None
    try:
        for step in range(start_step, stop_step):
            lr = sched.step()
            t0 = _now()
            ids, labels, seg = corpus.take(int(a.batch))
            opt.zero_grad()
            loss, fmeta = forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                                       top_k=int(a.top_k), shared=(not a.no_shared),
                                       use_seg=(not a.no_seg))
            lv = _scalar(loss)
            if not _is_finite(lv):
                aborted = ("loss", lv, step, ids, labels, seg)
                break
            loss.backward()
            del loss
            gnorm = float(T.clip_grad_norm_(params, float(a.grad_clip))) \
                if float(a.grad_clip) > 0 else 0.0
            if not _is_finite(gnorm):
                aborted = ("grad_norm", gnorm, step, ids, labels, seg)
                break
            opt.step()
            # ★ 权重已更新 ⇒ 必须让 fp16 权重影子失效，否则下一步会用上一步的 fp16 权重。
            #   `cuda.py` 写明"缓存失效责任在调用方"；只 amp_on 不失效是静默错误。
            if _AMP["gemm"]:
                T.amp_invalidate()
                # ★ 还要清运行时层的 fp16 权重影子缓存：`cuda.py` 里那份 `_f16_cache`
                #   只有 Fp16MomAdamW 会主动清，`T.amp_invalidate()` 清的是模块级影子。
                #   不清就会用上一步的 fp16 权重做前向 ⇒ 梯度与当前权重不匹配（实测
                #   grad_norm 从 12.49 飙到 145775，训练直接坏掉）。
                try:
                    rt.f16_cache_clear()
                except Exception:
                    pass
            if _AMP["mlp"]:
                amp_mlp.invalidate()
            rt.sync()
            dt = _now() - t0
            losses.append(lv)
            secs.append(dt)
            hist_tail.append([int(step), round(lv, 8)])
            if len(hist_tail) > 64:
                del hist_tail[:-64]
            tok = int(a.batch) * int(cfg["seq"])
            rec = {"step": int(step), "loss": lv, "lr": float(lr), "gnorm": gnorm,
                   "sec": dt, "tok": tok, "tok_s": tok / dt if dt > 0 else 0.0,
                   "consumed": int(corpus.consumed),
                   "peak_live": int(rt.pool_stats()["peak_live_bytes"]),
                   "n_valid": fmeta["n_valid"]}
            if log_f is not None:
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_f.flush()
            if int(a.log_every) > 0 and ((step - start_step) % int(a.log_every) == 0
                                        or step == stop_step - 1):
                ps = rt.pool_stats()
                print("step %6d/%d loss=%.6f lr=%.3e gnorm=%9.3f %.3fs tok/s=%8.1f "
                      "peak_live=%.2fGiB consumed=%d"
                      % (step, stop_step, lv, lr, gnorm, dt, rec["tok_s"],
                         _gib(ps["peak_live_bytes"]), corpus.consumed), flush=True)
            # ---- 周期性存档 ----
            if a.out and int(a.ckpt_every) > 0 and (step + 1) % int(a.ckpt_every) == 0:
                d = save_run(a.out, step + 1, corpus, P, opt, sched,
                             state_extra={"seed": seed, "model": _jsonable(cfg),
                                          "schedule": dict(sc), "batch": int(a.batch),
                                          "history_tail": hist_tail},
                             extra_meta={"loss": lv, "lr": float(lr)})
                rm = ckpt.rotate(a.out, int(a.keep))
                print("[ckpt] %s rotated_out=%d" % (d, len(rm)), flush=True)
    finally:
        stop_ev.set()
        th.join(timeout=3)
        if log_f is not None:
            log_f.close()

    # ---- NaN/Inf：停下 + 存现场（绝不静默继续）----
    if aborted is not None:
        kind, value, bad_step, ids, labels, seg = aborted
        dump = os.path.join(a.out or ".", "nan", "nan_step_%06d" % int(bad_step))
        os.makedirs(dump, exist_ok=True)
        np.savez(os.path.join(dump, "batch.npz"), ids=ids, labels=labels, seg=seg)
        try:
            save_params(os.path.join(dump, "params"), P)
            save_optimizer(os.path.join(dump, "opt"), opt)
            with open(os.path.join(dump, "train_state.json"), "w", encoding="utf-8") as f:
                json.dump({"schema": SCHEMA, "step": int(bad_step), "kind": "nan_dump",
                           "seed": seed, "model": _jsonable(cfg), "schedule": dict(sc),
                           "data": corpus.state_dict(), "history_tail": hist_tail,
                           "bad": {"kind": kind, "value": value}}, f,
                          ensure_ascii=False, indent=1)
        except Exception as e:                    # pragma: no cover - 现场落盘失败
            print("[nan] WARN 现场落盘失败：%s" % (e,), flush=True)
        print("[nan] ABORT step=%d %s=%r dump=%s" % (bad_step, kind, value, dump), flush=True)
        print("RESULT " + json.dumps({"status": "aborted", "kind": kind, "value": value,
                                      "step": int(bad_step), "dump": dump},
                                     ensure_ascii=False), flush=True)
        return 2

    # ---- 收尾存档 ----
    final_dir = None
    if a.out and not a.no_final_save:
        final_dir = save_run(a.out, step + 1, corpus, P, opt, sched,
                             state_extra={"seed": seed, "model": _jsonable(cfg),
                                          "schedule": dict(sc), "batch": int(a.batch),
                                          "history_tail": hist_tail},
                             extra_meta={"loss": losses[-1] if losses else None,
                                         "final": True})
        rm = ckpt.rotate(a.out, int(a.keep))
        print("[ckpt] final %s rotated_out=%d" % (final_dir, len(rm)), flush=True)

    ps = rt.pool_stats()
    steady = secs[2:] if len(secs) > 3 else (secs[1:] if len(secs) > 1 else secs)
    res = {
        "status": "ok",
        "schema": SCHEMA,
        "seed": seed,
        "cfg": _jsonable(cfg),
        "kinds": kinds,
        "step_range": [int(start_step), int(step) + 1],
        "losses": [round(x, 8) for x in losses],
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "step_s_mean": float(np.mean(secs)) if secs else None,
        "step_s_steady": float(np.mean(steady)) if steady else None,
        "tok_s_steady": (int(a.batch) * int(cfg["seq"]) / float(np.mean(steady))
                         if steady else None),
        "pool_peak_live_bytes": int(ps["peak_live_bytes"]),
        "pool_peak_bytes": int(ps["peak_bytes"]),
        "pool_peak_live_gib": round(_gib(ps["peak_live_bytes"]), 3),
        "nvidia_base_mib": base_mib,
        "nvidia_peak_mib": prof.get("peak_mib"),
        "data": corpus.stats(),
        "consumed": int(corpus.consumed),
        "data_state": corpus.state_dict(),
        "params": nparam,
        "wall_sec": round(_now() - t_run0, 3),
        "log": log_path,
        "resume": resume_dir,
        "out": a.out,
        "final_ckpt": final_dir,
    }
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "losses"},
                                 ensure_ascii=False), flush=True)
    print("losses %r" % (res["losses"],), flush=True)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
