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


def _aux_coefs():
    """读 MoE 辅助项系数：`(z_loss 系数, aux_bal 系数)`，默认都是 0。

    对齐 HOPE V3 的 `moe_router_z_loss` / `moe_aux_alpha`（各 1e-4）。
    默认 0 ⇒ `forward_loss` 里那段**完全跳过**，训练损失与 1.0.7 逐位相同。
    非法值抛 ValueError（不静默当 0，免得"以为开了其实没开"）。
    """
    out = []
    for nm in ("ORV_Z_LOSS", "ORV_AUX_BAL"):
        raw = os.environ.get(nm, "")
        if raw == "":
            out.append(0.0)
            continue
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(
                "%s=%r 不是数字。修法：传浮点数（如 0.0001），或删掉该环境变量"
                % (nm, raw))
        if v < 0:
            raise ValueError("%s 不能为负（实际 %r）。修法：传 >=0 的浮点数" % (nm, v))
        out.append(v)
    return tuple(out)


# ═════════════════════════════════════════════════════════════════════════
# 小工具
# ═════════════════════════════════════════════════════════════════════════
def _check_overlap_stats(st, world):
    """重叠路径的**硬自检**：世界>1 时它必须真的产生了通信时间。

    2026-11-21 实测（2×4090, `--dp 2`, mode=host, 13 桶）：
    `[dp-ov] comm=0.00ms exposed=0.03ms pending_zeroed=True` —— 看起来完全健康，
    但普通 `all_reduce_grads` 一次都没被调用、两个 rank 的 gnorm 相差 1e-2
    ⇒ **梯度根本没被归约，数据并行退化成"没开 DP"**，而 loss 照样在降。

    这种"静默少做工作"比慢一万倍更危险，所以宁可当场炸掉：
    `comm_ms` 几乎为 0 就是"没搬东西"的确定信号（无 P2P 时归约 3 GiB 至少几十毫秒）。
    """
    if st is None or int(world) <= 1:
        return
    try:
        comm = float(st.get("comm_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        comm = 0.0
    if comm <= 0.5:
        # 先把诊断打全：每桶耗时 / 模式 / 参数数 / missing 计数，再炸。
        try:
            per = st.get("bucket_comm_ms")
            print("[dp-ov] 诊断：mode=%s params=%s deduped=%s buckets=%s "
                  "per_bucket_ms=%s"
                  % (st.get("mode"), st.get("params"), st.get("deduped"),
                     st.get("buckets"), per), flush=True)
        except Exception:                            # noqa: BLE001
            pass
        raise RuntimeError(
            "重叠归约没有真的发生（[dp-ov] comm=%.2fms，world=%d）" % (comm, int(world)) +
            "。这说明梯度没有被归约，数据并行会退化成各卡各训（loss 仍会下降，"
            "极难发现）。修法：设 TG_DP_OVERLAP=0 用不重叠路径；"
            "若你必须用重叠路径，先修好 HostBucketTransport 的后台线程再开")


def _overlap_enabled():
    """「反向即归约」是否启用。**默认关（opt-in）**，这是 2026-11-21 实测后改的。

    为什么默认关掉：2×4090 实测（`--dp 2`，fp32，host 后端，4 步，gnorm 打 9 位小数）

    | 配置 | 普通 `all_reduce_grads` 被调用 | `[dp-ov] comm` | 两 rank 的 gnorm |
    | --- | --- | --- | --- |
    | `TG_DP_OVERLAP=0` | 是（每步 1 次）| 无 | **逐位相同**（4/4 步）|
    | 默认（重叠开）| **0 次** | **0.00 ms** | **相差 1e-2 ~ 3e-4** |

    也就是说重叠路径**根本没有真的归约梯度**（却报 `pending_zeroed=True`，看起来很正常），
    数据并行静默退化成「各卡用自己的本地梯度」—— 这比慢一万倍更危险：
    loss 照样下降，只有逐位对拍才看得出来。

    在 `docs/DP_OVERLAP_2026-11-21.md` 的 bug 修好、并且有逐位证据之前，它只能是**显式开关**。
    要开：`TG_DP_OVERLAP=1`。
    """
    v = os.environ.get("TG_DP_OVERLAP", "")
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def _now():
    """计时统一用 perf_counter（Windows 上 time.time() 粒度 15.6 ms，不可用）。"""
    return time.perf_counter()


def _bf16_requested():
    """用户是否显式要求 bf16 GEMM（`TG_BF16=1`，与 `cuda.py` 的判定同一套取值）。

    抽出来只为一件事：本文件的 AMP 分支固定 fp16，`TG_BF16=1` 会被
    `amp_api._set()` 主动关掉 —— 那是个**静默**失效，实测没有任何日志。
    这个函数让调用点能把"被忽略"写进日志（见 main() 的 --amp 分支）。
    """
    return str(os.environ.get("TG_BF16", "")).strip().lower() in (
        "1", "true", "yes", "on")


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


def _ce_chunk_rows(cli_value=0):
    """`--ce-chunk` / `TG_CE_CHUNK` 解析（环境变量优先，便于 A/B 不改命令行）。

    返回**每块的 token 行数**；0 / 负数 / 解析失败 ⇒ 0（= 不分块，走原来的整块
    `logits = xh @ head.T` 路径，行为与引入本开关之前**逐位一致**）。
    """
    v = os.environ.get("TG_CE_CHUNK", "")
    if str(v).strip() not in ("", "0"):
        try:
            return max(0, int(str(v).strip()))
        except (TypeError, ValueError):
            pass
    try:
        return max(0, int(cli_value or 0))
    except (TypeError, ValueError):
        return 0


def _chunked_ce(xh, target, head, chunk):
    """把 head + CE 切成 `chunk` 行一块，**按有效 token 数加权**合并。

    ## 为什么必须配梯度重算

    只切 matmul 不重算的话，autograd 会把**每块**的 logits 留到反向（`ce_bwd` 需要
    logits 算 dlogits），峰值等于没省。所以每块都过 `CLI._checkpoint_layer`：
    前向不建图（中间量立刻释放），反向用保存的那一小块输入重跑。仓库里已验证
    重算的梯度**逐位等价**（`lab/prof/ckpt_proto.py`，max_abs = 0.000e+00）。

    ## 为什么不能"各块 mean 再平均"

    不分块的语义是 `sum(有效行) / N`（`ignore_index=-100` 的行被排除）。
    `mean_c` 的分母是**本块有效行数** `n_c`，不是块大小；先还原 `sum_c = mean_c × n_c`
    再按 `n_c / N` 加权，才等于原来的 `sum/N`。
    （仓库里记过同类坑：无效行若计入均值，loss 会偏大 —— 实测 8.09 vs 4.69。）
    """
    rows = int(xh.shape[0])
    chunk = int(chunk)
    nvalid = int((target != IGNORE_INDEX).sum())
    if chunk <= 0 or chunk >= rows or nvalid <= 0:
        logits = xh @ head.transpose(0, 1)
        return T.functional.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    inv = 1.0 / float(nvalid)
    loss = None
    for s in range(0, rows, chunk):
        e = min(rows, s + chunk)
        tc = target[s:e]
        nc = int((tc != IGNORE_INDEX).sum())
        if nc == 0:                        # 整块被 mask 掉：对 loss 与梯度都无贡献
            continue
        xc = xh[s:e]

        # ★ `head.transpose(0, 1)` 必须**在块内**、且每块各自一份：
        #   放到块外共用同一个转置视图节点时，两个块的反向会共用同一条
        #   (headT → head) 边 —— 实测第 2 个块的反向把第 1 个块的 head 梯度**覆盖**掉，
        #   gnorm 变成 15.3866 → 10.759（≈ 15.3866/√2，正好是"少一半"的特征）。
        def _one(xin, tc=tc):
            return T.functional.cross_entropy(
                xin @ head.transpose(0, 1), tc, ignore_index=IGNORE_INDEX)

        # ★★ 2026-11-21：这里**可以**用 `_checkpoint_layer` —— 但前提是那个多次并列调用的
        #   梯度 bug 已修（见 `v3_train_cli._checkpoint_layer` 的 2026-11-21 段）。
        #   修之前的实测（留档，别再踩）：
        #       chunk=128（4 块，不重算）gnorm 逐位相同；chunk=256 d=1.49e-06
        #       套上 `_checkpoint_layer` 就变 gnorm ≈ base/√k（2 块 15.3866→10.7590，
        #       4 块 →7.28，8 块 →5.12）= k 个块只有 1 个的梯度活下来。
        #   修好后（同一套测量）：chunk=64/128/256 的 gnorm d ≤ 5.7e-06、loss d ≤ 6.7e-06。
        part = CLI._checkpoint_layer(_one, xc) * (float(nc) * inv)
        loss = part if loss is None else loss + part
    if loss is None:                       # 理论上到不了（nvalid>0 已判过）
        logits = xh @ head.transpose(0, 1)
        return T.functional.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    return loss


def _overlap_order(P, params):
    """反向重叠归约的桶**就绪顺序**（下标排列，各 rank 必须逐字相同）。

    * 反向是前向的逆序 ⇒ 默认取 `params` 倒序（`dp_overlap.build_plan` 的默认）；
    * 但 `head` 的梯度在反向**一开始**就有（CE→logits 的 matmul 反向），而 `emb`
      的梯度要到**最后**（embedding 反向）⇒ head 提到最前、emb 压到最后，
      桶才会尽早满、尽早开始集合通信。
    * tied（`P["head"] is P["emb"]`）：同一个张量要等 embedding 那条路也累加完才完整，
      ⇒ 只能放**最后**，绝不能放最前（否则第一个桶永远等到反向结束才发）。
    """
    from . import dp_overlap as _ovl
    emb = P.get("emb")
    head = P.get("head")
    if head is not None and head is emb:
        return _ovl.plan_order_adapter(params, last=[emb])
    return _ovl.plan_order_adapter(params, first=[head] if head is not None else (),
                                   last=[emb] if emb is not None else ())


def forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                 top_k=2, shared=True, use_seg=True, ce_chunk=0):
    """一次前向 + shift 后的 CE。返回 `(loss, meta)`，`loss` 是 0 维张量。

    与参考实现（`model_v3.py:1368-1515`）对齐的三点：
      ① `seg_id` → 块对角因果掩码（`causal=False` 交给掩码，避免叠两次）；
      ② CSA 层把 `seg` 传给 `make_plan`（按文档边界池化）；
      ③ `CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)`。
    """
    # ★ 2026-11-20：MoE 辅助项（对齐 HOPE V3 的 `model_v3.py::aux_loss()`）。
    #   参考公式（`model_v3.py:1079-1086` 与 `1126-1133`）：
    #       z   = (logsumexp(router_logits, -1) ** 2).mean()
    #       f   = (E / (top_k*N)) * onehot.sum(0)      # 每专家被选比例（无梯度）
    #       P   = probs.mean(0)                        # 平均路由概率（有梯度）
    #       aux = z * moe_router_z_loss + (f*P).sum() * moe_aux_alpha
    #   系数默认 0 ⇒ **行为与以前逐位相同**；用 ORV_Z_LOSS / ORV_AUX_BAL 打开。
    _aux_z, _aux_a = _aux_coefs()
    aux_terms = []
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
            # ★ MoE 辅助项（系数为 0 时这段完全跳过 ⇒ 与以前逐位相同）
            if _aux_z or _aux_a:
                _E = int(L["router"].shape[0])
                _N = int(hn.shape[0])
                if _aux_z:
                    _lg = hn @ L["router"].transpose(0, 1)
                    # `x_j - log_softmax_j == lse`（对每列成立），见 nn.MoE._router_z_loss
                    _lp = T.functional.log_softmax(_lg, axis=-1)
                    _lse = (_lg - _lp).sum(axis=-1) / float(_E)
                    aux_terms.append((_lse * _lse).mean() * _aux_z)
                if _aux_a:
                    _cnt = np.bincount(np.asarray(eids).reshape(-1), minlength=_E)
                    _f = T.tensor((_E / float(int(top_k) * _N))
                                  * _cnt.astype(np.float32))
                    _P = _s.sum(axis=0) * (1.0 / float(_N))
                    aux_terms.append((_f * _P).sum() * _aux_a)
            return h + mo.reshape(B, Tlen, d)

        x = CLI._checkpoint_layer(_layer, x) if use_ckpt else _layer(x)

    xh = V.rmsnorm_orv(x, P["ln_f"])[:, :Tlen - 1, :].reshape(B * (Tlen - 1), d)
    target = np.ascontiguousarray(labels[:, 1:]).reshape(-1)
    # ★ 2026-11-21：head+CE 可选分块（`--ce-chunk` / `TG_CE_CHUNK`）。
    #   默认 0 ⇒ 走下面这两行**原样不动**，与引入开关之前逐位一致。
    _cc = _ce_chunk_rows(ce_chunk)
    if _cc and _cc < int(xh.shape[0]):
        loss = _chunked_ce(xh, target, P["head"], _cc)
    else:
        logits = xh @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    _aux_val = 0.0
    if aux_terms:                      # 只有系数非 0 时才会有内容
        _acc = aux_terms[0]
        for _t in aux_terms[1:]:
            _acc = _acc + _t
        loss = loss + _acc
        try:
            _aux_val = float(_acc.read() if hasattr(_acc, "read") else _acc.data)
        except Exception:              # noqa: BLE001 - 纯日志用，取不到就留 0
            _aux_val = float("nan")
    meta = {"counts": counts, "n_csa": n_csa, "n_global": len(kinds) - n_csa,
            "aux": _aux_val,
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
    g.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true",
                   help="emb 与 head 共享同一个参数张量（参考 tie_embeddings=True）；"
                        "用于验 DP 重叠归约在 tied 下逐位正确")
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
                   choices=["adamw", "muon", "fp16mom", "adafactor", "int8", "int8gpu"],
                   help="优化器。状态显存（每参数）：adamw 8B / muon(fp16 动量) 2B / fp16mom 4B "
                        "/ adafactor 因子化 O(n+m) / int8 约 1-2B。"
                        "★ 注意：非 adamw 时 checkpoint 的优化器状态暂不支持续训（会明确拒绝）")
    g.add_argument("--opt-state", dest="opt_state", default=None,
                   choices=["fp16", "bf16", "fp32"],
                   help="--opt fp16mom 的动量存储精度。实测 fp16 会让 Adam 的二阶动量 v"
                        "（常 1e-8 量级，fp16 最小正规数 6.1e-5）下溢成 0，"
                        "第 2 步 loss 就从 11.7 炸到 135；bf16 指数范围与 fp32 相同不会下溢，"
                        "V100 上可用（存储 bf16、计算 fp32）")
    g.add_argument("--amp", default="off", choices=["off", "gemm", "mlp", "both", "tf32"],
                   help="混合精度。gemm=仅 GEMM 输入 fp16（参数/激活/梯度仍 fp32）；"
                        "mlp=MLP 内激活存 fp16（省带宽，带梯度缩放）；both=两者都开；"
                        "tf32=只把 cuBLAS 切到 TF32 math mode（tensor core，**0 次 cast "
                        "发射**，Ampere+ 才有）。"
                        "★ 实测：V100 上 fp16 张量核峰值 125 TFLOPS vs fp32 15.7，"
                        "但小配置下 cast 开销可能盖过收益，必须在目标配置上实测。"
                        "★ 2026-11-21（5090/seq512/12L）：`gemm` 要付 985 次 cast 发射"
                        "（11.6 ms/步）⇒ 比 off 慢 6%%；`tf32` 没有 cast ⇒ 比 off 快 2~3%%。")
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
    g.add_argument("--ce-chunk", type=int, default=0, dest="ce_chunk",
                   help="head+CE 分块的行数（0 = 不分块，与以前逐位一致）。"
                        "每块过 `_checkpoint_layer` ⇒ 峰值只留 1 块的 logits"
                        "（seq2048 整块是 2047x96000x4B ≈ 786 MB）。"
                        "数值已验证：chunk=64/128/256 的 gnorm d ≤ 5.7e-06、loss d ≤ 6.7e-06。"
                        "注意：`_checkpoint_layer` 的多次并列调用 bug 修好后才可用"
                        "（2026-11-21 修复；修之前它会丢 k-1 份梯度）。"
                        "也认环境变量 TG_CE_CHUNK。")
    g.add_argument("--seed", type=int, default=None,
                   help="不给则 42；续训时自动沿用 checkpoint 里的 seed")
    g.add_argument("--policy", default=None,
                   choices=["standard", "speed", "extreme", "memory"],
                   help="ORV 执行档位（显存/速度取舍，见 T.set_policy）")
    g.add_argument("--fp16-gemm", action="store_true",
                   help="打开 fp16 GEMM（V100 实测慢 16.7%% 且多占 3GB，仅复现用）")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--dp", type=int, default=0, metavar="N",
                   help="数据并行 world 大小（0 = 读 TGAI_DP_WORLD，默认 1 = 单卡）。"
                        "rank 仍由 TGAI_DP_RANK 指定，且每个进程必须只看见一张卡"
                        "（CUDA_VISIBLE_DEVICES=0 / 1）。归约后端由 TG_DP_BACKEND="
                        "host|nccl|auto 选（默认 host）。world=1 时本开关无任何影响")
    # ---- 存档 ----
    g = ap.add_argument_group("checkpoint")
    g.add_argument("--out", default=None,
                   help="checkpoint 根目录（写 step_%%06d 子目录）；不给则只训练不存档")
    g.add_argument("--ckpt-every", type=int, default=0, help="每 N 步存一次（0=只在结束时存）")
    g.add_argument("--keep", type=int, default=2, help="只保留最近 K 个 checkpoint")
    g.add_argument("--resume", default=None,
                   help="checkpoint 目录（或 --out 根目录，取最新的）")
    g.add_argument("--allow-lossy-int8", dest="allow_lossy_int8",
                   action="store_true",
                   help="允许 --opt int8gpu 这条**实测会发散**的有损路径（仅供实验）")
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


def _gpu_query_index():
    """`_vram_peak` 该看哪张卡：CUDA_VISIBLE_DEVICES 的第一个。

    ★ 为什么必须这么改（2026-11-21）：`nvidia-smi --query-gpu=memory.used` 会按
    PCI 顺序列出**所有**卡，旧代码取 `splitlines()[0]` ⇒ 双卡时 rank1 一直在报
    **rank0 那张卡**的显存。"每卡峰值显存"这一列会变成两次同一个数。
    解析失败就返回 None（退回旧行为：第一张）。
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    if not first:
        return None
    try:
        return int(first)
    except ValueError:
        # CUDA_VISIBLE_DEVICES 允许写 GPU UUID；此时无法用序号查询，退回第一张
        return None


def _nvidia_used_mib():
    idx = _gpu_query_index()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if idx is not None:
        cmd.append("--id=%d" % int(idx))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _vram_peak(stop, base, out):
    """后台采样 nvidia-smi 的显存峰值（池高水位之外的独立证据）。"""
    peak = base if base is not None else 0
    idx = _gpu_query_index()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if idx is not None:
        cmd.append("--id=%d" % int(idx))
    while not stop.is_set():
        try:
            o = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    out["peak_mib"] = peak


def _expand_cfg(a):
    cfg = dict(d=a.d, d_ff=(a.d_ff or 4 * a.d), E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=a.vocab, seq=a.seq)
    if getattr(a, "tie_embeddings", False):
        # emb 与 head 共享**同一个张量**（与 `v3_train_cli --tie-embeddings` 同义）。
        # ★ 数据并行重叠归约必须在 tied 下和「反向结束后一次性归约」逐位一致：
        #   这个开关就是用来做那条对拍的（见 dp_overlap 模块文档的 tied 一节）。
        cfg["tie"] = True
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
    elif a.opt == "int8gpu":
        # GPU 融合的量化 AdamW：量化/更新/反量化全在显存里，一次发射。
        # 默认 state="m8v16"：m 用 int8、v 用 bf16（≈3.06 字节/参数，安全）。
        # 纯 int8（≈2.125 字节/参数）**实测会发散**，必须显式 --allow-lossy-int8：
        #   4 层/20 步：step5 gnorm 405、step19 910（v 在 block 内被线性量化压成 0）
        #   bnb 能用全 int8 是因为用 256 项 quantile 非线性量化表。
        from tgai_go.train.int8_adamw_gpu import Int8AdamWGpu
        st = "int8" if getattr(a, "allow_lossy_int8", False) else "m8v16"
        opt = Int8AdamWGpu(params, lr=float(a.lr),
                           betas=(float(a.beta1), float(a.beta2)),
                           eps=float(a.eps), weight_decay=float(a.wd),
                           block=64, state=st)
        print("[int8gpu] state=%s，动量 %.3f 字节/参数（bf16 4.0，fp32 8.0）"
              % (st, opt.bytes_per_param()), flush=True)
    elif a.opt == "fp16mom":
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        # ★ 2026-11-19 修「默认值陷阱」：`fp16` 存二阶动量必然下溢 ——
        #   v 常为 1e-8 量级，而 fp16 最小正规数 6.1e-5，开根后 denom≈eps，
        #   步长被放大几个数量级（实测第 3 步 loss 11.6→94.6，末值 275）。
        #   bf16 与 fp32 同指数范围、同样 2 字节，不会下溢 ⇒ 默认改用 bf16。
        #   要强制 fp16 必须显式传 --opt-state fp16，并会打印警告。
        st_ff = str(getattr(a, "opt_state", None) or "")
        if not st_ff:
            st_ff = "bf16"
        if st_ff == "fp16":
            print("[fp16mom] 警告：--opt-state fp16 会让二阶动量下溢，"
                  "历史上这条路会让 loss 发散。建议 bf16（同样 2 字节）或 fp32。",
                  flush=True)
        opt = Fp16MomAdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),
                           eps=float(a.eps), weight_decay=float(a.wd),
                           state=st_ff)
        print("[fp16mom] state=%s，动量 %.1f 字节/参数" % (st_ff, opt.bytes_per_param()),
              flush=True)
    elif a.opt == "adafactor":
        opt = T.optim.Adafactor(params, lr=float(a.lr), weight_decay=float(a.wd))
    elif a.opt == "int8":
        from tgai_go.train.int8_adamw import Int8AdamW
        opt = Int8AdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),
                        eps=float(a.eps), weight_decay=float(a.wd))
    else:
        raise SystemExit("未知的 --opt=%r。修法：用 adamw / fp16mom / adafactor / int8"
                         % (a.opt,))
    # ★ 梯度裁剪的缩放**融合进优化器内核**（只有 int8gpu 的 Int8AdamWGpu 支持）：
    #   defer 模式不物理缩放 6.8 GB 梯度，只返回 (norm, scale)，由内核读梯度时乘。
    #   其它优化器一律保持旧的物理缩放路径（拿到的仍是 float，行为不变）。
    _defer_clip = (a.opt == "int8gpu")
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
    # tf32: 只切 cuBLAS 的 TF32 math mode —— tensor core，但**没有任何 cast 发射**。
    #       2026-11-21 实测（5090 / seq512 / 12L / 848M / int8gpu）：
    #       off 0.1389 / gemm 0.1472（+6%，985 次 cast 发射 = 11.6 ms/步）/
    #       tf32 0.1359（−2.2%，峰值显存不变 11.22 GiB，20 步 loss 与 fp16 同轨迹）。
    #       为什么 TF32 能赢：这台机器是**主机受限**，一次内核发射值 11.7 µs，
    #       fp16 的 cast 账单比它买到的 GPU 时间还贵；TF32 不付这笔账。
    _AMP = {"gemm": a.amp in ("gemm", "both"), "mlp": a.amp in ("mlp", "both"),
            "tf32": a.amp == "tf32"}
    if _AMP["tf32"]:
        try:
            T.amp_on(dtype="tf32")
            print("[amp] tf32 math mode on（0 次 cast 发射）；state=%r"
                  % (T.amp_state(),), flush=True)
        except Exception as _e:
            raise SystemExit(
                "[amp] --amp tf32 在本机不可用：%r\n"
                "原因：TF32 需要 Ampere(sm_80)+；V100/P100 没有。"
                "修法：改用 --amp fp16 路径（--amp gemm）或 --amp off" % (_e,))
    if _AMP["gemm"]:
        # ★ 2026-11-20：这里**刻意**写死 fp16，并显式告诉用户 TG_BF16=1 被忽略了。
        #   为什么不是 bug 修一修就能开：`amp_api._set()` 对 fp16 分支会主动把
        #   `_bf16_state()["on"]` 置 False，所以 TG_BF16=1 在这条路径上**必然无效**
        #   （实测 amp_state() 一直是 {'fp16': True, 'bf16': False}）。
        #   而 bf16 GEMM 的权重影子缓存失效由 `_bf16_epoch_bump()` 负责，它只挂在
        #   `_Optim` 家族（`T.optim.AdamW` / opt_extra）的 step 上 ——
        #   `Int8AdamWGpu.step()`（--opt int8gpu，本基准/生产真用的那个）**没有调用它**
        #   ⇒ 直接放开 bf16 会用到上一步的 bf16 权重，梯度静默错掉（与 fp16 那次
        #   `grad_norm 12.49 → 145775` 是同一类事故）。
        #   所以在把 `_bf16_step_boundary()` 接进 Int8AdamWGpu 并做过数值对拍之前，
        #   这里只做"不再静默"：行为不变，但把原因写进日志。
        if _bf16_requested():
            print("[amp] 注意：TG_BF16=1 已被显式忽略 —— 本路径固定 fp16 GEMM；"
                  "bf16 需要先给 Int8AdamWGpu.step 接上 _bf16_step_boundary()"
                  "（否则影子权重会跨步变旧，梯度静默错）。state 会显示 fp16=True/bf16=False。",
                  flush=True)
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
    stop_ev = threading.Event()
    prof = {}
    th = threading.Thread(target=_vram_peak, args=(stop_ev, base_mib, prof), daemon=True)
    th.start()

    # ── 数据并行（可选）：--dp N 或 TGAI_DP_WORLD>1 时启用 ──
    #   契约（见 dist.DataParallel 的 docstring）：
    #     * 建好模型后先 broadcast 一次，保证各 rank 权重逐位相同；
    #     * 每个 rank 吃**不同的**样本（这里用"跳过前 rank 个 batch"的简单做法，
    #       多读 world-1 份数据换取实现简单、不碰语料迭代器的内部状态）；
    #       顺序上等价于「单卡 --batch world」按 rank 连续切分；
    #     * 反向之后、裁剪之前做 all_reduce（默认**取平均**）⇒ 与单卡语义一致；
    #       要让等效 batch 变大，按 world 放大 lr。
    _dp = None
    try:
        _dp_env = int(os.environ.get("TGAI_DP_WORLD", "1"))
    except Exception:                                    # noqa: BLE001
        _dp_env = 1
    _dpworld = int(a.dp) if int(a.dp) > 0 else _dp_env
    if _dpworld > 1:
        _rank_env = os.environ.get("TGAI_DP_RANK")
        if _rank_env is None or str(_rank_env).strip() == "":
            raise SystemExit(
                "数据并行需要每个 rank 显式给 TGAI_DP_RANK（当前 --dp %d）" % _dpworld +
                "。修法：两个进程分别设 TGAI_DP_RANK=0 / TGAI_DP_RANK=1，并且"
                " CUDA_VISIBLE_DEVICES 分别只暴露一张卡（0 / 1），"
                "TGAI_DP_JOB 两边必须一致")
        try:
            _rank_v = int(_rank_env)
        except (TypeError, ValueError):
            raise SystemExit("TGAI_DP_RANK=%r 不是整数" % (_rank_env,) +
                             "。修法：设成 0..%d" % (_dpworld - 1))
        if not (0 <= _rank_v < _dpworld):
            raise SystemExit("TGAI_DP_RANK=%d 越界（world=%d）" % (_rank_v, _dpworld) +
                             "。修法：rank 必须落在 [0, world) 内")
        from . import dist as _dist_mod
        _dp = _dist_mod.DataParallel(world=_dpworld)
        _dp.broadcast(params)                # ← 失败（含 NCCL 不可用）在这里就报
        if _dp.rank == 0:
            print("[dp] world=%d rank=%d job=%s dir=%s backend=%s"
                  % (_dp.world, _dp.rank, _dp.job, _dp.dir, _dp.backend), flush=True)
    _is_main = (_dp is None) or (_dp.rank == 0)
    _prefix = "" if _is_main else "[r%d] " % (_dp.rank,)
    # ★ 只有 rank0 写 train_log.jsonl：两个进程同时 append 同一个文件会交错出
    #   半行 JSON（实测按 8 KiB 缓冲写时更明显）。其余 rank 写**带后缀**的文件，
    #   这样「world=2 的 loss 轨迹」可以取各 rank 的均值与单卡 batch=2 对拍。
    if a.out and _is_main:
        log_path = os.path.join(a.out, "train_log.jsonl")
    elif a.out and _dp is not None:
        log_path = os.path.join(a.out, "train_log.r%d.jsonl" % int(_dp.rank))
    else:
        log_path = None
    log_f = open(log_path, "a", encoding="utf-8") if log_path else None

    hist_tail = []
    losses, secs = [], []
    comm_hist = []                 # 每步 all_reduce 的墙钟（world=1 时全是 None）
    t_run0 = _now()
    step = start_step - 1
    aborted = None
    try:
        for step in range(start_step, stop_step):
            lr = sched.step()
            t0 = _now()
            if _dp is not None:
                # 每个 rank 取不同的样本：先跳掉 rank 份，取自己那份，再跳掉后面剩下的。
                for _ in range(_dp.rank):
                    corpus.take(int(a.batch))
                ids, labels, seg = corpus.take(int(a.batch))
                for _ in range(_dp.world - 1 - _dp.rank):
                    corpus.take(int(a.batch))
            else:
                ids, labels, seg = corpus.take(int(a.batch))
            opt.zero_grad()
            _t_fwd = _now()
            loss, fmeta = forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                                       top_k=int(a.top_k), shared=(not a.no_shared),
                                       use_seg=(not a.no_seg),
                                       ce_chunk=_ce_chunk_rows(
                                           getattr(a, "ce_chunk", 0)))
            _fwd_s = _now() - _t_fwd
            lv = _scalar(loss)
            if not _is_finite(lv):
                aborted = ("loss", lv, step, ids, labels, seg)
                break
            # ★ 数据并行「反向即归约」：必须在 backward **之前**装钩子（钩子挂在
            #   `Tensor.backward` 的叶子 .grad 写入点，见 dp_overlap 模块文档）。
            #   world<=1 / TG_DP_OVERLAP=0 / 图里有梯度检查点 / 已经有 .grad …… 时
            #   返回 None，走原来的「反向结束后一次性 all_reduce」路径（零行为变化）。
            _ov = None
            if _dp is not None and _overlap_enabled():
                try:
                    _ov = _dp.begin_overlap(params, order=_overlap_order(P, params))
                except Exception as _ovexc:              # noqa: BLE001
                    # 重叠归约是**可选优化**：初始化失败只该退回老路径，不该把训练搞崩。
                    # 但必须**响亮**（打印 + 写进 stats），不允许静默降级。
                    print("%s[dp] WARN 重叠归约初始化失败（%r）⇒ 本步退回不重叠路径"
                          % (_prefix, _ovexc), flush=True)
                    _dp.abort_overlap()
                    _ov = None
            _t_bwd = _now()
            try:
                loss.backward()
            except BaseException:
                # 反向中途异常：会话可能已经把部分桶发了出去，必须拆掉传输/钩子，
                # 否则下一步 begin_overlap 会报「上一个会话没结束」。
                if _ov is not None:
                    _dp.abort_overlap()
                _ov = None
                raise
            _bwd_s = _now() - _t_bwd
            del loss
            # ★ 必须在裁剪**之前**归约：否则 gnorm 只是本 rank 的范数，
            #   裁剪阈值在各 rank 上会不一致，梯度就不同步了。
            _comm_s = None
            if _dp is not None:
                _t_r = _now()
                _st = None
                if _ov is not None:
                    # 反向结束时把剩下的桶补发、等齐、写回；返回 None 表示会话在
                    # 反向开始时已安全退回（原因在 dp.overlap_stats["reason"]）。
                    _st = _dp.finish_overlap()
                    _check_overlap_stats(_st, _dp.world)
                if _st is None:
                    _dp.all_reduce_grads(params)
                else:
                    # 任务要求的 stats：暴露时间 / 重叠时间 / 桶数 / pending 是否归零
                    if (os.environ.get("TG_DP_TIMING")
                            or (int(a.log_every) > 0
                                and (step - start_step) % int(a.log_every) == 0)):
                        print("%s[dp-ov] exposed=%.2fms comm=%.2fms overlapped=%.2fms "
                              "buckets=%d pending_zeroed=%s zero_filled=%d late=%d "
                              "bwd=%.4fs open=%.2fms mode=%s params=%d deduped=%d"
                              % (_prefix, _st.get("exposed_ms", -1.0),
                                 _st.get("comm_ms", -1.0), _st.get("overlapped_ms", -1.0),
                                 int(_st.get("buckets", 0)),
                                 str(_st.get("pending_zeroed")),
                                 int(_st.get("zero_filled", 0)),
                                 int(_st.get("late_buckets", 0)), _bwd_s,
                                 float(_st.get("open_ms", -1.0)),
                                 str(_st.get("mode")), int(_st.get("params", 0)),
                                 int(_st.get("deduped", 0))), flush=True)
                        if os.environ.get("TG_DP_OV_DEBUG") and _st.get("bucket_comm_ms"):
                            print("%s[dp-ov-dbg] per_bucket_ms=%s"
                                  % (_prefix, _st.get("bucket_comm_ms")), flush=True)
                _comm_s = _now() - _t_r
                if os.environ.get("TG_DP_TIMING") and _st is None:
                    # 老路径才打打包/带宽统计（重叠路径的 exposed/comm 已在上面打印）
                    _nred = sum(1 for q in params if getattr(q, "grad", None) is not None)
                    _bytes = sum(int(q.grad._gpu.n) * 4 for q in params
                                 if getattr(q, "grad", None) is not None)
                    print("[dp] reduce %.4f s  grads=%d  bytes=%.2f GiB  => %.1f GB/s"
                          "[%s]"
                          % (_comm_s, _nred, _bytes / float(1 << 30),
                             (_bytes / _comm_s) / 1e9 if _comm_s > 0 else 0.0,
                             _dp.backend), flush=True)
            _t_opt = _now()
            gscale = 1.0
            if float(a.grad_clip) > 0:
                if _defer_clip:
                    # 不物理缩放；scale 交给 opt.step 在内核里乘（省一趟 读+写）。
                    gnorm, gscale = T.clip_grad_norm_(
                        params, float(a.grad_clip), defer_scale=True)
                    gnorm = float(gnorm)
                    gscale = float(gscale)
                else:
                    gnorm = float(T.clip_grad_norm_(params, float(a.grad_clip)))
            else:
                gnorm = 0.0
            if not _is_finite(gnorm):
                aborted = ("grad_norm", gnorm, step, ids, labels, seg)
                break
            if _defer_clip:
                opt.step(grad_scale=gscale)
            else:
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
            _opt_s = _now() - _t_opt
            dt = _now() - t0
            if os.environ.get("TG_DP_TIMING"):
                print("%s[phase] fwd=%.4fs bwd=%.4fs opt=%.4fs total=%.4fs rest=%.4fs"
                      % (_prefix, _fwd_s, _bwd_s, _opt_s, dt,
                         dt - _fwd_s - _bwd_s - _opt_s), flush=True)
            losses.append(lv)
            secs.append(dt)
            comm_hist.append(_comm_s)
            hist_tail.append([int(step), round(lv, 8)])
            if len(hist_tail) > 64:
                del hist_tail[:-64]
            tok = int(a.batch) * int(cfg["seq"])
            rec = {"step": int(step), "loss": lv, "lr": float(lr), "gnorm": gnorm,
                   "sec": dt, "tok": tok, "tok_s": tok / dt if dt > 0 else 0.0,
                   "comm_s": _comm_s,
                   "consumed": int(corpus.consumed),
                   "peak_live": int(rt.pool_stats()["peak_live_bytes"]),
                   "n_valid": fmeta["n_valid"]}
            if log_f is not None:
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_f.flush()
            if int(a.log_every) > 0 and ((step - start_step) % int(a.log_every) == 0
                                        or step == stop_step - 1):
                ps = rt.pool_stats()
                print("%sstep %6d/%d loss=%.6f lr=%.3e gnorm=%9.3f %.3fs tok/s=%8.1f "
                      "peak_live=%.2fGiB consumed=%d%s"
                      % (_prefix, step, stop_step, lv, lr, gnorm, dt, rec["tok_s"],
                         _gib(ps["peak_live_bytes"]), corpus.consumed,
                         "" if _comm_s is None else " comm=%.3fs" % _comm_s),
                      flush=True)
            # ---- 周期性存档（★ 只有 rank0 写盘）----
            if (a.out and _is_main and int(a.ckpt_every) > 0
                    and (step + 1) % int(a.ckpt_every) == 0):
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

    # ★ 通信器在**所有 rank 都跑完训练之后**统一关闭（ncclCommDestroy 内部会等齐）：
    #   提前关会让还在最后一步 all_reduce 的 rank 挂住。
    if _dp is not None:
        _dp.close()

    # ---- NaN/Inf：停下 + 存现场（绝不静默继续）----
    if aborted is not None:
        kind, value, bad_step, ids, labels, seg = aborted
        # ★ 每个 rank 都有自己的现场；文件名带 rank 后缀，避免 rank1 覆盖 rank0
        _nan_suf = "" if _is_main else "_r%d" % int(_dp.rank)
        dump = os.path.join(a.out or ".", "nan",
                            "nan_step_%06d%s" % (int(bad_step), _nan_suf))
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
            print("%s[nan] WARN 现场落盘失败：%s" % (_prefix, e), flush=True)
        print("%s[nan] ABORT step=%d %s=%r dump=%s"
              % (_prefix, bad_step, kind, value, dump), flush=True)
        if _is_main:
            print("RESULT " + json.dumps({"status": "aborted", "kind": kind,
                                          "value": value, "step": int(bad_step),
                                          "dump": dump},
                                         ensure_ascii=False), flush=True)
        return 2

    # ---- 收尾存档（★ 只有 rank0 写盘）----
    final_dir = None
    if a.out and not a.no_final_save and _is_main:
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
    _comm_vals = [float(x) for x in (rec_comm for rec_comm in comm_hist) if x is not None]
    _world = 1 if _dp is None else int(_dp.world)
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
        "batch": int(a.batch),
        "world": _world,
        "global_batch": int(a.batch) * _world,
        "dp": (_dp.stats() if _dp is not None else None),
        "comm_s_median": float(np.median(_comm_vals)) if _comm_vals else None,
        "comm_s_min": float(np.min(_comm_vals)) if _comm_vals else None,
        "comm_s_max": float(np.max(_comm_vals)) if _comm_vals else None,
        "comm_frac": (float(np.median(_comm_vals)) / float(np.mean(steady))
                      if _comm_vals and steady else None),
        "gpu_index": _gpu_query_index(),
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
    if _is_main:
        print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "losses"},
                                     ensure_ascii=False), flush=True)
        print("losses %r" % (res["losses"],), flush=True)
        if a.json_out:
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
    else:
        print("%s[done] step=%d world=%d backend=%s"
              % (_prefix, int(step) + 1, _world, _dp.backend), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
