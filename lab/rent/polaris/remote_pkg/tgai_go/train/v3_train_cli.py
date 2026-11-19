# -*- coding: utf-8 -*-
"""V3 架构在 ORV 上的**真实训练步**（完整 AdamW、权重真更新）——包内可执行入口。

结构 = V3 的骨架（pre-norm 残差 + GlobalAttention + MoE(E 专家, top_k=1|2) + SiTU-GLU
+ 可选共享专家）：

```
x = emb[ids]
for layer: h = x + attn(rmsnorm(x));  x = h + moe(rmsnorm(h))
loss = CE(head(rmsnorm(x)), targets);  loss.backward(); opt.step()
```

用 `scatter_add` 做专家分派（每个专家只算自己的 token）。
权重随机初始化（同构对比用；V3 的 ORV 格式权重尚无导出器）。

指标：峰值显存（nvidia-smi 采样 + 运行时池高水位）、每步耗时、tok/s、逐 step loss。

## 为什么放进包里（而不是只留 tools/）

远端 V100 原先只能通过中继 `python tgai_ops.py v100py '<code>'` 执行代码，而实测**单条
命令只能可靠承载约 1 KB 的 base64**（~600 B 成功；5.7 KB / 9 KB / 14.8 KB 全失败，见
`docs/V3_V100_MEASURED.md` §8.1）——几 KB 的训练脚本永远传不过去。做成 `tgai_go.train`
的模块后，它随 `pip install` 到达远端，直接：

    python -m tgai_go.train.v3_train_cli --d 1024 --layers 20 --E 4 --seq 256 \\
        --steps 2 --top_k 2 --d_ff 2816 --vocab 96000

即使后来有了 scp 直传（2026-11-16 验证可用），进包仍然有独立价值：远端不用手工传文件、
`python -m` 一行可跑、脚本与引擎**同源同版本**（不会出现"脚本 0.9.83 / 引擎 0.9.62"的错配）。

## 与 `tools/v3_orv_train.py` 的关系

本文件是**唯一实现**。`tools/v3_orv_train.py` 现在是薄封装（准备 sys.path +
re-export `build` / `params_of` / `main`）—— 因为 `tools/probe_*.py` 与
`tools/v3_orv_train_ckpt*.py` 等 11 个探针脚本都 `from v3_orv_train import build, params_of`。
**不要两边各改一份**（两份实现迟早漂移）。

## ★ `--fp16-gemm` 为什么默认关（实测，反直觉）

V100 上同配置（d=1024 / 20 层 / E4 / seq256，双方 fp32 对拍）：

| 设置 | s/step | tok/s | 池峰值 |
| --- | ---: | ---: | ---: |
| `fp16_gemm = False` | **0.4347** | **588.9** | **23.07 GB** |
| `fp16_gemm = True` | 0.5073 | 504.6 | 26.12 GB |

**慢 16.7%、多占 3 GB**（见 `docs/V3_V100_MEASURED.md` §2）。
推断原因：权重影子按「代」失效，而 AdamW 每步都更新权重 ⇒ 每步都要把 1.3B 参数重新 cast，
再加 fp16 缓冲。所以这个开关**默认关**，只用于复现实验，不要当加速手段。

真尺寸另有一条实测：`d_ff=2816 / vocab=96000`、20 层 / seq256 / steps2 时
**0.5927 s/step、431.9 tok/s、池峰值 18.13 GB** —— 比 d_ff=4096 的配置慢，
因为 96K 词表的 lm_head 更大（不是实现退化）。
2026-11-16 从**本入口**（`python -m tgai_go.train.v3_train_cli ...`，V100 / 0.9.83）复跑同一命令：
0.5966 s/step、429.1 tok/s、池 `peak_live_bytes=19,825,109,504`（18.46 GiB）—— 与上一致到 1% 内。

## ★ 真 V3 的 3:1 混合注意力（`--csa N`，默认关）

`--csa 4`（别名 `--attn-global-every 4`）把注意力换成 V3 的真实结构：

* **层判定**逐字照 `model_v3.py:215-222` 的 `(i % N) == (N - 1)`（`i` 是 0-based）
  ⇒ `N=4` 命中 0-based 的 3/7/11/15/19 = **1-based 第 4/8/12/16/20 层**为 Global，
  其余 **15 层**为 CSA（正好 3:1）。**不是** `i % 4 == 0`（那会选到第 1/5/9/13/17 层）。
* **Global 层**：`n_kv = n_q`（V3 的 `n_kv_heads=0` ⇒ MHA，不是 GQA-2）；
  θ=500000，与 CSA 层**共用同一张 RoPE 表**（对应 V3 的 `hetero_rope=False`）。
* **CSA 层**：`v3_csa.csa_forward(ln1(x), w["csa"], plan, cos, sin)`，权重由
  `v3_csa.init_weights` 建（`params_of` 会把权重 dict 摊平进优化器）。
  残差 / LayerNorm 结构与 `TransformerBlockV3` 一致：
  `h = x + attn(ln1(x))`、`x = h + moe(ln2(h))`。
* `plan`（掩码/索引）与 `rope_tables`（全表）**建模型时算一次**，15 个 CSA 层与每一步复用。
* 默认（`--csa 0`）：`layer_kinds` 全返回 `'global'`，参数集合、逐 step 打印、RESULT 里的
  cfg 都与改动前**逐位一致**（同一 rng 种子对拍 323 个参数的形状+数值指纹全等）。

### 实测（2026-11-16，V100-32G / tgai-orv 0.9.83 / fp32 / B=1 / `--top_k 2`）

`d=1024 L=20 E=4 d_ff=2816 vocab=96000 seq=256`（除 seq 列另有标注）。稳态 = step 1+
（step 0 两侧都含 5.6~7.0 s 的内核 JIT 预热，不计入）：

| seq | n_q | 注意力 | s/step（稳态） | tok/s | 池 `peak_live_bytes` |
| ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 16 | 20 Global（改动前） | **0.5175** | 494.7 | 19,489,722,880（18.15 GiB） |
| 256 | 16 | 15 CSA + 5 Global | **0.5475** | 467.6 | 20,095,954,432（18.72 GiB） |
| 512 | 16 | 20 Global | 0.59 | 867.8 | 20,972,131,840（19.53 GiB） |
| 512 | 16 | 15 CSA + 5 Global | 0.62 | 825.8 | 21,951,052,288（20.44 GiB） |
| 1024 | 16 | 20 Global | 0.77 | 1329.9 | 23,938,126,336（22.29 GiB） |
| 1024 | 16 | 15 CSA + 5 Global | 0.79 | 1296.2 | 26,050,324,480（24.26 GiB） |

⇒ **步时代价随 seq 递减**（+5.8% → +5.1% → +2.6%）：全局分支 O(S²) 涨，CSA 的压缩分支只涨
O(S·S/c) + 固定 32 个 chunk 的局部窗。**显存代价递增**（+0.57 → +0.91 → +1.97 GiB），因为
`v3_csa` 会物化 score（它自己 docstring 的「已知限制③」，用 mem-efficient SDPA 可解，未做）。
**全程没有 OOM**（最大 24.26 GiB / 32 GB）。

任务给定命令（未给 `--n_q` ⇒ 缺省 4 ⇒ d_k=256，**不是** V3 的 16 头）同条件对照：
改动前 0.46 s/step / 19,825,109,504 B，`--csa 4` 0.54 s/step / 20,018,948,608 B。

复现（远端 V100，d 1024 / 20 层）：

    python -m tgai_go.train.v3_train_cli --d 1024 --layers 20 --E 4 --seq 256 \\
        --steps 5 --top_k 2 --d_ff 2816 --vocab 96000 --n_q 16 --csa 4

## ★ 与 PyTorch 参考的 6 项架构对齐（**默认全关**；`--faithful-v3` 一次全开）

参考实现：`D:\\TGAI-V3-1.5B\\03_src\\model_v3.py`（V100 上的官方副本）。本机等价快照
`_src15b_current/model_v3.py` 行号差 +25，差异只有 2026-10-19 的提速改写**与插桩**，
6 项的语义逐字相同（已 diff 核对）。

| 开关 | 参考实现（默认） | 本文件默认 | 具体行为 |
| --- | --- | --- | --- |
| `--tie-embeddings` | `tie_embeddings=True`（`model_v3.py` 的 `TGAIV3Model.__init__`：`self.lm_head.weight = self.embed.weight`）| 独立 emb/head | `P["head"]` **就是** `P["emb"]` 同一个张量；`params_of` 按 `id()` 去重（否则 AdamW 每步会把同一张量更新两次）|
| `--moe-post-norm` | `moe_post_norm=True`（`MoEv3.__init__` 的 `self.post_norm`）| 无 | routed 聚合后、**加共享专家之前**做一次 `RMSNorm(d_model)`（`MoEv3.forward` ⑤/⑥ 的顺序）|
| `--qk-norm` | `attn_qk_norm=True`（`GlobalAttention.__init__` 的 `q_norm/k_norm`）| 无 | Global 层的 q/k 在 **RoPE 之前**各自沿 head 维 `d_k` 做 `RMSNorm`（`GlobalAttention.forward`：先 norm 后 rope，顺序不能反）|
| `--pad-id N` | `pad_token_id=256`（`_init_weights`：`m.weight[pad].zero_()`）| None | emb 的该行初始化为 0，且**该行拿不到梯度**（PyTorch `padding_idx` 语义；做法见 `_embed` 的说明）|
| `--moe-balance quantile` | `moe_balance='quantile'`（K3 QB）| `sign` | 亲和度用 **sigmoid**（不是全专家 softmax），门控在**选中集合内重新归一化**；并实现 K3 的偏置更新（`update_bias`），否则第二步起路由就与参考不同 |
| `--attn-res`（已有，本轮改成**包内直连**）| `attn_res_enabled=True` | False | BlockAttnRes 四段式层循环，直接调包里的 `v3_attnres.run_attn_res_layers`（不再依赖源码树的 `tools/`，pip 形态也能用）|

`--faithful-v3` = 上表全开（`--csa` 不在其中：3:1 混合注意力是另一维度，需要时单独给
`--csa 4`；两者**可以同时用**——本轮解除了旧的互斥，因为 AttnRes 的层循环现在按
`layer_kinds` 分派 attn，CSA 层同样适用）。

**默认路径逐位不变**：不开任何新开关时，`build()` 的参数集合、`params_of()` 的顺序、
前向表达式、逐 step 打印、RESULT 的 cfg 都与改动前**逐字一致**（同 rng 种子的参数指纹
对拍见 `docs/V3_ARCH_ALIGN_2026-11-16.md`）。

**已知仍未对齐的项（不在本轮 6 项内，不藏）**：训练损失口径（参考实现是
`logits[:, :-1]` vs `labels[:, 1:]`，本文件是 `roll(ids,-1)` 全 T 位置）、MoE 的
z-loss / 互补均衡 aux loss（参考 `aux_loss()` 以 1e-4 权重加进总损失，本文件未加）、
QK-Clip 与 dropout（参考里 `qk_clip_tau` 会改权重、dropout 默认 0）。
"""
import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np

from tgai_go import train as T
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V

__all__ = ["build", "params_of", "build_arg_parser", "main",
           "is_global_layer", "layer_kinds", "forward_logits",
           "update_moe_bias", "ARCH_ITEMS"]

# 惰性导入的 CSA 实现（见 `_csa_module`）。`--csa` 关闭时保持为 None。
_CSA = None
# 惰性导入的 BlockAttnRes 实现（见 `_attnres_module`）。`--attn-res` 关闭时保持为 None。
_ATTNRES = None

# 6 项架构对齐的「项名 → (cfg 键, argparse 目的名)」。给 `--faithful-v3` 与外部脚本
# （`lab/e2e/items_parity.py`）当**单一来源**：新加一项只改这里 + build/main。
ARCH_ITEMS = (
    ("tie_embeddings", "tie"),
    ("moe_post_norm", "moe_post_norm"),
    ("qk_norm", "qk_norm"),
    ("pad_token_id", "pad_id"),
    ("moe_balance", "moe_balance"),
    ("attn_res", "attn_res"),
)


def _attnres_module():
    """导入 `tgai_go.train.v3_attnres`（**惰性**，只在 `--attn-res` 打开时调用）。

    ★ 与 `_csa_module` 同样的理由：默认路径（不带 `--attn-res`）必须能在任何
      tgai-orv 版本上跑通，包括还没打包 `v3_attnres` 的 wheel —— 顶层 import 会让
      "不加开关"的既有命令直接 ImportError，那是真回归。
    """
    global _ATTNRES
    if _ATTNRES is not None:
        return _ATTNRES
    try:
        from tgai_go.train import v3_attnres as A
    except ImportError as e:                      # pragma: no cover - 依赖安装形态
        raise SystemExit(
            "--attn-res 需要 tgai_go.train.v3_attnres，但当前包没有它：%s\n"
            "修法：① 换用含 v3_attnres 的 tgai-orv（源码树 / 新 wheel）；"
            "② 不加 --attn-res（默认路径完全不受影响）。" % (e,))
    _ATTNRES = A
    return A


def _csa_module():
    """导入 `tgai_go.train.v3_csa`（**惰性**，只在 `--csa` 打开时调用）。

    ★ 为什么不写在模块顶层：默认路径（`--csa 0`）必须能在**任何** tgai-orv 版本上跑通，
      包括还没打包 `v3_csa` 的 wheel（远端 `D:\\orv_pip_0962` 的 0.9.83 就没有这个文件）。
      顶层 import 会让"不加 --csa"的既有命令在旧包上直接 ImportError —— 那是真回归。
      惰性导入的另一个好处是错误信息能给出修法，而不是一句 ImportError。
    """
    global _CSA
    if _CSA is not None:
        return _CSA
    try:
        from tgai_go.train import v3_csa as C
    except ImportError as e:                      # pragma: no cover - 依赖安装形态
        raise SystemExit(
            "--csa 需要 tgai_go.train.v3_csa，但当前包没有它：%s\n"
            "修法：① 换用含 v3_csa 的 tgai-orv（源码树 / 新 wheel）；"
            "② 不加 --csa（默认路径完全不受影响）。" % (e,))
    _CSA = C
    return C


def _vram_peak(stop, base, out):
    peak = base
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    out["peak"] = peak


# ═════════════════════════════════════════════════════════════════════════
# 层类型：真 V3 的 3:1 混合注意力（`--csa N`）
# ═════════════════════════════════════════════════════════════════════════
def is_global_layer(i, every):
    """第 `i` 层（**0-based**）是不是 Global 层。

    ★ 判定条件**照抄** `_src15b_current/model_v3.py:215-222`：

        def is_global_layer(self, i):
            return (i % self.attn_global_every) == (self.attn_global_every - 1)

    ⇒ `every=4` 时命中的是 0-based 的 **3/7/11/15/19**，也就是 1-based 的第
      **4/8/12/16/20** 层，等价于 `(i + 1) % 4 == 0`。
      **不是** `i % 4 == 0`（那样会选到第 1/5/9/13/17 层，把层类型整体错位一格）。
      这里不把 4 写死，任何 N 都由同一个式子推出。
    """
    every = int(every)
    if every <= 0:
        raise ValueError("attn_global_every 必须 >=1，收到 %r。"
                         "修法：--csa N 传 >=1 的整数（0 = 关闭混合注意力，"
                         "由 layer_kinds 提前拦掉，不会走到这里）" % (every,))
    return (int(i) % every) == (every - 1)


def layer_kinds(cfg):
    """逐层的注意力类型：`'global'`（GlobalAttention）或 `'csa'`（CompressedSparseAttention）。

    `cfg["csa"]`（CLI 的 `--csa N`）为 0 时**每层都是 `'global'`** ——
    这正是改动前的行为（每层都走 `v3_ops.attention_orv`），所以默认路径不受影响。
    """
    n = int(cfg["layers"])
    every = int(cfg.get("csa", 0) or 0)
    if every <= 0:
        return ["global"] * n
    return ["global" if is_global_layer(i, every) else "csa" for i in range(n)]


def _check_layer_weights(P, kinds):
    """层类型与权重必须一一对上（不符立刻炸，别等前向算出怪数字）。"""
    for i, (L, k) in enumerate(zip(P["layers"], kinds)):
        has_attn = all(x in L for x in ("wq", "wk", "wv", "wo"))
        has_csa = isinstance(L.get("csa"), dict)
        if k == "csa" and not has_csa:
            raise RuntimeError("第 %d 层应为 CSA 层，但参数里没有 csa 权重 dict。"
                               "修法：用 build() 建参数（它按 layer_kinds 分配权重），"
                               "不要手工拼 P['layers']" % i)
        if k != "csa" and not has_attn:
            raise RuntimeError("第 %d 层应为 Global 层，但缺 wq/wk/wv/wo。"
                               "修法：用 build() 建参数，别手工拼 P['layers']" % i)
        if has_attn and has_csa:
            raise RuntimeError("第 %d 层同时存在 Global 与 CSA 两套注意力权重。"
                               "修法：检查 cfg['csa'] 是否在 build 前后被改过"
                               "（权重与层类型必须同源）" % i)
    return True


def build(cfg, rng):
    """随机初始化参数（同构对比用）。返回值都是 requires_grad 的 ORV 张量。

    ## `--csa N` 打开时（真 V3 的 3:1 混合注意力）

    逐层按 `layer_kinds(cfg)` 分配权重：

    * **Global 层**（`(i % N) == N-1`）：原来的 `wq/wk/wv/wo`。真 V3 的
      `n_kv_heads=0` ⇒ `cfg["n_kv_attn"] = n_q`（MHA），与 V3 配置一致。
    * **CSA 层**：**不建** `wq/wk/wv/wo`，改成一个权重 dict
      （`v3_csa.init_weights` 的产物，键 `wq/wkv/wk/wv/wo/conv/ln/gate[/qn/kn]`）。
      dict 里全是 requires_grad 张量，`params_of` 会把它们摊平交给优化器。

    ★ 副作用（刻意）：`CSAConfig` 只在这里创建一次，并**写回 `cfg["csa_cfg"]`**，
      保证 `main()` 里的 `make_plan` / `rope_tables` 与这里的权重用的是**同一个对象**
      （形状口径只有一个来源，不会出现两份配置漂移）。调用方也可预先放一个
      `cfg["csa_cfg"]` 来覆盖默认；若它与 `cfg["d"]/cfg["n_q"]` 不自洽会立刻报错。
    """
    d, d_ff, E, L = cfg["d"], cfg["d_ff"], cfg["E"], cfg["layers"]
    n_q, n_kv = cfg["n_q"], cfg["n_kv"]
    # `--csa` 时 Global 层走 MHA（= V3 的 n_kv_heads=0）；默认仍是原来的 GQA-2。
    # 用 cfg.get 而不是直接加键 ⇒ 默认路径的 cfg 字典与改动前**逐字一致**。
    n_kv_attn = int(cfg.get("n_kv_attn", n_kv))
    d_k = d // n_q
    kinds = layer_kinds(cfg)
    every = int(cfg.get("csa", 0) or 0)
    C = csac = None
    # ★ 条件是 `every > 0` 而不是 `"csa" in kinds`：`--csa 1`（或 N > 层数）时
    #   一层 CSA 都没有，但 RoPE 表仍要用 V3 的 θ ⇒ CSAConfig 照样得建出来。
    if every > 0:
        # ★ 旧版这里禁止 `--csa` 与 `attn_res` 同时打开，理由是"AttnRes 循环假定每层都有
        #   wq/wk/wv/wo"。现在层循环按 `layer_kinds` 分派（见 `_attn_part`），CSA 层走
        #   `csa_forward`，与参考实现的 `TransformerBlockV3.attn_part` 同构 ⇒ 两者可以共存，
        #   这也正是真 V3 的配置（3:1 CSA + AttnRes 全开）。
        C = _csa_module()
        csac = cfg.get("csa_cfg")
        if csac is None:
            csac = C.CSAConfig(
                d_model=d, n_heads=n_q, n_kv_heads=0,          # V3: n_kv_heads=0 ⇒ MHA
                csa_compress=int(cfg.get("csa_compress", 4)),
                csa_latent=int(cfg.get("csa_latent", 0)),
                csa_conv_kernel=int(cfg.get("csa_conv_kernel", 3)),
                csa_local_window=int(cfg.get("csa_local_window", 0)))
        if int(csac.d_model) != int(d) or int(csac.n_heads) != int(n_q):
            raise ValueError(
                "cfg['csa_cfg'] 与 cfg 不自洽：CSAConfig(d_model=%d, n_heads=%d) vs "
                "cfg(d=%d, n_q=%d)。修法：删掉 cfg['csa_cfg'] 让 build() 自己建，"
                "或改成一致的取值" % (csac.d_model, csac.n_heads, d, n_q))
        cfg["csa_cfg"] = csac                      # 单一来源（main 从这里取）

    def p(*shape, scale=0.02):
        return T.tensor(rng.normal(0, scale, shape).astype(np.float32), requires_grad=True)

    # ★ `--pad-id`（对应 `model_v3.py::TGAIV3Model._init_weights` 的
    #   `m.weight[self.config.pad_token_id].zero_()`，config 里 `pad_token_id=256`）：
    #   参考实现是 `nn.Embedding(..., padding_idx=pad)`，该行**恒为 0**，且 PyTorch 的
    #   Embedding 反向会把该行梯度清零（`padding_idx` 语义）。
    #   这里在建表时就把该行写 0（等价于"生成后再 zero_"，但省一次整表 D2H/H2D）。
    #   ★ rng 抽样顺序不变：`rng.normal(0, 0.02, (vocab, d))` 与 `p(vocab, d)` 消耗相同。
    pad_id = cfg.get("pad_id")
    if pad_id is not None:
        pad_id = int(pad_id)
        if not (0 <= pad_id < int(cfg["vocab"])):
            raise ValueError(
                "cfg['pad_id']=%d 不在词表范围 [0,%d)。修法：--pad-id 传 < vocab 的整数"
                % (pad_id, int(cfg["vocab"])))
        _emb = rng.normal(0, 0.02, (int(cfg["vocab"]), int(d))).astype(np.float32)
        _emb[pad_id] = 0.0
        emb = T.tensor(_emb, requires_grad=True)
    else:
        emb = p(cfg["vocab"], d)
    P = {"emb": emb, "ln_f": T.tensor(np.ones(d, np.float32), requires_grad=True),
         "head": p(cfg["vocab"], d), "layers": []}
    if cfg.get("tie"):
        # 参考实现 `if config.tie_embeddings: self.lm_head.weight = self.embed.weight`
        # ⇒ **同一个 Parameter**（不是拷一份）：优化器只看到一份，梯度从两条路径累加。
        # `params_of()` 用 `id()` 去重，保证 AdamW 每步只更新它一次。
        P["head"] = P["emb"]
    for i in range(L):
        # 插入顺序与改动前**逐字一致**（ln1, ln2, router, [注意力权重], W1, W2, W3,
        # S1, S2, S3）⇒ 默认路径下 `params_of` 给出的参数顺序也完全不变。
        rec = dict(
            ln1=T.tensor(np.ones(d, np.float32), requires_grad=True),
            ln2=T.tensor(np.ones(d, np.float32), requires_grad=True),
            # router 是**独立参数**（参考实现 `nn.Linear(d_model, n_experts)`），
            # 不能每步用 cat([W1[e]…]) 现拼（既偏架构，又多拷 E×d_ff×d 个元素）
            router=p(E, d),
        )
        if kinds[i] == "csa":
            # CSA 层：整套低秩下投影 + 压缩卷积 + q/k/v/o（v3_csa.init_weights）
            rec["csa"] = C.init_weights(rng, csac)
        else:
            rec.update(wq=p(n_q * d_k, d), wk=p(n_kv_attn * d_k, d),
                       wv=p(n_kv_attn * d_k, d), wo=p(d, n_q * d_k))
        rec.update(
            # ★ 专家权重用**独立参数列表**，不用三维张量再 [e] 索引：
            #   实测 `W3d[e]` 每次都拷贝（4 MiB/次），MoE 每层索引 24 次（8 专家 × 3 权重），
            #   拷出来的副本还会被自动微分保留到反向 —— 一层约 200 MiB 的纯浪费。
            #   列表索引不产生任何 GPU 拷贝。
            W1=[p(d_ff, d) for _ in range(E)],
            W2=[p(d_ff, d) for _ in range(E)],
            W3=[p(d, d_ff) for _ in range(E)],
            S1=p(d_ff, d), S2=p(d_ff, d), S3=p(d, d_ff),     # 共享专家的 FFN（对齐 PyT 的 n_shared_experts=1）
        )
        P["layers"].append(rec)
        # ★ `--qk-norm`：Global 层多两个 per-head 的 RMSNorm γ（`d_k` 维）。
        #   参考 `GlobalAttention.__init__`：`self.q_norm = RMSNorm(self.d_k, eps)` 等。
        #   CSA 层不走这里 —— 它的 qk_norm 在 `v3_csa` 内部（`w["qn"/"kn"]`，由
        #   `CSAConfig.attn_qk_norm` 控制，默认已经是 True，见模块文档的说明）。
        if cfg.get("qk_norm") and kinds[i] != "csa":
            P["layers"][-1].update(
                qn=T.tensor(np.ones(d_k, np.float32), requires_grad=True),
                kn=T.tensor(np.ones(d_k, np.float32), requires_grad=True),
            )
        # ★ `--moe-post-norm`：routed 聚合后的 RMSNorm γ（参考 `MoEv3.post_norm`）。
        if cfg.get("moe_post_norm"):
            P["layers"][-1].update(
                post_norm=T.tensor(np.ones(d, np.float32), requires_grad=True))
        # BlockAttnRes 的参数：每层两组（attn 前 / mlp 前），各一个伪查询 w 与 RMSNorm 的 γ。
        # 对应 model_v3.py 的 `attn_res_proj[i]` / `mlp_res_proj[i]`（各含 proj.weight + norm.weight）。
        # 只在 cfg 打开 attn_res 时才建，默认不建 ⇒ 与改动前的参数集合完全一致。
        if cfg.get("attn_res"):
            P["layers"][-1].update(
                res_w_attn=p(d),
                res_gn_attn=T.tensor(np.ones(d, np.float32), requires_grad=True),
                res_w_mlp=p(d),
                res_gn_mlp=T.tensor(np.ones(d, np.float32), requires_grad=True),
            )
    _check_layer_weights(P, kinds)          # 层类型与权重必须一一对上（不符立刻炸）
    return P


def params_of(P):
    """把 `build()` 的参数字典摊平成一个列表（交给 `AdamW`）。

    ★ 三种值形态都要处理：
      * `dict` —— **CSA 层的权重 dict**（只有 `--csa` 打开才存在），值里可能含
        `None`（`attn_qk_norm=False` 时 `qn/kn` 是 None），必须跳过；
      * `list` —— 专家权重（`W1/W2/W3`，每专家一个张量，列表索引不产生 GPU 拷贝）；
      * 其余 —— 单个张量。

    ★ `--tie-embeddings` 时 `P["head"] is P["emb"]`：必须按 `id()` **去重**。
      PyTorch 的 `model.parameters()` 默认 `remove_duplicate=True`，绑定权重只出现一次；
      若这里不去重，AdamW 会把同一个张量当成两个参数 ⇒ 一步更新两次、m/v 被算两遍，
      立刻与参考实现分叉。去重不改变默认路径的顺序与内容（默认路径没有重复）。
    """
    out, seen = [], set()

    def add(t):
        if t is None or id(t) in seen:
            return
        seen.add(id(t))
        out.append(t)

    add(P["emb"])
    add(P["ln_f"])
    add(P["head"])
    for L in P["layers"]:
        for k, v in L.items():
            if isinstance(v, dict):          # CSA 权重 dict（--csa）
                for t in v.values():
                    add(t)
            elif isinstance(v, list):        # 专家权重是列表
                for t in v:
                    add(t)
            else:
                add(v)
    return out


def _opt(opts, name, default):
    """从 argparse 命名空间（或任意对象）取开关，缺省用 `default`。

    `forward_logits` 既被 `main()` 调用（传 argparse 结果），也被对拍脚本调用
    （传一个最小 namespace），所以不能假定 opts 带全部属性。
    """
    return getattr(opts, name, default) if opts is not None else default


def _sigmoid_orv(z):
    """`sigmoid` 的惰性包装（`nn_ops.sigmoid` 是组合实现，CPU/CUDA 都有）。

    惰性导入的理由同 `_csa_module`：默认路径（`moe_balance='sign'`）根本不走这里，
    不该在模块顶层多引一个依赖。
    """
    from tgai_go.train import nn_ops
    return nn_ops.sigmoid(z)


def _attention_qk_norm(x, wq, wk, wv, wo, qn, kn, cos, sin, rot_dim,
                       n_q, n_kv, causal=True, lowmem=False, flash=False):
    """`v3_ops.attention_orv` 的**逐行副本**，只在 RoPE 之前多两行 QK-Norm。

    ## 为什么复制而不是改 `v3_ops.attention_orv`

    本轮任务不允许改 `v3_ops.py`（已 GPU 验证的公共模块），而 QK-Norm 必须插在
    「投影之后、RoPE 之前」—— `attention_orv` 把 q/k 投影与 RoPE 都包在内部，没有注入口。

    ## 与参考实现的对应（`model_v3.py::GlobalAttention.forward`）

        q = q_proj(x).view(B,T,n_q,d_k).transpose(1,2)
        k = k_proj(x).view(B,T,n_kv,d_k).transpose(1,2)
        if cfg.attn_qk_norm: q, k = self.q_norm(q), self.k_norm(k)   # ★ 沿 d_k
        q = self.rope(q); k = self.rope(k)                            # ★ 之后才 RoPE

    顺序反了就成了"先转再归一化"，结果不同 —— 本实现严格照参考顺序。
    q_norm/k_norm 都是 `RMSNorm(d_k, norm_eps)`，与序列位置无关，所以 4D 张量
    `(B,H,T,d_k)` 上沿最后一维归一化即可。

    ★ RoPE 走组合实现（`v3_ops.v3_rope`），不走向量化的 `fused_rope`：数学相同
      （`x*cos + rotate_half(x)*sin`），只是多几次 elementwise。理由是本文件不该依赖
      `v3_ops._rope_table` 这种私有缓存函数（跨版本会碎）。`--qk-norm` 是忠实对齐
      路径，不是性能路径。
    """
    B = int(x.shape[0])
    Tlen = int(x.shape[1])            # 不能叫 T：会遮蔽模块别名 T（= tgai_go.train）
    n_q, n_kv, rot_dim = int(n_q), int(n_kv), int(rot_dim)
    d_k = int(wq.shape[0]) // n_q
    if int(qn.shape[0]) != d_k or int(kn.shape[0]) != d_k:
        raise ValueError(
            "QK-Norm 的 γ 形状不对：qn=%s kn=%s，但 d_k=%d。"
            "修法：用 build() 建 P['layers'][i]['qn'/'kn']（形状 (d//n_q,)），"
            "不要手工拼参数字典" % (tuple(qn.shape), tuple(kn.shape), d_k))
    if rot_dim != d_k:
        raise ValueError(
            "QK-Norm 路径只支持全维 RoPE（rot_dim=%d != d_k=%d）。"
            "修法：partial RoPE 需要把 q/k 的旋转段与直通段分开处理，"
            "本实现未覆盖；请用全维 RoPE（rope_partial=1.0）" % (rot_dim, d_k))

    q = (x @ wq.transpose(0, 1)).reshape(B, Tlen, n_q, d_k).transpose(1, 2)
    k = (x @ wk.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)
    v = (x @ wv.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)

    q = V.rmsnorm_orv(q, qn)
    k = V.rmsnorm_orv(k, kn)

    c = cos.reshape(1, 1, Tlen, rot_dim)
    s = sin.reshape(1, 1, Tlen, rot_dim)
    q = V.v3_rope(q, c, s)
    k = V.v3_rope(k, c, s)

    if n_kv != n_q:
        # GQA 的**交错**复制（与 attention_orv / 参考实现一致）
        rep = n_q // n_kv
        k = T.cat([k[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)
        v = T.cat([v[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)

    if flash:
        from tgai_go.train import v3_flash_op as _FO
        o = _FO.flash_attn(q, k, v, causal=causal)
        return o.transpose(1, 2).reshape(B, Tlen, n_q * d_k) @ wo.transpose(0, 1)
    if lowmem:
        out = T.functional.scaled_dot_product_attention(q, k, v, causal=causal,
                                                        lowmem=True)
    else:
        out = T.functional.scaled_dot_product_attention(q, k, v, causal=causal)
    out = out.transpose(1, 2).reshape(B, Tlen, n_q * d_k)
    return out @ wo.transpose(0, 1)


def _route_quantile(hn, router_w, top_k, bias_np=None):
    """K3 Quantile Balancing 的**路由**（`moe_balance='quantile'`）。

    与参考实现 `model_v3.py::MoEv3.forward` ①②③ 的 quantile 分支逐条对应：

        s_all = sigmoid(router(x))                     # ★ K3 Eq.13：逐专家独立 sigmoid
        sel   = s_all + b                              # b 只影响**选择**
        topi  = top-k(sel)
        gsel  = s_all[topi]
        gate  = gsel / clamp_min(gsel.sum(-1, keepdim), 1e-9)   # ★ 选中集合内重归一化

    与 `v3_moe.route_topk`（softmax 口径）的差别正好是这两处：亲和度函数、门控归一化。
    选择用运行时内核 `moe_topk`（CUDA 上没有 `T.topk`），取门控用 `moe_gather_pairs`，
    反向用 `moe_scatter_pairs` 散回 —— 与 `route_topk` 同一套原语，只是 `s_all` 换了。

    返回 `(eids, gates, s_all)`：`eids` 是 (N,k) int64 主机下标（不参与求导），
    `gates` 是 (N,k) 带梯度的**归一化后**门控，`s_all` 是 (N,E) 亲和度（供偏置更新）。
    """
    rt = T._cuda()
    if getattr(hn, "device", "cpu") != "cuda":
        raise SystemExit(
            "--moe-balance quantile 需要 CUDA：门控选择走运行时内核 `moe_topk`，"
            "CPU 上没有等价原语。修法：在 CUDA 设备上跑（本 CLI 本来就要求 CUDA），"
            "或改用 --moe-balance sign。")
    z = hn @ router_w.transpose(0, 1)
    s_all = _sigmoid_orv(z)
    n, E = int(s_all.shape[0]), int(s_all.shape[1])
    k = int(top_k)
    if bias_np is None:
        sel = s_all
    else:
        b = np.asarray(bias_np, np.float32).reshape(E)
        sel = s_all + T.tensor(b)
    idx_buf, _val = rt.moe_topk(sel._gpu, k)
    gate_buf = rt.moe_gather_pairs(s_all._gpu, idx_buf, n, E, k)
    need = bool(s_all.requires_grad)
    gates = T.Tensor._from_gpu(gate_buf, requires_grad=need,
                              _parents=(s_all,) if need else ())
    if need:
        def bw(g, grads):
            T._acc(grads, s_all, rt.moe_scatter_pairs(g, idx_buf, n, E, k))

        gates._backward = bw
    # ★ 选中集合内重归一化（参考 ③）。sigmoid 输出恒 > 0，clamp 只是防 fp32 下溢到 0
    #   时出现 0/0（参考实现同样 clamp 到 1e-9）。
    gates = gates / gates.sum(axis=-1, keepdims=True).clamp(min=1e-9)
    eids = np.asarray(idx_buf.to_numpy() if hasattr(idx_buf, "to_numpy") else idx_buf)
    eids = eids.astype(np.int64).reshape(n, -1)
    return eids, gates, s_all


def update_moe_bias(P, cfg, state):
    """K3 Quantile Balancing 的**偏置更新**（`MoEv3.update_bias` 的 'quantile' 分支）。

    逐行照 `model_v3.py` 的 `update_bias`：

        s      = 本步 forward 的 sigmoid 亲和度 (N,E)（detach）
        sel    = s + b                       # 用**旧** b 做的选择（因果性：新 b 下一步生效）
        kk     = min(k+1, E)
        alpha  = topk(sel, kk).values[:, -1]  # token 侧门槛（省一次分位数）
        r      = alpha[:,None] - s
        b_hat  = -quantile_{1-k/E}(r, dim=0)  # 逐专家分位数
        b     <- b_hat - mean(b_hat)          # 中心化，避免整体漂移

    ## 为什么必须实现（只做前向是不够的）

    参考实现的 `Trainer.optim_step()` **每步**调 `model.update_moe_bias()`
    （`train_v3.py:297`，在 `opt.step()` 之后）。少了它，第 2 步起 b 不同 ⇒ 路由不同
    ⇒ loss 曲线必然分叉。所以 `moe_balance='quantile'` 这一项 = 前向（sigmoid + 重归一化）
    + 本函数。

    ## 实现口径

    分位数在**主机侧**用 numpy 算：`(N,E)` 只有几 KB，一次 D2H 可忽略；ORV 没有
    `quantile` 算子，而 `torch.quantile` 默认 linear 插值与 `np.quantile(method="linear")`
    **定义相同**。累加精度（fp32 vs fp64）会让 b 有 ~1e-7 量级抖动 —— 这是本实现唯一
    的非逐位等价点，对拍里已量化。
    """
    scores = (state or {}).get("scores") or {}
    if not scores:
        return
    top_k = int(cfg.get("top_k", 2) or 2)
    E = int(cfg["E"])
    kk = min(top_k + 1, E)
    q = 1.0 - float(top_k) / float(E)
    bias = state.setdefault("bias", {})
    for i, s in scores.items():
        s = np.asarray(s, np.float32)
        b = bias.get(i)
        if b is None:
            b = np.zeros(E, np.float32)
        sel = s + np.asarray(b, np.float32).reshape(1, E)
        # 第 kk 大的值：`-sel` 的第 kk 小（partition 是 O(N·E)，不排序）
        alpha = -np.partition(-sel, kk - 1, axis=1)[:, kk - 1]
        r = alpha[:, None] - s
        b_hat = -np.quantile(r, q, axis=0)
        b_hat = b_hat - b_hat.mean()
        bias[i] = np.asarray(b_hat, np.float32)


def _embed(ids, P, cfg, seq, d):
    """嵌入 → `(1,T,d)`，并实现 `--pad-id` 的**梯度**语义。

    ★ PyTorch 的 `nn.Embedding(padding_idx=pad)`：该行初始化为 0，且反向时该行梯度清零
      （`padding_idx` 语义）。`build()` 已经把该行建成 0；这里再给输出乘一张 per-token
      掩码（pad token 的输出乘 0），使**反向也不会**给该行送梯度。

    ★ 为什么不用"反向之后 mask 一次 `.grad`"：`--tie-embeddings` 时 emb 与 head 是同一个
      张量，事后 mask 会把 **head 那条路径**的梯度一起清掉；而参考实现只清 Embedding
      自己那份（`lm_head` 的贡献保留）。掩码只作用在 embedding 这条边上，正好是
      per-edge 语义。

    ★ 前向数值不变：该行恒为 0 ⇒ 取到的就是 0 向量，乘 1/0 结果相同。
    """
    x = P["emb"][ids]                                  # (T,d)，整数索引取行
    pad_id = cfg.get("pad_id")
    if pad_id is not None:
        keep = (np.asarray(ids).reshape(-1) != int(pad_id)).astype(np.float32)
        x = x * T.tensor(keep.reshape(-1, 1))
    return x.reshape(1, int(seq), int(d))


def _attn_part(h, i, L, cfg, kinds, plan, cos, sin, opts, counts=None):
    """一层的「pre-norm → 注意力」半段，返回 `[B,T,D]`（**还没加残差**）。

    与参考 `TransformerBlockV3.attn_part` 同构：`o = attn(ln1(h))`，残差由调用方加
    （普通循环 `h = x + attn`；AttnRes 循环 `new_partial = partial + a`）。

    `counts` 给出时顺带累计层类型计数 —— main 用它做"真的按 kinds 走了"的断言
    （不是"跑通了就算验证"）。
    """
    xn = V.rmsnorm_orv(h, L["ln1"])
    if kinds[i] == "csa":
        if counts is not None:
            counts["csa"] += 1
        C = _csa_module()
        return C.csa_forward(xn, L["csa"], plan, cos, sin)
    if counts is not None:
        counts["global"] += 1
    n_q = int(cfg["n_q"])
    d_k = int(cfg["d"]) // n_q
    n_kv = int(cfg.get("n_kv_attn", cfg["n_kv"]))
    chunk = int(_opt(opts, "chunk", 0) or 0)
    lowmem = bool(_opt(opts, "lowmem", False))
    flash = bool(_opt(opts, "flash", False))
    if cfg.get("qk_norm"):
        if chunk > 0:
            raise SystemExit(
                "--qk-norm 与 --chunk 不能同时用：分块注意力在 v3_attn_chunked 里"
                "自带 q/k 投影，插不进 QK-Norm。"
                "修法：去掉 --chunk（要省 S×S 显存可用 --flash 或 --lowmem），"
                "或去掉 --qk-norm。")
        return _attention_qk_norm(xn, L["wq"], L["wk"], L["wv"], L["wo"],
                                  L["qn"], L["kn"], cos, sin, d_k, n_q, n_kv,
                                  lowmem=lowmem, flash=flash)
    if chunk > 0:
        from tgai_go.train import v3_attn_chunked as CH
        return CH.attention_orv_chunked(xn, L["wq"], L["wk"], L["wv"], L["wo"],
                                        cos, sin, d_k, n_q, n_kv, chunk=chunk)
    return V.attention_orv(xn, L["wq"], L["wk"], L["wv"], L["wo"], cos, sin,
                           d_k, n_q, n_kv, lowmem=lowmem, flash=flash)


def _mlp_part(h, i, L, cfg, opts, state=None):
    """一层的「pre-norm → MoE」半段，返回 `[1,T,d]`（还没加残差）。

    顺序严格照参考 `MoEv3.forward`：routed 聚合 → [**post_norm**] → (+ 共享专家)。
    `post_norm` 必须在共享专家**之前**（参考 ⑤ 在 ⑥ 之前）—— 顺序反了就不是同一个算子。
    """
    seq, d = int(h.shape[1]), int(h.shape[2])
    hn3 = V.rmsnorm_orv(h, L["ln2"])
    hn = hn3.reshape(seq, d)                  # MoE 按行分派，用二维视图
    top_k = int(_opt(opts, "top_k", cfg.get("top_k", 2)))
    balance = cfg.get("moe_balance", "sign")
    bias = None
    if balance == "quantile" and state is not None:
        bias = (state.get("bias") or {}).get(i)
    if balance == "quantile":
        eids, gates, s_all = _route_quantile(hn, L["router"], top_k, bias)
        if top_k == 1:
            mo = MOE.moe_dispatch_orv(hn, eids[:, 0], gates[:, 0],
                                      L["W1"], L["W2"], L["W3"])
        else:
            mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
    elif top_k == 1:
        eids, g, s_all = MOE.route_top1(hn, L["router"])
        mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    else:
        eids, gates, s_all = MOE.route_topk(hn, L["router"], top_k)
        mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
    if balance == "quantile" and state is not None:
        sv = s_all.to_numpy() if hasattr(s_all, "to_numpy") else s_all
        state.setdefault("scores", {})[i] = np.asarray(sv, np.float32)
    if cfg.get("moe_post_norm"):
        mo = V.rmsnorm_orv(mo, L["post_norm"])
    if _opt(opts, "shared", True):
        # 参考实现 n_shared_experts=1：对全部 token 都生效的一个 FFN（**直接相加**，无门控）
        mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
    return mo.reshape(1, seq, d)


def _checkpoint_layer(fn, x):
    """梯度重算：`fn(x)` 前向不建图，反向时用原始 `x` 重跑再反传。

    原型已验证梯度**逐位等价**（`lab/prof/ckpt_proto.py`，max_abs = 0.000e+00）。
    注意 `Tensor.detach()` 在 ORV 上是**拷贝**不是视图（实测 ptr 不同），
    所以这里会多一份层输入（seq512 下约 2 MB/层），这是重算的必要代价。

    只在 `cfg["grad_ckpt"]` 打开时使用；默认路径完全不走这里。
    """
    Tn = T.Tensor
    if not isinstance(x, Tn) or not x.requires_grad:
        return fn(x)
    out = fn(x.detach())                     # 不建图，中间量随即释放
    res = Tn._from_gpu(out._gpu, requires_grad=True, _parents=(x,))

    def bw(g, grads):
        sub = fn(x)                          # 用原始输入重跑，建图
        x.grad = None
        sub.backward(g)                      # 子图反传（写自己那些参数的 .grad）
        if x.grad is not None:
            from . import _acc as _acc_fn
            _acc_fn(grads, x, x.grad._gpu)
            x.grad = None

    res._backward = bw
    return res


def forward_logits(P, cfg, kinds, plan, cos, sin, ids, opts=None,
                   counts=None, state=None):
    """整栈前向 → `logits (T, vocab)`。**训练步与对拍脚本共用这一条**。

    参数：
        P      : `build()` 的返回值（或对拍脚本灌过权重的同一结构）
        cfg    : 配置字典（d / seq / n_q / n_kv / layers / E / attn_res / ...）
        kinds  : `layer_kinds(cfg)`（每层 `'global'` / `'csa'`）
        plan   : `--csa` 的 `CSAPlan`；无 CSA 时为 None
        cos,sin: RoPE 表 `(seq, rot_dim)`
        ids    : 一维 int64 token（长度 = T）
        opts   : argparse 结果（读 top_k / shared / chunk / lowmem / flash）；None ⇒ 用默认
        counts : 传入 dict 时累计 `{'csa':n,'global':m}`（main 的层类型断言用）
        state  : `moe_balance='quantile'` 的状态（`{'bias':{i:arr},'scores':{i:arr}}`）
                  —— 前向把本步的 sigmoid 亲和度写进 `scores`，供 `update_moe_bias` 用

    ## 为什么抽成函数（而不是留在 main 的循环体里）

    对拍脚本要验的必须是**训练步真正跑的那条前向**。之前 `lab/e2e/e2e_parity.py` 自己
    复刻了一份循环 —— 一旦加开关就会出现"复刻已改、CLI 没改"或反之，对拍结论就失效了。
    现在只有这一份实现，`main()` 与对拍都调它。
    """
    seq, d = int(len(ids)), int(cfg["d"])
    x = _embed(ids, P, cfg, seq, d)
    if cfg.get("attn_res"):
        AR = _attnres_module()
        bs = int(cfg.get("attn_res_block", 2) or 2)
        layers, attn_res, mlp_res = [], [], []
        for i, L in enumerate(P["layers"]):
            def _a(h, i=i, L=L):
                return _attn_part(h, i, L, cfg, kinds, plan, cos, sin, opts, counts)

            def _m(h, i=i, L=L):
                return _mlp_part(h, i, L, cfg, opts, state)

            layers.append((_a, _m))
            attn_res.append((L["res_w_attn"], L["res_gn_attn"]))
            mlp_res.append((L["res_w_mlp"], L["res_gn_mlp"]))
        # 四段式循环直接复用包里的实现（与参考 `TGAIV3Model.forward` 的 use_attn_res
        # 分支逐行对应：attn 前深度注意力 → 残差 → 边界收块 → mlp 前深度注意力）
        x = AR.run_attn_res_layers(x, layers, attn_res, mlp_res, block_size=bs)
    else:
        # ★ 也接受环境变量 TG_GRAD_CKPT=1：
        #   `v3_pretrain.main()` 走的是 `_expand_cfg(a)`，不经过 CLI 的那份 cfg 构造，
        #   实测 `--grad-ckpt` 因此没生效（时间/显存逐位不变）。用环境变量兜住，
        #   保证这个开关在任何调用路径下都能打开。
        use_ckpt = (bool(cfg.get("grad_ckpt"))
                    or (os.environ.get("TG_GRAD_CKPT", "") not in ("", "0", "false", "False")))
        for i, L in enumerate(P["layers"]):
            if use_ckpt:
                # ★ 2026-11-17：梯度重算（用算力换显存）。前向不建图、只留层输入；
                #   反向时用原始输入重跑这一层再反传。原型已实测梯度**逐位等价**
                #   （`lab/prof/ckpt_proto.py`：max_abs = 0.000e+00）。
                def _layer(x, i=i, L=L):
                    attn = _attn_part(x, i, L, cfg, kinds, plan, cos, sin, opts, counts)
                    h = x + attn
                    return h + _mlp_part(h, i, L, cfg, opts, state)

                x = _checkpoint_layer(_layer, x)
            else:
                attn = _attn_part(x, i, L, cfg, kinds, plan, cos, sin, opts, counts)
                h = x + attn
                x = h + _mlp_part(h, i, L, cfg, opts, state)
    # head 是张量（vocab,d），不是模块 ⇒ 矩阵乘，别当函数调用
    xf = V.rmsnorm_orv(x, P["ln_f"])
    return xf.reshape(seq, d) @ P["head"].transpose(0, 1)


def build_arg_parser():
    """构造 CLI 解析器（单独拆出来，便于 `--help` 自检与外部脚本复用）。"""
    ap = argparse.ArgumentParser(
        description="在 ORV 上跑 V3 架构的真实训练步（完整 AdamW + 真更新）。"
                    "远端经 pip 包调用：python -m tgai_go.train.v3_train_cli ...")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_q", type=int, default=4,
                    help="注意力查询头数；d/n_q 即 dh。dh==64 时 sdpa_lowmem 可用")
    ap.add_argument("--lowmem", action="store_true",
                    help="走不物化 P 的注意力内核（需 dh==64 + causal + CUDA）")
    ap.add_argument("--chunk", type=int, default=0,
                    help=">0 时用分块注意力（照 PyTorch mem-efficient 的思路：不物化 S×S）")
    ap.add_argument("--flash", action="store_true",
                    help="块化在线 softmax 注意力（不物化 S×S，反向重算）")
    ap.add_argument("--no-shared", action="store_true",
                    help="关闭共享专家。PyT 侧是 n_shared_experts=1，默认开启以保持同构")
    ap.add_argument("--top_k", type=int, default=2,
                    help="每 token 激活专家数。真实 V3 是 2（model_v3.py:145），默认对齐")
    ap.add_argument("--d_ff", type=int, default=0,
                    help="每专家的中间维度。真实 V3 是 2816（model_v3.py:97）；0 = 用 4*d（旧行为）")
    ap.add_argument("--vocab", type=int, default=4096,
                    help="词表大小。真实 V3 是 96000（model_v3.py:93）；小值便于冒烟")
    ap.add_argument("--csa", "--attn-global-every", dest="csa", type=int, default=0,
                    metavar="N",
                    help="真 V3 的 3:1 混合注意力：每 N 层里有 1 层 GlobalAttention，"
                         "其余层用 CSA（v3_csa）。0 = 关闭 = 改动前的行为。"
                         "真 V3 是 4 ⇒ 1-based 第 4/8/12/16/20 层全局、其余 15 层 CSA"
                         "（判定式照 model_v3.py:222 的 (i %% N) == N-1；i 是 0-based）")
    ap.add_argument("--attn-res", dest="attn_res", action="store_true",
                    help="BlockAttnRes 深度注意力残差（K3，参考 attn_res_enabled=True）。"
                         "四段式循环直接调包里的 v3_attnres.run_attn_res_layers，"
                         "不再依赖源码树 tools/（pip 形态也能用）；可与 --csa 同时开")
    ap.add_argument("--attn-res-block", dest="attn_res_block", type=int, default=4,
                    metavar="N",
                    help="AttnRes 的 block 大小（每 N 层收一次 blocks）。"
                         "真 V3 是 4（model_v3.py 的 attn_res_block 默认值）；"
                         "旧版本文件写死 2，本轮改成可配且默认对齐参考。N 大于层数时"
                         "一个 block 都收不到 ⇒ AttnRes 退化为恒等（会让对拍失去意义）")
    # ── 6 项架构对齐开关（默认全关；--faithful-v3 一次全开）──
    ap.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true",
                    help="emb 与 head 共享同一个参数张量（参考 tie_embeddings=True）")
    ap.add_argument("--moe-post-norm", dest="moe_post_norm", action="store_true",
                    help="routed 聚合后、共享专家之前加一次 RMSNorm（参考 moe_post_norm=True）")
    ap.add_argument("--qk-norm", dest="qk_norm", action="store_true",
                    help="Global 层的 q/k 在 RoPE 之前沿 head 维做 RMSNorm"
                         "（参考 attn_qk_norm=True）。CSA 层的 qk_norm 一直由 "
                         "CSAConfig.attn_qk_norm 控制（默认 True），本开关只管 Global 层")
    ap.add_argument("--pad-id", dest="pad_id", type=int, default=None, metavar="N",
                    help="padding token 的 id（参考 pad_token_id=256）：该行 emb 初始化为 0 "
                         "且拿不到梯度（PyTorch padding_idx 语义）")
    ap.add_argument("--moe-balance", dest="moe_balance", default="sign",
                    choices=["sign", "quantile"],
                    help="MoE 负载均衡口径：sign = softmax 亲和度 + 原始门控（旧行为）；"
                         "quantile = K3 QB（sigmoid 亲和度 + 选中集合内归一化 + 分位数偏置更新，"
                         "参考 moe_balance='quantile'）")
    ap.add_argument("--faithful-v3", dest="faithful_v3", action="store_true",
                    help="一次性打开上表 6 项（= 真 V3 的默认配置）。"
                         "注意 --csa 不在其中：要完整 3:1 混合注意力请另外加 --csa 4")
    ap.add_argument("--fp16-gemm", dest="fp16_gemm", action="store_true",
                    help="打开 fp16 GEMM（rt.fp16_gemm=True，走 Tensor Core）。"
                         "默认关：V100 实测慢 16.7%% 且多占 3 GB，只用于复现实验")
    return ap


def _apply_faithful_v3(a, ap):
    """`--faithful-v3`：把 6 项一次全开（**显式给过的参数不覆盖**）。

    ★ 为什么不覆盖显式参数：`--faithful-v3 --pad-id 0` 这种组合里，用户显然想要
      pad=0 而不是 256；静默覆盖会让"我明明传了"变成假的。（`--moe-balance` /
      `--pad-id` 的默认值分别是 "sign" / None，正好能区分"没给"与"给了"。）
    """
    if a.pad_id is None:
        a.pad_id = 256          # 参考 `PAD_ID = 256`（model_v3.py 顶部常量）
    if a.moe_balance == "sign":
        a.moe_balance = "quantile"
    a.tie_embeddings = True
    a.moe_post_norm = True
    a.qk_norm = True
    a.attn_res = True
    return ap


def main(argv=None):
    ap = build_arg_parser()
    a = ap.parse_args(argv)
    a.shared = not a.no_shared          # 默认开启（与 PyT 对齐）

    if a.faithful_v3:
        _apply_faithful_v3(a, ap)

    # ── 开关的前置校验：不静默忽略，也不静默退化 ──
    if int(a.csa) < 0:
        raise SystemExit("--csa 必须是 >= 0 的整数（0 = 关闭）")
    if a.csa and (a.d // a.n_q) % 2 != 0:
        # CSA 的 d_k 取偶数（`d_k = d//n_heads`，奇数时减 1），而 Global 层用 d//n_q。
        # 两者不等会让 RoPE 表维数对不上 —— 这条守卫只是把那个模糊报错换成明确的话。
        raise SystemExit("--csa 要求 d/n_q 为偶数：d=%d, n_q=%d ⇒ d_k=%d"
                         % (a.d, a.n_q, a.d // a.n_q))
    if a.pad_id is not None and not (0 <= int(a.pad_id) < int(a.vocab)):
        raise SystemExit("--pad-id 必须落在 [0, vocab=%d)：收到 %d。"
                         "修法：--pad-id 传词表内的整数（真 V3 是 256，需要 vocab > 256）"
                         % (a.vocab, int(a.pad_id)))
    if a.qk_norm and int(a.chunk) > 0:
        raise SystemExit("--qk-norm 与 --chunk 不能同时用（分块注意力内部自带 q/k 投影，"
                         "插不进 QK-Norm）。修法：去掉 --chunk（要省 S×S 显存可用 "
                         "--flash 或 --lowmem），或去掉 --qk-norm")
    if int(a.attn_res_block) < 1:
        raise SystemExit("--attn-res-block 必须 >= 1：收到 %d。"
                         "修法：真 V3 是 4（旧版本写死 2）" % int(a.attn_res_block))
    if a.attn_res and int(a.attn_res_block) > int(a.layers):
        # 不报错（数学上合法），但会把 AttnRes 变成恒等，让"开了等于没开"——
        # 这对同构验证是陷阱，必须说出来。
        print("[attn-res] 警告：attn_res_block=%d > layers=%d ⇒ 一个 block 都收不到，"
              "AttnRes 退化为恒等（等于没开）。修法：--attn-res-block 取 <= 层数"
              % (int(a.attn_res_block), int(a.layers)), flush=True)

    T.set_default_device("cuda")
    if a.fp16_gemm:
        # 必须在**建参数/建图之前**设：`__init__.py` 的 GEMM 分派每步检查该属性。
        # 默认关是有实测依据的（见模块文档与 docs/V3_V100_MEASURED.md §2）。
        T._cuda().fp16_gemm = True
        print("[fp16] rt.fp16_gemm=True（V100 实测慢 16.7%、多占 3 GB，仅复现用）", flush=True)
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    cfg = dict(d=a.d, d_ff=(a.d_ff or 4 * a.d), E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=a.vocab, seq=a.seq, steps=a.steps,
               attn_res=bool(a.attn_res),
               # ★ 只有 --attn-res 打开时这个键才有意义；默认（关闭）仍写 2，
               #   保证默认路径的 cfg / RESULT 与改动前**逐字一致**。
               attn_res_block=(int(a.attn_res_block) if a.attn_res else 2))
    if a.csa:
        # 真 V3 的口径（model_v3.py:89-138）：
        #   * n_kv_heads=0 ⇒ Global 层是 **MHA**（kv 头数 = 查询头数），不是 GQA-2；
        #   * rope_theta=500000 且 hetero_rope=False ⇒ 全局层与 CSA 层共用同一张 RoPE 表；
        #   * csa_compress=4 / csa_conv_kernel=3 / csa_local_window=0 / csa_latent=0（默认）。
        # 这些键**只在 --csa 时**加进 cfg ⇒ 默认路径的 cfg 字典与改动前逐字一致。
        cfg.update(csa=int(a.csa), n_kv_attn=a.n_q, rope_theta=500000.0,
                   csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    # ★ 6 项开关：**只在打开时**写进 cfg ⇒ 默认路径的 cfg 字典与改动前逐字一致
    #   （RESULT 里的 cfg 也逐字一致，这是"默认路径逐位不变"的一部分）。
    if a.tie_embeddings:
        cfg["tie"] = True
    if a.moe_post_norm:
        cfg["moe_post_norm"] = True
    if a.qk_norm:
        cfg["qk_norm"] = True
    if a.pad_id is not None:
        cfg["pad_id"] = int(a.pad_id)
    if a.moe_balance != "sign":
        cfg["moe_balance"] = str(a.moe_balance)
    kinds = layer_kinds(cfg)
    n_csa = kinds.count("csa")
    n_global = len(kinds) - n_csa
    C = _csa_module() if a.csa else None
    arch_on = [name for name, key in ARCH_ITEMS
               if (cfg.get(key) not in (None, False, "sign"))]
    if arch_on:
        print("[arch] 已打开的架构对齐项：%s" % (arch_on,), flush=True)

    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    # `moe_balance='quantile'` 的偏置状态：{'bias': {层号: (E,)}, 'scores': {层号: (N,E)}}
    # 参考实现的 expert_bias 是 register_buffer（不参与梯度），这里同样放在主机侧。
    bias_state = {"bias": {}, "scores": {}} if cfg.get("moe_balance") == "quantile" else None
    if a.csa:
        csac = cfg["csa_cfg"]
        # ★ plan 与 RoPE 表只在建模型时算**一次**：plan 是掩码/索引（seg=None ⇒ 与步数无关
        #   的常量），rope 是 (seq, rot_dim) 全表。15 个 CSA 层与每一步都复用，不再重算。
        plan = C.make_plan(csac, B=1, Tlen=a.seq) if n_csa else None
        cos, sin = C.rope_tables(csac, seq_len=a.seq)
        print("[csa] %d 层 = global %d（1-based %s）+ CSA %d（1-based %s）"
              % (len(kinds), n_global,
                 [i + 1 for i, k in enumerate(kinds) if k == "global"],
                 n_csa, [i + 1 for i, k in enumerate(kinds) if k == "csa"]), flush=True)
        print("[csa] %r" % (csac,), flush=True)
        print("[csa] Global 层 n_kv=%d（MHA）, rope_theta=%g；CSA plan: B=%d T=%d Tc=%d "
              "lk=%d use_seg=%s"
              % (cfg["n_kv_attn"], csac.rope_theta,
                 plan.B if plan else -1, plan.Tlen if plan else -1,
                 plan.Tc if plan else -1, plan.lk if plan else -1,
                 plan.use_seg if plan else "-"), flush=True)
    else:
        plan = None
        cos, sin = V.rope_cos_sin(a.seq, a.d // cfg["n_q"], 10000.0)
        cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=a.lr)

    stop = threading.Event(); prof = {}
    th = threading.Thread(target=_vram_peak, args=(stop, base, prof), daemon=True); th.start()

    ids_all = rng.integers(0, cfg["vocab"], (a.steps + 1, a.seq)).astype(np.int64)
    losses, times = [], []
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        counts = {"csa": 0, "global": 0}
        # ★ 现在只有**一份**前向实现（`forward_logits`）：AttnRes、6 项开关、CSA 全在里面，
        #   对拍脚本调的也是它 —— 不会再出现"对拍复刻一份、CLI 另一份"的漂移。
        logits = forward_logits(P, cfg, kinds, plan, cos, sin, ids,
                                opts=a, counts=counts, state=bias_state)
        # ★ 计数断言：这一步**真的**按 kinds 走了（不是"跑通了就算验证"）。
        #   AttnRes 现在也走同一个分派（`_attn_part`）⇒ 两种循环都能断言。
        if (counts["csa"], counts["global"]) != (n_csa, n_global):
            raise SystemExit(
                "[csa] step %d 层类型计数不符：实际 %d CSA / %d Global，期望 %d / %d"
                % (step, counts["csa"], counts["global"], n_csa, n_global))
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        opt.zero_grad(); loss.backward(); opt.step()
        if bias_state is not None:
            # ★ 参考 `train_v3.py:297`：`optim_step()` 里 **opt.step() 之后**调
            #   `model.update_moe_bias()`（每优化器步一次）。位置必须一致：
            #   本步 forward 用的是更新前的 b，新 b 下一步才生效。
            update_moe_bias(P, cfg, bias_state)
        times.append(time.time() - t0); losses.append(lv)
        # 默认路径打印字符串与改动前**逐字相同**（tag 为空串）；--csa 时追加层类型计数
        tag = ("  [attn] csa=%d/%d global=%d/%d"
               % (counts["csa"], n_csa, counts["global"], n_global)) if a.csa else ""
        print("step %d loss=%.4f %.2fs%s" % (step, lv, times[-1], tag), flush=True)

    stop.set(); th.join(timeout=3)
    res = dict(
        # CSAConfig 不是 JSON 类型（只在 --csa 时出现）⇒ 落盘前剔掉，其余键原样保留
        cfg={k: v for k, v in cfg.items() if k != "csa_cfg"}, losses=losses,
        step_s=float(np.mean(times)),
        tok_s=a.seq / float(np.mean(times)),
        peak_mib=prof.get("peak"), base_mib=base,
        # ★ 精确高水位：运行时池自己记的峰值，不受 nvidia-smi 采样间隔影响
        pool=T._cuda().pool_stats())
    if a.csa:
        res["attn_mix"] = dict(
            every=int(a.csa), n_csa=n_csa, n_global=n_global, kinds=kinds,
            global_layers_1based=[i + 1 for i, k in enumerate(kinds) if k == "global"],
            csa_layers_1based=[i + 1 for i, k in enumerate(kinds) if k == "csa"],
            rope_theta=float(cfg["csa_cfg"].rope_theta),
            n_kv_global=int(cfg["n_kv_attn"]),
            csa_cfg=str(cfg["csa_cfg"]),
            plan=(dict(B=plan.B, T=plan.Tlen, Tc=plan.Tc, lk=plan.lk,
                       use_seg=plan.use_seg) if plan else None))
    if arch_on:
        # 只在打开 6 项里至少一项时出现 ⇒ 默认路径的 RESULT 与改动前逐字一致。
        # `n_params` 是**去重后**交给优化器的张量数：`--tie-embeddings` 会让它比
        # 未绑定时少 1（emb 与 head 共用同一个张量，对应 PyTorch `parameters()` 的口径）。
        res["arch"] = dict(
            items=arch_on, tie=bool(cfg.get("tie")),
            moe_post_norm=bool(cfg.get("moe_post_norm")),
            qk_norm=bool(cfg.get("qk_norm")), pad_id=cfg.get("pad_id"),
            moe_balance=str(cfg.get("moe_balance", "sign")),
            attn_res_block=int(cfg.get("attn_res_block", 2)),
            n_params=len(params_of(P)))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
    print("losses", losses)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
