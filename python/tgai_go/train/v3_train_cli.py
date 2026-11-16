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
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from tgai_go import train as T
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V

__all__ = ["build", "params_of", "build_arg_parser", "main",
           "is_global_layer", "layer_kinds"]

# 惰性导入的 CSA 实现（见 `_csa_module`）。`--csa` 关闭时保持为 None。
_CSA = None


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
        if cfg.get("attn_res"):
            raise ValueError("--csa 与 attn_res 不能同时使用：AttnRes 循环假定每层都有 "
                             "wq/wk/wv/wo，CSA 层没有这四个权重。"
                             "修法：cfg 里去掉一个（main 的 --csa/--attn-res 也互斥）")
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
    P = {"emb": p(cfg["vocab"], d), "ln_f": T.tensor(np.ones(d, np.float32), requires_grad=True),
         "head": p(cfg["vocab"], d), "layers": []}
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
    """
    out = [P["emb"], P["ln_f"], P["head"]]
    for L in P["layers"]:
        for k, v in L.items():
            if isinstance(v, dict):          # CSA 权重 dict（--csa）
                out.extend(t for t in v.values() if t is not None)
            elif isinstance(v, list):        # 专家权重是列表
                out.extend(v)
            else:
                out.append(v)
    return out


def _repo_root():
    """本文件在源码树里是 `<repo>/python/tgai_go/train/v3_train_cli.py` ⇒ 上溯 4 层。

    pip 安装形态（site-packages）下上溯出来的不是仓库根，`tools/` 自然不存在 ——
    调用方只用它给 `--attn-res` 找源码树脚本，找不到会明确报错，不静默降级。
    """
    p = os.path.abspath(__file__)
    for _ in range(4):
        p = os.path.dirname(p)
    return p


def _load_attnres_loop():
    """加载 `run_layers_attn_res`（AttnRes 的四段式层循环）。

    ## 为什么不在包里

    `--attn-res` 是实验路径（数值已验，但默认关），循环实现留在源码树的
    `tools/v3_orv_attnres_loop.py`（它只依赖包里已有的 `v3_attnres` / `v3_ops` /
    `v3_moe` / `v3_attn_chunked`）。**本轮任务写作用域只允许改两个文件**，所以不搬进包；
    这里按「源码树存在就加载，否则明确报错」处理：

      1. 先按普通导入（脚本在 tools/ 同级运行、或 tools 已在 sys.path 时能成）；
      2. 不行就由本文件位置反推仓库根，把 `<repo>/tools` 加进 sys.path 再导一次；
      3. 仍不行 ⇒ `SystemExit` 说明原因。**不静默退化成别的路径**（那会得到一个
         看起来正常、其实没走 AttnRes 的训练结果，比报错危险得多）。
    """
    try:
        from v3_orv_attnres_loop import run_layers_attn_res
        return run_layers_attn_res
    except ImportError:
        pass
    tools_dir = os.path.join(_repo_root(), "tools")
    if os.path.isfile(os.path.join(tools_dir, "v3_orv_attnres_loop.py")):
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from v3_orv_attnres_loop import run_layers_attn_res
        return run_layers_attn_res
    raise SystemExit(
        "--attn-res 需要源码树里的 tools/v3_orv_attnres_loop.py，但没找到：%s\n"
        "修法：① pip 安装形态不含 tools/，--attn-res 只在源码树可用；"
        "② 不用该开关（其余训练路径不受影响）。"
        % os.path.join(tools_dir, "v3_orv_attnres_loop.py"))


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
                    help="BlockAttnRes 深度注意力残差（K3）。需要源码树的 tools/，pip 形态会明确报错")
    ap.add_argument("--fp16-gemm", dest="fp16_gemm", action="store_true",
                    help="打开 fp16 GEMM（rt.fp16_gemm=True，走 Tensor Core）。"
                         "默认关：V100 实测慢 16.7%% 且多占 3 GB，只用于复现实验")
    return ap


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    a.shared = not a.no_shared          # 默认开启（与 PyT 对齐）

    # ── --csa 的前置校验：不静默忽略，也不静默退化 ──
    if int(a.csa) < 0:
        raise SystemExit("--csa 必须是 >= 0 的整数（0 = 关闭）")
    if a.csa and a.attn_res:
        raise SystemExit(
            "--csa 与 --attn-res 不能同时打开：AttnRes 的四段式循环"
            "（tools/v3_orv_attnres_loop.py）假定每层都是 GlobalAttention，"
            "CSA 层没有 wq/wk/wv/wo。修法：只留一个。")
    if a.csa and (a.d // a.n_q) % 2 != 0:
        # CSA 的 d_k 取偶数（`d_k = d//n_heads`，奇数时减 1），而 Global 层用 d//n_q。
        # 两者不等会让 RoPE 表维数对不上 —— 这条守卫只是把那个模糊报错换成明确的话。
        raise SystemExit("--csa 要求 d/n_q 为偶数：d=%d, n_q=%d ⇒ d_k=%d"
                         % (a.d, a.n_q, a.d // a.n_q))

    T.set_default_device("cuda")
    if a.fp16_gemm:
        # 必须在**建参数/建图之前**设：`__init__.py` 的 GEMM 分派每步检查该属性。
        # 默认关是有实测依据的（见模块文档与 docs/V3_V100_MEASURED.md §2）。
        T._cuda().fp16_gemm = True
        print("[fp16] rt.fp16_gemm=True（V100 实测慢 16.7%、多占 3 GB，仅复现用）", flush=True)
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    if a.attn_res:
        # 源码树脚本；pip 形态下 _load_attnres_loop() 会给可执行的错误信息
        run_layers_attn_res = _load_attnres_loop()
    cfg = dict(d=a.d, d_ff=(a.d_ff or 4 * a.d), E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=a.vocab, seq=a.seq, steps=a.steps,
               attn_res=bool(a.attn_res), attn_res_block=2)
    if a.csa:
        # 真 V3 的口径（model_v3.py:89-138）：
        #   * n_kv_heads=0 ⇒ Global 层是 **MHA**（kv 头数 = 查询头数），不是 GQA-2；
        #   * rope_theta=500000 且 hetero_rope=False ⇒ 全局层与 CSA 层共用同一张 RoPE 表；
        #   * csa_compress=4 / csa_conv_kernel=3 / csa_local_window=0 / csa_latent=0（默认）。
        # 这些键**只在 --csa 时**加进 cfg ⇒ 默认路径的 cfg 字典与改动前逐字一致。
        cfg.update(csa=int(a.csa), n_kv_attn=a.n_q, rope_theta=500000.0,
                   csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    kinds = layer_kinds(cfg)
    n_csa = kinds.count("csa")
    n_global = len(kinds) - n_csa
    C = _csa_module() if a.csa else None

    rng = np.random.default_rng(0)
    P = build(cfg, rng)
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
    # `--csa` 时 Global 层走 MHA（cfg["n_kv_attn"]=n_q）；默认仍是 cfg["n_kv"]=2
    n_kv_attn = int(cfg.get("n_kv_attn", cfg["n_kv"]))
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        x = P["emb"][ids].reshape(1, a.seq, a.d)          # 嵌入（整数索引）→ (1,T,d)
        if a.attn_res:
            # BlockAttnRes 走四段式循环（attn 前/后各一次深度注意力），已数值验证 1.9e-06
            x = run_layers_attn_res(x, P, cfg, cos, sin, a)
        counts = {"csa": 0, "global": 0}
        for Li, L in enumerate([] if a.attn_res else P["layers"]):
            # pre-norm 只是提成局部变量，表达式与改动前完全相同（`rmsnorm_orv(x, ln1)`）
            xn = V.rmsnorm_orv(x, L["ln1"])
            if kinds[Li] == "csa":
                # 真 V3 的非全局层：CSA。输入同样是 ln1(h)、输出直接加到残差上
                # （model_v3.py:1211 `o, hm = self.attn(self.ln1(h))`，残差在 1221 行）。
                attn = C.csa_forward(xn, L["csa"], plan, cos, sin)
                counts["csa"] += 1
            else:
                counts["global"] += 1
                if a.chunk > 0:
                    from tgai_go.train import v3_attn_chunked as CH
                    attn = CH.attention_orv_chunked(xn, L["wq"], L["wk"], L["wv"], L["wo"],
                                                    cos, sin, a.d // cfg["n_q"],
                                                    cfg["n_q"], n_kv_attn, chunk=a.chunk)
                else:
                    attn = V.attention_orv(xn, L["wq"], L["wk"], L["wv"], L["wo"],
                                           cos, sin, a.d // cfg["n_q"], cfg["n_q"],
                                           n_kv_attn, lowmem=a.lowmem, flash=a.flash)
            h = x + attn
            hn3 = V.rmsnorm_orv(h, L["ln2"])
            hn = hn3.reshape(a.seq, a.d)                  # MoE 按行分派，用二维视图
            # ★ 必须用真正的 router 参数（[E, d]）。原先这行用 `cat([W1[e]])` 当占位，
            #   那会造出 [E·d_ff, d] = [16384, 512] 的矩阵，`hn @ routerᵀ` 得到 **[S, 16384]**，
            #   softmax 与掩码各再一张 ⇒ 每层约 384 MiB 的纯浪费（S=2048 时）。
            #   参考实现的 router 是 `nn.Linear(d_model, n_experts)`，只有 [E, d]。
            if int(a.top_k) == 1:
                eids, g, _s = MOE.route_top1(hn, L["router"])
                mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
            else:
                # 真实 V3 是 top_k=2（model_v3.py:145）：softmax 全体专家 → 取前 k →
                # 门控用原始亲和度（不重新归一化），见 route_topk 的说明。
                eids, gates, _s = MOE.route_topk(hn, L["router"], int(a.top_k))
                mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
            # ★ 共享专家：参考实现是 `n_activated=1, n_shared_experts=1`，即
            #   路由专家 top_k=1 的加权和 **再加上** 一个对所有 token 都生效的 FFN。
            #   之前 ORV 缺这一支，导致两侧参数不等（少 25.2 M），比较不是同构的。
            if a.shared:
                mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
            x = h + mo.reshape(1, a.seq, a.d)
        # ★ 计数断言：这一步**真的**按 kinds 走了（不是"跑通了就算验证"）。
        #   AttnRes 的层循环在 tools/ 里，不在这里重复计数。
        if not a.attn_res and (counts["csa"], counts["global"]) != (n_csa, n_global):
            raise SystemExit(
                "[csa] step %d 层类型计数不符：实际 %d CSA / %d Global，期望 %d / %d"
                % (step, counts["csa"], counts["global"], n_csa, n_global))
        # head 是张量（vocab,d），不是模块 ⇒ 矩阵乘，别当函数调用
        logits = V.rmsnorm_orv(x, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        opt.zero_grad(); loss.backward(); opt.step()
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
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
    print("losses", losses)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
