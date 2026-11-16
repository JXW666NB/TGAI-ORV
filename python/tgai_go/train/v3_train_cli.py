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

__all__ = ["build", "params_of", "build_arg_parser", "main"]


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


def build(cfg, rng):
    """随机初始化参数（同构对比用）。返回值都是 requires_grad 的 ORV 张量。"""
    d, d_ff, E, L = cfg["d"], cfg["d_ff"], cfg["E"], cfg["layers"]
    n_q, n_kv = cfg["n_q"], cfg["n_kv"]
    d_k = d // n_q
    def p(*shape, scale=0.02):
        return T.tensor(rng.normal(0, scale, shape).astype(np.float32), requires_grad=True)
    P = {"emb": p(cfg["vocab"], d), "ln_f": T.tensor(np.ones(d, np.float32), requires_grad=True),
         "head": p(cfg["vocab"], d), "layers": []}
    for _ in range(L):
        P["layers"].append(dict(
            ln1=T.tensor(np.ones(d, np.float32), requires_grad=True),
            ln2=T.tensor(np.ones(d, np.float32), requires_grad=True),
            # router 是**独立参数**（参考实现 `nn.Linear(d_model, n_experts)`），
            # 不能每步用 cat([W1[e]…]) 现拼（既偏架构，又多拷 E×d_ff×d 个元素）
            router=p(E, d),
            wq=p(n_q * d_k, d), wk=p(n_kv * d_k, d), wv=p(n_kv * d_k, d), wo=p(d, n_q * d_k),
            # ★ 专家权重用**独立参数列表**，不用三维张量再 [e] 索引：
            #   实测 `W3d[e]` 每次都拷贝（4 MiB/次），MoE 每层索引 24 次（8 专家 × 3 权重），
            #   拷出来的副本还会被自动微分保留到反向 —— 一层约 200 MiB 的纯浪费。
            #   列表索引不产生任何 GPU 拷贝。
            W1=[p(d_ff, d) for _ in range(E)],
            W2=[p(d_ff, d) for _ in range(E)],
            W3=[p(d, d_ff) for _ in range(E)],
            S1=p(d_ff, d), S2=p(d_ff, d), S3=p(d, d_ff),     # 共享专家的 FFN（对齐 PyT 的 n_shared_experts=1）
        ))
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
    return P


def params_of(P):
    out = [P["emb"], P["ln_f"], P["head"]]
    for L in P["layers"]:
        for k, v in L.items():
            if isinstance(v, list):      # 专家权重是列表
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
    ap.add_argument("--attn-res", dest="attn_res", action="store_true",
                    help="BlockAttnRes 深度注意力残差（K3）。需要源码树的 tools/，pip 形态会明确报错")
    ap.add_argument("--fp16-gemm", dest="fp16_gemm", action="store_true",
                    help="打开 fp16 GEMM（rt.fp16_gemm=True，走 Tensor Core）。"
                         "默认关：V100 实测慢 16.7%% 且多占 3 GB，只用于复现实验")
    return ap


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    a.shared = not a.no_shared          # 默认开启（与 PyT 对齐）

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

    rng = np.random.default_rng(0)
    P = build(cfg, rng)
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
        x = P["emb"][ids].reshape(1, a.seq, a.d)          # 嵌入（整数索引）→ (1,T,d)
        if a.attn_res:
            # BlockAttnRes 走四段式循环（attn 前/后各一次深度注意力），已数值验证 1.9e-06
            x = run_layers_attn_res(x, P, cfg, cos, sin, a)
        for L in ([] if a.attn_res else P["layers"]):
            if a.chunk > 0:
                from tgai_go.train import v3_attn_chunked as CH
                attn = CH.attention_orv_chunked(V.rmsnorm_orv(x, L["ln1"]),
                                                L["wq"], L["wk"], L["wv"], L["wo"],
                                                cos, sin, a.d // cfg["n_q"],
                                                cfg["n_q"], cfg["n_kv"], chunk=a.chunk)
            else:
                attn = V.attention_orv(V.rmsnorm_orv(x, L["ln1"]),
                                       L["wq"], L["wk"], L["wv"], L["wo"],
                                       cos, sin, a.d // cfg["n_q"], cfg["n_q"], cfg["n_kv"],
                                       lowmem=a.lowmem, flash=a.flash)
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
        # head 是张量（vocab,d），不是模块 ⇒ 矩阵乘，别当函数调用
        logits = V.rmsnorm_orv(x, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.time() - t0); losses.append(lv)
        print("step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)

    stop.set(); th.join(timeout=3)
    res = dict(cfg=cfg, losses=losses, step_s=float(np.mean(times)),
               tok_s=a.seq / float(np.mean(times)),
               peak_mib=prof.get("peak"), base_mib=base,
               # ★ 精确高水位：运行时池自己记的峰值，不受 nvidia-smi 采样间隔影响
               pool=T._cuda().pool_stats())
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
    print("losses", losses)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
