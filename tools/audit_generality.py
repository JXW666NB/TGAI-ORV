# -*- coding: utf-8 -*-
"""通用能力审计：ORV 的公开 API 能不能表达「用户自己的模型」。

## 为什么不是"移植 21e"

用户的要求是通用训练引擎。所以正确做法是：**像普通用户那样只用公开 API 去搭**，
凡是要绕路的地方，就是通用能力缺口。这个脚本只查"有没有"，不搭模型。

审计项来自 21e 暴露的真实需求，但都写成**通用形式**：
1. 每层不同头数 / 不同 d_head —— 用户能不能自己控制？
2. 可学习 RMSNorm（γ 是参数）—— 通用归一化
3. LayerScale —— 参数缩放
4. 任意 mask 的注意力（bool / float / 自定义）—— 不只是 causal
5. 大专家数 MoE（36 专家 top-6）
6. 分块 CE（避免 [B,T,V]）
7. fp16/bf16 激活存储（真 AMP）—— torch.autocast 的语义
"""
import inspect
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import tgai_go.train as T          # noqa: E402


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "<无签名>"


def has_param(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


rows = []


def chk(item, ok, detail):
    rows.append((item, "YES" if ok else "**NO**", detail))


# 1. 每层不同头数 / d_head
att = getattr(T.nn, "CausalSelfAttention", None) or getattr(T.nn, "Attention", None)
names = [n for n in dir(T.nn) if "ttention" in n]
d = ""
if att is not None:
    d = "签名 %s" % sig(att.__init__)
    chk("每层可指定不同 d_head", has_param(att.__init__, "d_head"),
        "有 d_head 参数；" + d if has_param(att.__init__, "d_head")
        else "只能由 d_model/n_heads 推出 d_head；" + d)
else:
    chk("有注意力层", False, "nn 里与注意力相关的名字：%s" % (names or "无"))

# 2. 可学习 RMSNorm
rms = getattr(T.nn, "RMSNorm", None)
if rms is not None:
    src = inspect.getsource(rms)
    learnable = ("Parameter" in src) and ("weight" in src)
    chk("RMSNorm γ 可学习", learnable,
        "签名 %s；含 Parameter(weight)=%s" % (sig(rms.__init__), learnable))
else:
    chk("RMSNorm 存在", False, "nn 里没有 RMSNorm")

# 3. LayerScale
has_ls = any("LayerScale" in n or "LayerScale" in n for n in dir(T.nn))
chk("有 LayerScale 层", has_ls,
    "可直接用 nn.Parameter 手工实现（乘一个标量），不算硬缺口" if not has_ls else "")

# 4. 任意 mask 的注意力
sdpa = getattr(T.functional, "scaled_dot_product_attention", None)
if sdpa is not None:
    p = inspect.signature(sdpa).parameters
    chk("SDPA 接受任意 mask（bool/float）", "attn_mask" in p or "mask" in p,
        "参数：%s" % list(p))
else:
    chk("SDPA 存在", False, "")

# 5. 大专家数 MoE
moe = getattr(T.nn, "MoE", None)
if moe is not None:
    p = inspect.signature(moe.__init__).parameters
    chk("MoE 支持 n_experts=36 / top_k=6", "n_experts" in p and "top_k" in p,
        "签名 %s" % sig(moe.__init__))
else:
    chk("MoE 存在", False, "")

# 6. 分块 CE
lce = getattr(T.functional, "linear_cross_entropy", None)
chk("分块 CE（避免 [B,T,V]）", lce is not None,
    "签名 %s" % sig(lce) if lce else "缺失")

# 7. 真 AMP：激活存 fp16/bf16
amp_src = ""
try:
    amp_src = inspect.getsource(T.amp)
except Exception:      # noqa: BLE001
    pass
chk("AMP 把激活存成 fp16/bf16（torch.autocast 语义）", False,
    "ORV 的 AMP 只 cast GEMM 输入，参数/激活/梯度全程 fp32（见 docs/GAP_VS_PYTORCH.md 第 7 条）")

w = max(len(r[0]) for r in rows)
print("| 能力 | 有？ | 说明 |")
print("| --- | --- | --- |")
for item, ok, detail in rows:
    print("| %s | %s | %s |" % (item, ok, detail))
