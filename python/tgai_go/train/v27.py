# -*- coding: utf-8 -*-
"""TGAI NB **V2.7**（3.0B MoE）→ ORV 的配置与权重映射。

事实来源（全部实测，不猜）：
* `.TG` 头：`L=28 D=1024 NH=16 FF=4096 V=32779 max_seq=8192 experts=8 activated=2`
  `weight_dtype=6(BF16)` `arch=0` `num_kv_heads=0`
* `last_step.pt`：`moe_balance="both"` `n_shared_experts=0` `rope_theta=10000.0`
* `tgai_cuda.cu:151-157`（C 引擎源码注释）：`_qb_bias` 形状 `[n_experts]`，
  **必须在 top-k 之前加到 router logits 上**（不读它 ⇒ top-k 选错专家 ⇒ 乱码）
* 引擎用 `d_emb_t`（由 embedding 转置）作 LM head，`.pt` 有 `lm_head` 而 `.TG` 无
  ⇒ **tied embedding**
"""
import re

#: V2.7 的架构常量（来自 `.TG` 头 + `model_config`，见 docs/V27_MOE_PORT.md）
V27 = dict(n_layers=28, d_model=1024, n_heads=16, d_ff=4096, vocab=32779,
           max_seq=8192, n_experts=8, n_activated=2, rope_theta=10000.0,
           tie=True, arch=0)

_ATTN = ("q_proj", "k_proj", "v_proj", "out_proj")


def cfg(**over):
    """返回 V2.7 配置字典（可覆盖，便于测试）。"""
    c = dict(V27)
    c.update(over)
    c.setdefault("n_kv_heads", 0)          # 0 = 与 n_heads 相同（V2.7 无 GQA）
    c.setdefault("head_dim", c["d_model"] // c["n_heads"])
    return c


def v27_names(c=None):
    """按实测命名规则生成 V2.7 的全部张量名（用于映射覆盖检查）。"""
    c = c or cfg()
    names = ["token_embedding.weight", "final_norm.weight"]
    for i in range(c["n_layers"]):
        p = "blocks.%d." % i
        names += [p + "ln1.weight", p + "ln2.weight"]
        names += [p + "attn.%s.weight" % a for a in _ATTN]
        names += [p + "moe.router.weight", p + "moe._qb_bias"]
        for e in range(c["n_experts"]):
            names += ["%smoe.experts.%d.%s.weight" % (p, e, w) for w in ("w1", "w2", "w3")]
    return names


#: 专家三矩阵 → ORV 的 gate/up/down（与引擎 `d_expert_w1/w2/w3` 一致）
_EXPERT_MAP = {"w1": "gate", "w2": "up", "w3": "down"}


def to_orv(name, c=None):
    """V2.7 张量名 → ORV 参数名。无法映射返回 None（**绝不瞎猜**）。"""
    c = c or cfg()
    if name == "token_embedding.weight":
        return "token_embedding.weight"          # tied：同时承担 lm_head
    if name == "final_norm.weight":
        return "final_norm.weight"
    m = re.match(r"^blocks\.(\d+)\.(.+)$", name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    if rest in ("ln1.weight", "ln2.weight"):
        return "blocks.%s.%s" % (i, rest)
    m2 = re.match(r"^attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$", rest)
    if m2:
        return "blocks.%s.attn.%s.weight" % (i, m2.group(1))
    if rest == "moe.router.weight":
        return "blocks.%s.moe.router.weight" % i
    if rest == "moe._qb_bias":
        # 旁路偏置：ORV 侧单独保存（`moe.qb_bias`），前向里加在 top-k 之前
        return "blocks.%s.moe.qb_bias" % i
    m3 = re.match(r"^moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$", rest)
    if m3:
        return "blocks.%s.moe.experts.%s.%s.weight" % (
            i, m3.group(1), _EXPERT_MAP[m3.group(2)])
    return None


def check_coverage(names=None, c=None, verbose=True):
    """返回 (mapped, unmapped, dup)。unmapped 非空即视为映射不完整。"""
    c = c or cfg()
    names = names if names is not None else v27_names(c)
    mapped, unmapped = {}, []
    dup = []
    for n in names:
        o = to_orv(n, c)
        if o is None:
            unmapped.append(n)
        elif o in mapped:
            dup.append(o)
        else:
            mapped[o] = n
    if verbose:
        print("V2.7 张量 %d 个 → 映射 %d 个；未映射 %d；重名 %d"
              % (len(names), len(mapped), len(unmapped), len(dup)))
        for n in unmapped[:5]:
            print("   未映射:", n)
    return mapped, unmapped, dup
