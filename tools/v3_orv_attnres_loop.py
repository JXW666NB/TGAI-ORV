# -*- coding: utf-8 -*-
"""V3 的 AttnRes 层循环（从 `tools/v3_orv_train.py` 的层体抽出来，供其调用）。

为什么单独一个文件：训练脚本里那段层体是"默认路径"，改动它风险高
（2026-11-16 就因为在里面加开关时写错引号，提交了一个语法错误的版本）。
这里做成**只新增、不改旧路径**的形式，接的时候只需在 main() 里加一个 if。

循环语义严格照 `model_v3.py:1356-1389`：

    partial = x; blocks = []
    for i, blk in enumerate(blocks):
        h = attn_res_proj[i](blocks, partial)      # ① attn 前的深度注意力
        a = blk.attn_part(h)
        new_partial = partial + a
        if (i + 1) % bs == 0: blocks.append(new_partial)
        partial = new_partial
        h = mlp_res_proj[i](blocks, partial)       # ② mlp 前的深度注意力
        m = blk.mlp_part(h)
        partial = partial + m
    x = partial

注意每层有**两次**深度注意力（attn 前、mlp 前），各自独立的伪查询与 RMSNorm γ。
"""
import numpy as np

from tgai_go.train import v3_attnres as AR


def run_layers_attn_res(x, P, cfg, cos, sin, opts):
    """跑完所有层，返回最终 hidden（形状 (1, T, d)）。

    x     : (1, T, d) 的嵌入输出
    P     : `build()` 的返回值（含 layers 列表）
    cfg   : 配置字典（用 n_q / n_kv）
    cos,sin: RoPE 表
    opts  : 与 argparse 同名的属性集合（seq / d / top_k / shared / chunk / lowmem / flash）
    """
    import tgai_go.train as T
    from tgai_go.train import v3_attn_chunked as CH
    from tgai_go.train import v3_ops as V
    from tgai_go.train import v3_moe as MOE

    dh = opts.d // cfg["n_q"]
    bs = int(cfg.get("attn_res_block", 2)) or 2
    partial = x
    blocks = []

    for i, L in enumerate(P["layers"]):
        # ① attn 前的深度注意力
        h = AR.block_attn_res(blocks, partial, L["res_w_attn"], L["res_gn_attn"])
        if opts.chunk > 0:
            attn = CH.attention_orv_chunked(V.rmsnorm_orv(h, L["ln1"]),
                                            L["wq"], L["wk"], L["wv"], L["wo"],
                                            cos, sin, dh,
                                            cfg["n_q"], cfg["n_kv"], chunk=opts.chunk)
        else:
            attn = V.attention_orv(V.rmsnorm_orv(h, L["ln1"]),
                                   L["wq"], L["wk"], L["wv"], L["wo"],
                                   cos, sin, dh, cfg["n_q"], cfg["n_kv"],
                                   lowmem=opts.lowmem, flash=opts.flash)
        new_partial = partial + attn
        if (i + 1) % bs == 0:
            blocks.append(new_partial)
        partial = new_partial

        # ② mlp 前的深度注意力
        h = AR.block_attn_res(blocks, partial, L["res_w_mlp"], L["res_gn_mlp"])
        hn3 = V.rmsnorm_orv(h, L["ln2"])
        hn = hn3.reshape(opts.seq, opts.d)          # MoE 按行分派，用二维视图
        if int(opts.top_k) == 1:
            eids, g, _s = MOE.route_top1(hn, L["router"])
            mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
        else:
            eids, gates, _s = MOE.route_topk(hn, L["router"], int(opts.top_k))
            mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
        if opts.shared:
            # 参考实现 n_shared_experts=1：对全部 token 都生效的一个 FFN
            mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
        partial = partial + mo.reshape(1, opts.seq, opts.d)

    return partial
