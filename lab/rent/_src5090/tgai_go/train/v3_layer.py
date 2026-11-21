# -*- coding: utf-8 -*-
"""V3 的 Transformer 层（最小可验证版）：pre-norm 残差 + 注意力 + 专家 FFN。

## 结构与顺序（与参考实现对应）

```
h = x + attn(rmsnorm(x))          # 注意力子层
y = h + moe(rmsnorm(h))           # FFN 子层（MoE / 单专家）
```

## 这一版刻意做的简化（写清限制，避免被当成"已完成"）

* **单专家**：真正的 V3 用 8～16 个专家 + top-k 路由。这里只走一个专家，
  因为**按行分派需要 gather/scatter，ORV 还没有**（见 `docs/V3_PORT_STATUS.md`）。
  单专家时路由恒等于"全部 token 走同一个 FFN"，所以这一步验证的是**胶水**：
  norm 放哪、残差怎么加、形状怎么对齐、通道顺序对不对。
* **不用 BlockAttnRes**：V3 的块级注意力残差留到后面。
* CSA 未接：这里用 GlobalAttention（零件已全部验证过）。

## 为什么先验胶水

norm/残差/形状顺序这类错误在整模型里表现为"loss 略偏但还在降"，最难定位；
在单层里它们会直接放大成可比的 logits 差异。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgai_go.train import v3_ops as V  # noqa: E402


def _get(w, k):
    """权重既可以是 dict（带键名）也可以是按顺序的元组，统一取出。"""
    return w[k] if isinstance(w, dict) else w[k]


def layer_numpy(x, w, cos, sin, n_q, n_kv, rot_dim, eps=1e-6,
                add_mask=None, beta1=4.0, beta2=25.0):
    """单层前向的 numpy 参考：`x → h → y`。"""
    ln1 = _get(w, "ln1")
    from tgai_go.train import v3_masks as M
    mask = add_mask if add_mask is not None else M.causal_additive(x.shape[1])
    h = x + V.attention_numpy(V.rmsnorm_numpy(x, ln1, eps),
                              w["wq"], w["wk"], w["wv"], w["wo"],
                              cos, sin, rot_dim, mask, n_q, n_kv)
    ln2 = _get(w, "ln2")
    y = h + V.situ_glu_numpy(V.rmsnorm_numpy(h, ln2, eps),
                             w["w1"], w["w2"], w["w3"], beta1, beta2)
    return y


def layer_orv(x, w, cos, sin, n_q, n_kv, rot_dim, eps=1e-6,
              add_mask=None, beta1=4.0, beta2=25.0):
    """同一层的 ORV 张量版（单专家）。"""
    T = V.T
    xt = x if hasattr(x, "to_numpy") else T.tensor(np.ascontiguousarray(x, np.float32))
    ct = cos if hasattr(cos, "to_numpy") else T.tensor(np.ascontiguousarray(cos, np.float32))
    st = sin if hasattr(sin, "to_numpy") else T.tensor(np.ascontiguousarray(sin, np.float32))
    ln1 = T.tensor(np.ascontiguousarray(w["ln1"], np.float32))
    ln2 = T.tensor(np.ascontiguousarray(w["ln2"], np.float32))

    def M(name):
        return T.tensor(np.ascontiguousarray(w[name], np.float32))

    h = xt + V.attention_orv(V.rmsnorm_orv(xt, ln1, eps),
                            M("wq"), M("wk"), M("wv"), M("wo"),
                            ct, st, rot_dim, n_q, n_kv,
                            causal=(add_mask is None), attn_mask=add_mask)
    y = h + V.situ_glu_orv(V.rmsnorm_orv(h, ln2, eps),
                           M("w1"), M("w2"), M("w3"), beta1, beta2)
    return y
