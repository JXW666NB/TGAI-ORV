# 长上下文实现路线（第二版，2026-11-17）

> 第一版设计（`LONG_CONTEXT_DESIGN.md`）建议写 CSA 专用分块 CUDA 内核。
> 读完源码后发现**有更省的路**：用已验证的 checkpoint 模式，不必写内核。

## 一、关键发现：ORV 已有"梯度逐位等价"的重算模式

`v3_train_cli._checkpoint_layer`：

```python
def _checkpoint_layer(fn, x):
    out = fn(x.detach())                          # 不建图 ⇒ 中间量随即释放
    res = Tensor._from_gpu(out._gpu, requires_grad=True, _parents=(x,))
    def bw(g, grads):
        sub = fn(x)                               # 反向时用原输入**重跑**，建图
        x.grad = None
        sub.backward(g)                           # 子图反传
        if x.grad is not None:
            _acc(grads, x, x.grad._gpu); x.grad = None
    res._backward = bw
    return res
```

原型验证记录：`lab/prof/ckpt_proto.py`，**max_abs = 0.000e+00**（梯度逐位一致）。
注意 `Tensor.detach()` 在 ORV 上是**拷贝**，所以重算要付一份输入的代价。

## 二、把同一个模式用在注意力上 ⇒ 天然 O(seq)

CSA 全局分支：q `(B,nq,T,dk)`、k/v `(B,nq,Tc,dk)`，`Tc=T/c`。物化的 `scores/p` 是 O(T·Tc)。

分块 + 重算：

```
前向：把 q 按 query 行切成 cq 一块，逐块算 _attn，只保留输出（不建图）
反向：逐块重跑 _attn 并反传 —— 每个 chunk 的 scores/p 都活不过一个 chunk
```

* **前向峰值**：`(B,nq,cq,Tc)` ⇒ 对 T **线性**
* **反向峰值**：同样只有一个 chunk 的中间量 ⇒ 线性
* **数学**：重算是精确的 ⇒ 与前向整块计算等价（与 `ckpt_proto` 同源，可复现逐位一致）
* **代价**：注意力部分多一次前向计算（约 1.33× 注意力 FLOPs），可用 `cq` 调

### 需要处理的两个细节（已想清）

1. **k/v 的梯度**：分块后每个 chunk 都要对 k/v 贡献梯度。做法是**只 detach 一次**
   `kd, vd = k.detach(), v.detach()`（各 O(T·dk)，**线性**，16K 时仅几十 MB），
   逐块反传后把 `kd.grad/vd.grad` 累加回 k/v 的原始节点。
2. **上游梯度按 chunk 切**：`bw(g, grads)` 里的 `g` 是 **buffer** 不是 Tensor，
   需要 `Tensor._from_gpu(g, requires_grad=False)` 再切片，或直接用 `rt.sub_view`。

## 三、为什么仍然建议同时做"惰性掩码"

`_mask_global_fixed` 结构只是 chunk 粒度因果（`ik <= iq//c`），却物化了
`B·T·Tc·4` 字节：**16K 时 268 MB、64K 时 4.3 GB**。

分块之后这个掩码仍是整块驻留。两种处理：
* **过渡**（低风险）：保持整块掩码，先拿到"分块 + 重算"的收益。
  16K 时掩码 268 MB、chunk 的 scores 只有 `cq·Tc` ⇒ 已经是 O(T)，只剩掩码这一项是 O(T²)。
* **彻底**：把掩码换成"chunk 粒度因果"的语义参数（`causal_chunk=c`），内核里按需判断；
  seg 情形改用文档区间表。

**建议顺序**：先做分块+重算（收益最大、改动集中在 `v3_csa._attn` 调用处），
再决定是否把掩码也惰性化（取决于 64K 时 4.3 GB 是否可接受）。

## 四、开关与回退

```python
T.set_csa_attn(chunk=1024)        # 默认开，chunk=0 表示关（走旧物化路径）
# 或环境变量 TG_CSA_ATTN_CQ=1024 / TG_CSA_ATTN_CQ=0
```

两条路径都要有测试：同一输入下 logits 与梯度逐位一致（模板见 `ckpt_proto.py`）。

## 五、验收不变（与目标一致）

显存阶数 p ≤ 1.15；64K 可训；对拍 ≤1e-5；814 passed；4 层 loss 基线不退化；可回退；发版。
