# ORV 侧首个 V3 训练步（2026-11-13 实测）

> 这是 ORV **第一次**跑通 V3 架构的完整训练步：AdamW、权重真更新、损失真反传。
> 但**不是** V3 1.5B，也**不是**与 PyTorch 同配置 —— 差别逐条列在 §3，不要当作对拍数据引用。

## 1. 怎么跑起来的

`tools/v3_orv_train.py` 组装的骨架：

```
x = emb[ids]                                     # 整数索引查表（含 scatter-add 反向）
for layer:
    h = x + GlobalAttention(rmsnorm(x))          # 部分 RoPE + GQA + 因果
    hn = rmsnorm(h)
    eids, gate = route_top1(hn, router)          # top_k=1
    x = h + MoE_dispatch(hn, eids, gate)         # scatter_add 按专家散回
logits = head(rmsnorm(x));  loss = CE(logits, next_token)
loss.backward(); AdamW.step()
```

关键打通的三块：

| 能力 | 实现 | 验证 |
| --- | --- | --- |
| **取行** | `x[ids]`（整数数组索引） | 反向按出现次数累加，未选中行梯度为 0 |
| **散回** | `v3_scatter.scatter_add`（运行时 `scatter_add_rows` + 自定义 VJP） | V100 上与 numpy 基准 **0.0** 误差（前向/均匀反向/加权反向） |
| **专家分派** | 排序 → 逐专家切片 → `cat` → `scatter_add` | 与 numpy 参考 **5.07e-07** |

## 2. 实测数字（真实训练步）

| 配置 | 峰值显存 | 吞吐 | 每步耗时 | 步数 |
| --- | --- | --- | --- | --- |
| d=256 / 4 层 / 4 专家 / seq 128 / vocab 4096 | **530 MiB** | **2015 tok/s** | 0.064 s | 3 |
| d=512 / 8 层 / 8 专家 / seq 512 / vocab 4096 | **4068 MiB** | **1189 tok/s** | 0.43 s | 5 |

loss ≈ 8.39（`ln(4096) = 8.32`）—— 随机初始化下的正常值，5 步内看不出下降属于预期。

## 3. 与真实 V3 的差别（必须先看这一节）

| 项 | 本次 | 真 V3 |
| --- | --- | --- |
| 注意力 | **GlobalAttention** | 部分层是 **CSA**（压缩稀疏）+ 部分 Global |
| 专家 | **top_k = 1** | top_k = 2 + 共享专家 |
| 路由 | `max` 代替 `topk`（CUDA topk 未实现） | `topk(s_all + bias)` + 偏置队列 |
| BlockAttnRes | **未实现** | 有（块级注意力残差） |
| 规模 | 0.02B 级（d=512） | 1.5B |
| 权重 | **随机初始化** | 真实 checkpoint |

**因此：这些数字证明"ORV 能训 V3 骨架"，不能用来算与 PyTorch 的显存倍数。**

## 4. 途中踩的坑（都会重复出现）

1. `T.topk` **CUDA 未实现**（只有 CPU）⇒ top_k=1 用 `max + argmax` 代替。
2. ORV 切片**不支持 `None`（newaxis）**⇒ 列向量用 `reshape(-1, 1)`。
3. `W3[e]` 是**每专家一份**（三维），我一开始建成二维，得到 `(2,)` 这种莫名形状。
4. `P["head"]` 是**张量不是模块**，不能当函数调用。
5. 对已是张量的 `W1[e]` 再包 `T.tensor(...)` 会走 `np.ascontiguousarray` 而报错。

## 5. 下一步（要拿到与 PyTorch 的同配置对拍）

1. 在 PyTorch 侧用 **`TGAIV3Config` 手工构造与上表相同的配置**（d/层/专家/seq/vocab），
   两边都用随机初始化 ⇒ 可比的是**显存/速度**（loss 不可比，因为初始化不同）；
2. 两侧都开 `grad_ckpt`、同为 AdamW、同 batch、同 seq、同步数；
3. 多组 (seq, 层数) 扫一遍，出**显存上升曲线**与**速度下降曲线**；
4. 再补 CSA / top_k=2 / BlockAttnRes，把差距收窄到"只剩引擎差异"。
