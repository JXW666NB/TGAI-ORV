# 算子覆盖清单（以 PyTorch 为基准）

> 由 `python tools/pytorch_op_surface.py --write` 生成，基准是本机 `torch` 的真实公开面。

## 总览

| 项 | 数量 |
| --- | --- |
| PyTorch 公开函数 | 1016 |
| PyTorch Tensor 方法 | 604 |
| **Tier-1 训练必需算子** | **64** |
| 其中我们已具备 | **46（72%）** |

## Tier-1：训练别人的模型所必需

| PyTorch 算子 | 我们的对应 | 状态 | 备注 |
| --- | --- | --- | --- |
| `torch.matmul` | `@ (Tensor.__matmul__)` | 有 |  |
| `torch.addmm` | — | 缺 | bias 融合乘加，训练热点 |
| `torch.bmm` | `batched @` | 有 | 注意力里的批量矩阵乘 |
| `torch.einsum` | — | 缺 | 很多实现用它写注意力 |
| `torch.nn.functional.linear` | `nn_ops.linear` | 有 | 已验证 |
| `torch.nn.functional.rms_norm` | `nn_ops.rms_norm` | 有 | 已验证（Qwen2 对齐用到） |
| `torch.nn.functional.layer_norm` | `nn_ops.layer_norm` | 有 | 已验证 |
| `torch.nn.functional.batch_norm` | — | 缺 | CNN 系，训练/推理行为不同 |
| `torch.nn.functional.silu` | `F.silu` | 有 |  |
| `torch.nn.functional.gelu` | `nn_ops.gelu` | 有 | 只支持 tanh 近似 |
| `torch.nn.functional.relu` | — | 缺 | 基础 |
| `torch.nn.functional.softmax` | `F.softmax` | 有 |  |
| `torch.nn.functional.log_softmax` | `F.log_softmax` | 有 |  |
| `torch.nn.functional.scaled_dot_product_attention` | `F.scaled_dot_product_attention` | 有 | GQA/因果掩码要覆盖 |
| `torch.nn.functional.cross_entropy` | `F.cross_entropy` | 有 |  |
| `torch.nn.functional.nll_loss` | — | 缺 | log_softmax 之后常用 |
| `torch.nn.functional.mse_loss` | — | 缺 | 回归/对齐 |
| `torch.sum` | `Tensor.sum` | 有 | 需要支持 dim/keepdim |
| `torch.mean` | `Tensor.mean` | 有 | 已支持 dim/keepdim |
| `torch.max` | `Tensor.maximum` | 有 | 逐元素有；沿维取最大值+索引没有 |
| `torch.min` | — | 缺 |  |
| `torch.amax` | — | 缺 |  |
| `torch.norm` | — | 缺 | 梯度裁剪要用 |
| `torch.add` | `Tensor.__add__` | 有 |  |
| `torch.mul` | `Tensor.__mul__` | 有 |  |
| `torch.div` | `Tensor.__truediv__` | 有 | float/Tensor 目前不支持（反向标量除法缺） |
| `torch.sub` | `Tensor.__sub__` | 有 |  |
| `torch.pow` | `nn_ops.pow` | 有 | 整数幂精确，非整数走 exp/log |
| `torch.exp` | `Tensor.exp` | 有 |  |
| `torch.log` | `Tensor.log` | 有 |  |
| `torch.sqrt` | `Tensor.sqrt` | 有 |  |
| `torch.rsqrt` | `nn_ops.rsqrt` | 有 | 已验证 |
| `torch.tanh` | `nn_ops.tanh` | 有 | 已验证 |
| `torch.erf` | — | 缺 | 精确 GELU 要用 |
| `torch.sigmoid` | `nn_ops.sigmoid` | 有 | 已验证 |
| `torch.clamp` | `Tensor.clamp` | 有 |  |
| `torch.abs` | `Tensor.abs` | 有 |  |
| `torch.reshape` | `Tensor.reshape` | 有 |  |
| `torch.view` | `Tensor.view` | 有 |  |
| `torch.transpose` | `Tensor.transpose` | 有 |  |
| `torch.permute` | `Tensor.permute` | 有 |  |
| `torch.cat` | `T.cat` | 有 |  |
| `torch.stack` | `nn_ops.stack` | 有 | 已验证 |
| `torch.split` | `nn_ops.split` | 有 | 已验证 |
| `torch.chunk` | — | 缺 |  |
| `torch.gather` | — | 缺 | embedding 查表的最小依赖 |
| `torch.embedding` | `nn_ops.embedding` | 有 | 有，但**不支持反向**（缺 gather 内核） |
| `torch.repeat_interleave` | `Tensor.repeat_interleave` | 有 | GQA 扩头 |
| `torch.triu/tril` | `nn_ops.triu` | 有 | triu 已验证；tril 未实现 |
| `torch.where` | `nn_ops.where` | 有 | 已验证 |
| `torch.tensor` | `T.tensor` | 有 |  |
| `torch.zeros/ones/empty` | `T.zeros/T.ones` | 有 |  |
| `torch.from_numpy` | `T.tensor(numpy)` | 有 |  |
| `Tensor.to(dtype/device)` | `Tensor.to_numpy / .data` | 有 | dtype 转换目前只能绕 |
| `Tensor.float/half/bfloat16` | — | 缺 | 真 AMP 的前置 |
| `torch.randn` | — | 缺 | 初始化/dropout |
| `torch.manual_seed` | — | 缺 | 可复现 |
| `Tensor.backward` | `Tensor.backward` | 有 |  |
| `Tensor.detach` | `Tensor.detach` | 有 |  |
| `torch.optim.AdamW` | `optim.AdamW` | 有 |  |
| `torch.optim.SGD` | — | 缺 | 最常用优化器之一 |
| `torch.nn.utils.clip_grad_norm_` | `nn_ops.clip_grad_norm_` | 有 | 已验证 |
| `torch.arange` | — | 缺 | 位置编码 |
| `torch.cumsum` | — | 缺 | 部分注意力实现用 |

## PyTorch 函数面分类计数

| 类别 | 数量 |
| --- | --- |
| 类型/设备 | 507 |
| 逐元素 | 95 |
| 归约 | 80 |
| 形状 | 78 |
| 激活 | 59 |
| 矩阵乘 | 49 |
| 归一化 | 34 |
| 损失 | 30 |
| 随机 | 25 |
| 索引 | 23 |
| 其他 | 18 |
| 卷积 | 18 |

## 说明

- 「有」= 公开 API 里能找到对应实现；不等于已对齐 PyTorch 语义或用例覆盖完整。
- 很多能力我们**内核里有、但没公开成算子**（如 rmsnorm/rope 在 `cuda.py` 里），清单里按「缺」计，因为它们不能被用户代码直接调用。
- 下一步：按 Tier-1 顺序逐项实现，每项配 ①与 PyTorch 的数值对齐测试 ②速度对比 ③错误信息三要素。
