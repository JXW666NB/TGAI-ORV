# 算子覆盖清单（以 PyTorch 为基准）

> 由 `python tools/pytorch_op_surface.py --write` 生成，基准是本机 `torch` 的真实公开面。

## 总览

| 项 | 数量 |
| --- | --- |
| PyTorch 公开函数 | 1016 |
| PyTorch Tensor 方法 | 604 |
| **Tier-1 训练必需算子** | **63** |
| 其中我们已具备 | **62（98%）** |

## Tier-1：训练别人的模型所必需

| PyTorch 算子 | 我们的对应 | 状态 | 备注 |
| --- | --- | --- | --- |
| `torch.matmul` | `@ (Tensor.__matmul__)` | 有 |  |
| `torch.addmm` | `nn_ops.addmm` | 有 | 组合实现，已验证形状检查 |
| `torch.bmm` | `batched @` | 有 | 注意力里的批量矩阵乘 |
| `torch.einsum` | `nn_ops.einsum` | 有 | 支持 1~2 操作数常见形式，已验证 |
| `torch.nn.functional.linear` | `nn_ops.linear` | 有 | 已验证 |
| `torch.nn.functional.rms_norm` | `nn_ops.rms_norm` | 有 | 已验证（Qwen2 对齐用到） |
| `torch.nn.functional.layer_norm` | `nn_ops.layer_norm` | 有 | 已验证 |
| `torch.nn.functional.silu` | `F.silu` | 有 |  |
| `torch.nn.functional.gelu` | `nn_ops.gelu / contrib.gelu_exact` | 有 | tanh 近似在核心，精确版在 contrib |
| `torch.nn.functional.relu` | `nn_ops.relu` | 有 | 已验证 |
| `torch.nn.functional.softmax` | `F.softmax` | 有 |  |
| `torch.nn.functional.log_softmax` | `F.log_softmax` | 有 |  |
| `torch.nn.functional.scaled_dot_product_attention` | `F.scaled_dot_product_attention` | 有 | GQA/因果掩码要覆盖 |
| `torch.nn.functional.cross_entropy` | `F.cross_entropy` | 有 |  |
| `torch.nn.functional.nll_loss` | `nn_ops.nll_loss` | 有 | log_softmax 之后常用 |
| `torch.nn.functional.mse_loss` | `nn_ops.mse_loss` | 有 | 回归/对齐 |
| `torch.sum` | `Tensor.sum` | 有 | 需要支持 dim/keepdim |
| `torch.mean` | `Tensor.mean` | 有 | 已支持 dim/keepdim |
| `torch.max` | `nn_ops.amax / Tensor.maximum` | 有 | 逐元素有；沿维取最大值+索引没有 |
| `torch.min` | `nn_ops.amin` | 有 | 已验证 |
| `torch.amax` | `nn_ops.amax` | 有 | 已验证 |
| `torch.norm` | `nn_ops.norm` | 有 | p=2 已验证 |
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
| `torch.erf` | `contrib.erf` | 有 | 官方 contrib 插件 |
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
| `torch.chunk` | `nn_ops.chunk` | 有 | 已验证 |
| `torch.gather` | `nn_ops.embedding(仅前向)` | 有 | embedding 查表的最小依赖 |
| `torch.embedding` | `nn_ops.embedding` | 有 | 有，但**不支持反向**（缺 gather 内核） |
| `torch.repeat_interleave` | `Tensor.repeat_interleave` | 有 | GQA 扩头 |
| `torch.triu/tril` | `nn_ops.triu / nn_ops.tril` | 有 | triu 已验证；tril 未实现 |
| `torch.where` | `nn_ops.where` | 有 | 已验证 |
| `torch.tensor` | `T.tensor` | 有 |  |
| `torch.zeros/ones/empty` | `T.zeros/T.ones` | 有 |  |
| `torch.from_numpy` | `T.tensor(numpy)` | 有 |  |
| `Tensor.to(dtype/device)` | `Tensor.to_numpy / .data` | 有 | dtype 转换目前只能绕 |
| `Tensor.float/half/bfloat16` | — | 缺 | 真 AMP 的前置 |
| `torch.randn` | `nn_ops.randn` | 有 | 已验证形状/分布 |
| `torch.manual_seed` | `nn_ops.manual_seed` | 有 | 已验证可复现 |
| `Tensor.backward` | `Tensor.backward` | 有 |  |
| `Tensor.detach` | `Tensor.detach` | 有 |  |
| `torch.optim.AdamW` | `optim.AdamW` | 有 |  |
| `torch.optim.SGD` | `nn_ops.sgd_step` | 有 | 最常用优化器之一 |
| `torch.nn.utils.clip_grad_norm_` | `nn_ops.clip_grad_norm_` | 有 | 已验证 |
| `torch.arange` | `nn_ops.arange` | 有 | 已验证 |
| `torch.cumsum` | `nn_ops.cumsum` | 有 | 已验证(含反向) |

## 范围裁剪：我们不做哪些（明确不做，不是待办）

我们是 **LLM 训练引擎**，不是通用张量库。PyTorch 公开函数 1016 个，其中 **240 个** 属下列领域，**永远不在范围内**；剩下 776 个才进入考量的池子。

| 领域 | 例子 | 为什么不做的 |
| --- | --- | --- |
| 卷积/视觉 | `conv2d / max_pool2d / interpolate` | LLM 没有卷积；视觉任务不在产品范围（89 个） |
| 线性代数/科研 | `svd / qr / cholesky / fft` | 科研张量库的需求，训练用不到（101 个） |
| 概率分布 | `Normal / Categorical` | 需要时用 numpy 采样即可，不值得进内核（9 个） |
| 稀疏 | `to_sparse / spmm` | LLM 训练用稠密权重（9 个） |
| 量化/图/编译器 | `quantize / torch.compile` | 我们有 NVRTC + CUDA Graph 这条自己的路（17 个） |
| 视觉/音频 IO | `imread / spectrogram` | decode 归数据管线，不归训练内核（1 个） |
| 其他非训练 | `cdist / histogram / vmap` | 非训练必需（14 个） |

## PyTorch 函数面分类计数（仅范围内）

| 类别 | 数量 |
| --- | --- |
| 类型/设备 | 409 |
| 逐元素 | 84 |
| 形状 | 68 |
| 归约 | 43 |
| 归一化 | 32 |
| 矩阵乘 | 29 |
| 损失 | 28 |
| 随机 | 24 |
| 激活 | 24 |
| 索引 | 23 |
| 其他 | 12 |

## 说明

- 「有」= 公开 API 里能找到对应实现；不等于已对齐 PyTorch 语义或用例覆盖完整。
- 很多能力我们**内核里有、但没公开成算子**（如 rmsnorm/rope 在 `cuda.py` 里），清单里按「缺」计，因为它们不能被用户代码直接调用。
- 下一步：按 Tier-1 顺序逐项实现，每项配 ①与 PyTorch 的数值对齐测试 ②速度对比 ③错误信息三要素。
