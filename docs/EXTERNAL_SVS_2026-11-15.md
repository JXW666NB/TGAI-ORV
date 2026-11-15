# 把虚拟歌手项目（yuaz-sgr）接到 ORV：做到哪一步了

> 2026-11-15。对象：`https://github.com/Yuaz-Club/yuaz-sgr`（DDSP + RVQ + 潜空间流匹配 GAN，
> 虚拟歌手 / 歌声合成，**不是 LLM**）。仓库克隆在 `D:\DSH-Workspace\_ext\yuaz-sgr`（HEAD `1faa71a`）。

## 一、为什么必须写插件

ORV 是通用训练引擎，**没有音频能力**。盘点他们的模型路径（encoder / rvq / ddsp_decoder /
conditioning / svs_model / losses / trainer 的 AE 段），缺的是音频与信号处理这一批：

| 类别 | 缺的接口 |
| --- | --- |
| 频域 | `torch.stft`、`torch.istft`、`torch.hann_window` |
| 形状 | `F.interpolate`、`F.pad`、`F.avg_pool1d` |
| 逐元素 | `sigmoid`、`softplus`、`log1p`、`log10`、`pow` |
| 扫描/索引 | `torch.cumsum`、`torch.gather`、`torch.searchsorted` |
| 损失 | `F.l1_loss`、`F.mse_loss`、`F.binary_cross_entropy(_with_logits)` |

## 二、插件已实现并通过数值对拍

`examples/plugins/svs_audio_ops.py`（组合式，**不写反向**，梯度由 ORV 的 autograd 自动得出）：

- `stft_mag`：**不需要复数张量** —— 分帧用 `conv1d`（恒等核 + stride=hop），
  再乘预先算好的余弦/正弦 DFT 基矩阵，`|X| = sqrt(re²+im²)`
- `cumsum`：三角矩阵乘；`pad1d`（constant/replicate/reflect）；`hann_window`
- `sigmoid`（用 tanh 表达，数值稳定）、`softplus`、`log1p`、`log10`、`pow`、`leaky_relu`
- `l1_loss` / `mse_loss` / `bce_with_logits` / `bce` / `avg_pool1d`
- `gather_last` / `isfinite_np`

**对拍结果**：`tools/verify_svs_ops.py` 拿真 PyTorch 当考官，**39 项全过、0 失败**，
相对误差约 1e-7。覆盖：各逐元素算子的前向+反向、三种 pad 模式、cumsum 前向+反向、
损失与其反向、`hann_window`、以及 **STFT 在 1024/512/2048 三档的前向、形状、反向**。

## 三、速度对比（B=2、1×96000，6 档多分辨率 STFT 损失，前向+反向，V100）

| 实现 | 时间 | loss |
| --- | ---: | --- |
| ORV（组合式插件） | **1690.7 ms** | 2.326434 |
| PyTorch（cuFFT） | **65.6 ms** | 2.326434 |
| 比 | **25.8× 慢** | 相对差 1.0e-07 ✓ |

**为什么慢**：组合式 STFT 是 O(n_fft²) 的基矩阵乘，而分帧用恒等核 `conv1d` 更是
O(n_fft² × 帧数) 的浪费（2048² × 188 ≈ 7.9 亿次乘加，只为把数据摆成帧）；
PyTorch 走 cuFFT，复杂度 O(n log n)。**数值对但性能差一个数量级。**

## 四、出路已确认：cuFFT 就在机器上

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cufft64_11.dll
```

⇒ 不需要自己写 FFT 内核：用 `ctypes` 绑 `cufftPlanMany` + `cufftExecR2C`，
把分帧结果直接送进 cuFFT，STFT 就能接近 torch 的速度。
分帧也要改掉恒等卷积（改成按帧切片 + `cat`，把 O(n²) 降到 O(L)）。

## 五、顺带发现的引擎 BUG（重要，未修）

**ORV 的惰性转置 `_lazy_T` 会被绝大多数算子忽略，静默给出错数据。**

复现：`x = arange(6).reshape(3,2)`，`t = x.T`

| 表达式 | 结果 | 期望 |
| --- | --- | --- |
| `t.to_numpy()` | ✓ 正确 | — |
| `t.reshape(6)` | `[0,1,2,3,4,5]` | `[0,2,4,1,3,5]` |
| `t + 1`、`t * 2`、`-t`、`t.exp()` | 全部按**未转置**的缓冲算 | 应按转置后 |
| `t.sum(axis=1)` | 形状都不对 | `[6, 9]` |
| `t.T` | 不等于 `x` | 应等于 `x` |
| `t.contiguous().to_numpy()` | 表面正确（仍带 `_lazy_T`）| — |

根因：`_transpose_cuda`（`__init__.py:1128`）不物化，只打 `_lazy_T` 标记；
只有 `matmul`（走 `gemm_nt`）和 `to_numpy`（`__init__.py:1933`）认这个标记，
其余算子直接读 `self._gpu`。

影响：任何 `x.transpose(...)` 之后接逐元素/形状/归约操作都会算错 —— 这对迁移项目是致命的
（他们代码里 `transpose(1,2)` 用得很多）。

插件里已**绕开**（不用 `.T` 后接 reshape），但引擎本身应当修：
最小改法是给 `_gpu` 加一个属性 getter，遇到 `_lazy_T` 先物化；同时让 `matmul` 继续读原始缓冲以保留优化。

## 六、还没做完的（要跑通整个 AE 训练步还缺）

按规格书，AE 步除了上述算子还需要：

| 缺的能力 | 用在哪 | 难度 |
| --- | --- | --- |
| **双向 GRU** | encoder（512→256×2）| 中（可先用逐时刻循环实现，慢）|
| **MultiheadAttention** | DDSP 的 Emformer（4 层、8 头）| 中（ORV 只有 CausalSelfAttention）|
| **istft** | DDSP 谐波/噪声合成，**训练路径上就有** | 中（可用 cuFFT 的 C2R + 重叠相加）|
| `interpolate`（linear/nearest）| 全模型 28 处 | 低（nearest=重复，linear=矩阵插值）|

⇒ 全模型端到端跑通是**多日工程**；当前可选的两条路：
（a）先做 cuFFT 版 STFT/iSTFT，把音频这一块做到与 PyTorch 同速；
（b）补齐 GRU/MHA/interpolate，让整个 AE 步跑起来再比总步时。
