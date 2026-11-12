# 新测试者迁移协议（2026-11-12）

> 目标：**让第一位外部测试者在十分钟内跑出一个可对比的数字**，或者准确告诉我们卡在哪。
> 这里假设你只有 `pip`，没有 clone 仓库。

## 0. 用哪个版本

```
pip install -U tgai-orv==0.9.53
```

必须是 **0.9.53**。早期版本在这几个地方会直接撞上（都是外部用户实测出来的）：

| 版本 | 修了什么 |
| --- | --- |
| 0.9.50 | NVRTC 的 `nvrtc-builtins64_*.dll` 找不到（不再需要手动改 PATH） |
| 0.9.51 | `Tensor.cos/sin`；SDPA 接受 `[B,H,S,Dh]` 与 `is_causal=` |
| 0.9.52 | CUDA 上 `x[idx]` 数组索引（embedding 查表会用到） |
| **0.9.53** | **梯度检查点重做**：20 层 / 606M 的峰值从 31.3 GiB 降到 **7.6 GiB**。之前 16GB 卡跑不了就是这个 |

## 1. 第一步：环境自检（30 秒）

```bash
python -m tgai_go.train.doctor
```

正常会逐项打 `PASS`，最后一张汇总表。
**如果它退出了并且打印"自检没跑完 —— 环境不满足"，把那一段整体贴回来即可**，不用自己判断原因。

快速版（跳过耗时项）：`python -m tgai_go.train.doctor --quick`

## 2. 第二步：最小训练（1 分钟）

`pip` 装的包里没有 `examples/`，所以请把下面这段存成 `orv_smoke.py` 再运行。
它**只依赖 tgai_go.train 和 numpy**，跑一个 2 层小模型 20 步，打印 loss / 显存 / 速度。

```python
import time
import numpy as np
import tgai_go.train as T
from tgai_go.train import functional as Fn

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)

V, D, L, NH, FF, S, B = 512, 256, 2, 4, 1024, 128, 4
DH = D // NH

def dev(n, m):
    return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

emb_w = dev(V, D)
qkv = [dev(3 * D, D) for _ in range(L)]
o = [dev(D, D) for _ in range(L)]
g = [dev(FF, D) for _ in range(L)]
u = [dev(FF, D) for _ in range(L)]
dn = [dev(D, FF) for _ in range(L)]
head = dev(V, D)
for t in [emb_w, head] + qkv + o + g + u + dn:
    t.requires_grad = True

ids = rng.integers(0, V, (B, S)).astype(np.int64)
tgt = rng.integers(0, V, (B, S)).astype(np.int64)

emb_layer = T.nn.Embedding(V, D, weight=T.nn.Parameter(np.zeros((V, D), np.float32)))
emb_layer.weight._gpu = emb_w._gpu
x = emb_layer(ids.reshape(-1)).reshape(B, S, D)

for i in range(L):
    z = x.reshape(-1, D) @ qkv[i].T
    q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
    y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                        k.reshape(B * NH, S, DH),
                                        v.reshape(B * NH, S, DH), is_causal=True)
    x = x + (y.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(-1, D) @ o[i].T).reshape(B, S, D)
    f = x.reshape(-1, D)
    h = Fn.silu(f @ g[i].T) * (f @ u[i].T)
    x = x + (h @ dn[i].T).reshape(B, S, D)

params = [emb_w, head] + qkv + o + g + u + dn
opt = T.optim.AdamW(params, lr=1e-3)
for step in range(20):
    t0 = time.perf_counter()
    logits = (x.reshape(-1, D) @ head.T).reshape(-1, V)
    loss = Fn.cross_entropy(logits, tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    rt.sync()
    if step in (0, 19):
        st = rt.pool_stats()
        print("step %2d  loss %.6f  峰值活张量 %.3f GiB  耗时 %.1f ms"
              % (step, float(np.asarray(loss.to_numpy()).reshape(-1)[0]),
                 st["peak_live_bytes"] / (1 << 30),
                 (time.perf_counter() - t0) * 1e3))
```

跑起来后请顺手再跑一次**关掉检查点的版本**做对照（把层数改成 8，用 `T.checkpoint` 包一层）：

```python
x = T.checkpoint(lambda h: block(h), x)
```

## 3. 请贴回来的东西

把下面这张表填好发回来（有数字比有描述有用一百倍）：

```
【环境】
GPU 型号 / 显存：
驱动版本（nvidia-smi 左上角）：
CUDA 版本（nvidia-smi 右上角）：
系统：Windows / Linux，版本
torch 装了吗，版本：
tgai-orv 版本：0.9.53

【doctor】
（整段粘贴，含最后的汇总表）

【最小训练】
loss 首 → 末：
峰值活张量（GiB）：
每步耗时（ms）：

【失败的项】
（哪一步报的错，整段错误信息；有 TG_DEBUG=1 的日志更好）
```

## 4. 已知会撞的地方（先说清楚，省你时间）

| 现象 | 原因 / 怎么办 |
| --- | --- |
| `找不到 nvcuda.dll` | 驱动没装好 / 跑在无 GPU 的机器上。**0.9.50 起不再需要手动改 PATH** |
| `failed to open nvrtc-builtins64_*.dll` | 同上，0.9.50 已修；仍出现请贴 `doctor` 输出 |
| SDPA 维度报错 | 0.9.51 起接受 `[B,H,S,Dh]`；更早版本要自己展平成 `[B*H,S,Dh]` |
| 想省显存但没省 | `sdpa_lowmem` **只支持 `dh==64`**，`dh=96/128` 会静默退回全量注意力 |
| 开 fp16 后 loss 不对 | fp16 反向也走，但**长 K 的 dW 有精度风险**（k > 8192 时误差可达 2e-2）。用 `TG_FP16_MAX_K` 限制，或对该层关掉 |
| 显存比 PyTorch 多 | **确实如此**（实测 1.2–1.9×）。ORV 用显存换速度和"能装下更大的模型"，不是省显存 |
| Windows 上 Tesla 卡不能用 Vulkan | 数据中心卡常处于 **TCC** 模式，Vulkan 按设计不可用（见 `HARDWARE_INVENTORY.md`）|

## 5. 如果你愿意多测一点

| 想测什么 | 怎么做 |
| --- | --- |
| 和 PyTorch 直接对比 | 仓库里 `tools/h2h_clean.py`（需要 clone；`--impl orv|torch`） |
| 完整验收（形状矩阵 + 精度 + 速度） | 仓库里 `python tools/accept.py --quick` |
| 你的真实模型能不能跑 | 把模型结构发我，我按通用 API 给一份 ORV 版（**不会为你的模型特判**，见 `GENERALITY_GAPS.md`）|

## 6. 我们最想知道的三件事

1. **你的卡上 `doctor` 全 PASS 吗** —— 尤其是 sm_60/75/89 这些我们没实测过的代际。
2. **最小训练能不能跑通、loss 是否下降。**
3. **你有没有同时装了 AMD 和 NVIDIA 卡的机器**（笔记本 Optimus 也算）——
   这是 TG Link（跨厂商 P2P）唯一能拿到真数据的地方，云上租不到。
