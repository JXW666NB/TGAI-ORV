# -*- coding: utf-8 -*-
"""GPU 融合的 int8 分块 AdamW（省显存的动量存储）。

与 `int8_adamw.py` 的区别：**那个是主机侧 numpy 实现** —— 每步要把 m/v 从显存
拷回主机、量化、再拷回去。1.5B 上光这一来一回就是约 12 GiB 的 PCIe 流量，
所以它在 CLI 里虽然存在，实际上不能用于大模型。

这个是**全显存**实现：量化/更新/反量化都在 `adamw_multi_int8` 一个内核里完成，
一次发射覆盖全部参数（与 `adamw_multi_bf16` 同一套指针数组管线）。

显存占用（每参数）：int8 的 m + int8 的 v + 每 block 一个 fp32 scale
  BLOCK=64  ⇒ 2 + 8/64 = **2.125 字节/参数**
  对照：fp32 8 字节、bf16 4 字节、fp16 4 字节。

**这是有损量化**：m/v 从 fp32 压到 int8（每 block 独立 scale）必然丢精度。
等价性靠实测判断（同种子跑 20 步比 loss），不达标就不要用。
"""

import numpy as np

__all__ = ["Int8AdamWGpu"]


def _rt():
    import tgai_go.train as T
    return T._cuda()


class Int8AdamWGpu(object):
    """int8 分块 AdamW（全显存）。

    参数语义与 `torch.optim.AdamW` 一致：`lr` / `betas` / `eps` / `weight_decay`，
    解耦权重衰减（`p -= lr*(upd + wd*p)`），偏差校正 `1-b1^t` / `1-b2^t`。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, block=64, state="m8v16"):
        self.params = list(params)
        self.lr = float(lr)
        self.b1 = float(betas[0])
        self.b2 = float(betas[1])
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.block = int(block)
        self.state = str(state)
        if self.state not in ("m8v16", "int8"):
            raise ValueError("state 只能是 'm8v16'（推荐）或 'int8'（实测发散）"
                             % () + "。修法：用 state='m8v16'")
        if self.block <= 0 or self.block % 32 != 0:
            raise ValueError("block 必须是 32 的正整数倍（当前 %d）" % self.block
                             + "。修法：用 block=32/64/128/256")
        self.t = 0
        self.mq, self.vq, self.ms, self.vs = [], [], [], []
        rt = _rt()
        for p in self.params:
            if getattr(p, "device", "cpu") != "cuda":
                raise ValueError("Int8AdamWGpu 只支持 CUDA 张量"
                                 + "。修法：先 T.set_default_device('cuda')")
            n = int(p._gpu.n)
            nb = (n + self.block - 1) // self.block
            self.mq.append(rt.zeros((n,), dtype=np.int8))
            self.ms.append(rt.zeros((nb,), dtype=np.float32))
            if self.state == "m8v16":
                # v 用 bf16（uint16 位模式）：指数范围同 fp32，不下溢
                self.vq.append(rt.zeros((n,), dtype=np.uint16))
                self.vs.append(None)
            else:
                self.vq.append(rt.zeros((n,), dtype=np.int8))
                self.vs.append(rt.zeros((nb,), dtype=np.float32))

    def bytes_per_param(self):
        if self.state == "m8v16":
            return 1.0 + 2.0 + 4.0 / float(self.block)      # m int8 + v bf16 + scale
        return 2.0 + 2.0 * 4.0 / float(self.block)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, grad_scale=1.0):
        """一步更新：一次内核发射覆盖所有带梯度的参数。

        `grad_scale`：梯度裁剪的缩放系数，**由内核在读梯度时乘上去**（默认 1.0，
        与旧行为逐位相同）。这样调用方不必先对 6.8 GB 梯度做一趟原地缩放；
        物理缩放与内核内乘同一个数**同为一次 IEEE 乘**，结果逐位等价。
        """
        self.t += 1
        rt = _rt()
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        gs = float(grad_scale)
        ents = []
        for i, p in enumerate(self.params):
            if p.grad is None or not getattr(p, "requires_grad", True):
                continue
            if self.mq[i] is None:
                continue
            if self.state == "m8v16":
                ents.append((p._gpu, p.grad._gpu, self.mq[i], self.ms[i],
                             self.vq[i]))
            else:
                ents.append((p._gpu, p.grad._gpu,
                             self.mq[i], self.vq[i], self.ms[i], self.vs[i]))
        if not ents:
            return
        if self.state == "m8v16":
            rt.adamw_multi_m8v16(ents, self.b1, 1.0 - self.b1,
                                 self.b2, 1.0 - self.b2, self.lr,
                                 self.lr * self.wd, self.eps, bc1, bc2,
                                 self.block, grad_scale=gs)
        else:
            rt.adamw_multi_int8(ents, self.b1, 1.0 - self.b1, self.b2,
                                1.0 - self.b2, self.lr, self.lr * self.wd,
                                self.eps, bc1, bc2, self.block, grad_scale=gs)
        # 参数变了 ⇒ fp16 权重影子作废（同 Fp16MomAdamW 的处理）
        try:
            rt.f16_cache_clear()
        except Exception:
            pass
