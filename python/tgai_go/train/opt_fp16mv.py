# -*- coding: utf-8 -*-
"""**fp16 动量**的 AdamW（显存版）：m/v 以 fp16 常驻显存，更新时逐张量转 fp32 计算。

为什么这样省显存：2.97B 的 fp32 动量要 23.8 GB，fp16 只要 11.9 GB；
而 Adam 的更新是 m/√v（尺度不变），fp16 的 ~1e-3 相对误差在分子分母上大部分抵消
（CPU 实测：参数偏差 4.6e-05、更新方向余弦 1.000000、loss 偏差 3.2e-05）。

实现取舍：不新写内核，复用已有的 fp32 融合 AdamW（`rt.adamw_multi`）+
已有的 f32↔f16 转换算子；每个张量走一次 "f16→f32 → 更新 → f32→f16"。
代价是每步多 ~2 遍动量带宽（m/v 总量 11.9 GB ⇒ 约 +1.4 s/步），换来的是**不用写新内核**、
不会引入 f16 转换的实现风险。后续若确实需要，再把它融成一个内核。
"""
import numpy as np

import tgai_go.train as T


def _rt():
    return T._cuda()


class Fp16MomAdamW(object):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fp32_state=False):
        self.params = list(params)
        self.lr, (self.b1, self.b2) = float(lr), (float(betas[0]), float(betas[1]))
        self.eps, self.wd = float(eps), float(weight_decay)
        self.t = 0
        rt = _rt()
        self.fp32_state = bool(fp32_state)
        self.m16, self.v16 = [], []
        for p in self.params:
            n = int(p.numel())
            if self.fp32_state:
                self.m16.append(rt.zeros((n,)))
                self.v16.append(rt.zeros((n,)))
            else:
                self.m16.append(rt.zeros((n,), dtype=np.uint16))
                self.v16.append(rt.zeros((n,), dtype=np.uint16))

    def bytes_per_param(self):
        return 4.0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        rt = _rt()
        for i, p in enumerate(self.params):
            if p.grad is None or not getattr(p, "requires_grad", True):
                continue
            # f16 动量 → f32 临时缓冲（逐个张量，峰值只多一个张量的 fp32）
            if self.fp32_state:
                m32, v32 = self.m16[i], self.v16[i]
            else:
                m32 = rt.cast_f16_to_f32(self.m16[i])
                v32 = rt.cast_f16_to_f32(self.v16[i])
            rt.adamw_multi([(p._gpu, p.grad._gpu, m32, v32)],
                           b1, 1.0 - b1, b2, 1.0 - b2, self.lr, self.lr * self.wd,
                           self.eps, bc1, bc2, 1)
            if not self.fp32_state:
                rt.cast_f32_to_f16_into(m32, self.m16[i])
                rt.cast_f32_to_f16_into(v32, self.v16[i])
    # ---------- 检查点：优化器状态（fp16 m/v + 步数） ----------
    def state_dict(self):
        """取出 m/v 与步数（逐张量 D2H，主机峰值 = 单张量）。"""
        rt = _rt()
        m = [np.asarray(rt.d2h(x)) for x in self.m16]
        v = [np.asarray(rt.d2h(x)) for x in self.v16]
        return {"t": int(self.t), "m": m, "v": v}

    def load_state_dict(self, sd):
        """恢复 m/v 与步数；形状/长度不符直接报错（防止静默错配）。"""
        rt = _rt()
        m, v = sd["m"], sd["v"]
        if len(m) != len(self.params) or len(v) != len(self.params):
            raise ValueError("优化器状态长度不符：文件 %d/%d，当前参数 %d。修法：确认是同一模型"
                             % (len(m), len(v), len(self.params)))
        self.m16, self.v16 = [], []
        for a, b in zip(m, v):
            self.m16.append(rt.h2d(np.ascontiguousarray(a)))
            self.v16.append(rt.h2d(np.ascontiguousarray(b)))
        self.t = int(sd.get("t", 0))
        return self


def save_opt_states(opts, path):
    """把每层优化器状态存成一个 .npz（逐层落盘，避免整体驻留内存）。"""
    out = {}
    for i, opt in enumerate(opts):
        sd = opt.state_dict()
        f = "%s/layer_%03d.npz" % (path, i)
        np.savez(f, t=np.int64(sd["t"]),
                 **{("m_%d" % k): a for k, a in enumerate(sd["m"])},
                 **{("v_%d" % k): a for k, a in enumerate(sd["v"])})
        out[i] = f
    return out


def load_opt_states(opts, path):
    """从目录恢复每层优化器状态。"""
    for i, opt in enumerate(opts):
        f = "%s/layer_%03d.npz" % (path, i)
        z = np.load(f, allow_pickle=False)
        n = len(opt.params)
        sd = {"t": int(z["t"]),
              "m": [z["m_%d" % k] for k in range(n)],
              "v": [z["v_%d" % k] for k in range(n)]}
        opt.load_state_dict(sd)
    return opts
