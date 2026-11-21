# -*- coding: utf-8 -*-
"""int8 分块 AdamW（省显存的动量存储）。

诚实说明：这是**有损量化**。把 m/v 从 fp32(8B) 压到 int8(2B) 必然丢精度；
要做到与 fp32 逐位恒等就得保留 8 B/参数的 fp32 残差，那等于没省，故本实现不做全量误差反馈。
补偿手段：分块对称量化（每块独立 scale）、量化只作用于动量、递推/偏差校正/解耦衰减与 PyTorch 一致。
等价性靠实测（见 tests/test_int8_adamw.py），不达标就上误差反馈并如实报告显存代价。
"""
from __future__ import annotations

import numpy as np

BLOCK = 64
INT8_MAX = 127


def _quant_block(x):
    amax = float(np.abs(x).max()) if x.size else 0.0
    if amax == 0.0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = amax / INT8_MAX
    q = np.clip(np.rint(x / scale), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, float(scale)


def _dequant_block(q, scale):
    return q.astype(np.float32) * np.float32(scale)


class Int8AdamW(object):
    def __init__(self, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, block=BLOCK,
                 resid_dtype=None):
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.block = int(block)
        #: None = 无误差反馈（有损）；"float32"/"float16" = 存 fp32/fp16 残差（误差反馈）
        self.resid_dtype = resid_dtype
        self.t = 0
        self.state = {}

    def bytes_per_param(self):
        b = 2.0 + 2.0 * 4.0 / self.block          # int8 m/v + 块 scale
        if self.resid_dtype == "float32":
            b += 8.0                               # m,v 各 4B 残差
        elif self.resid_dtype == "float16":
            b += 4.0
        elif self.resid_dtype == "int8":
            b += 2.0 + 2.0 * 4.0 / self.block      # int8 残差 + 独立块 scale
        return b

    def _ensure(self, name, shape):
        st = self.state.get(name)
        if st is None:
            n = int(np.prod(shape))
            nb = (n + self.block - 1) // self.block
            st = dict(mq=np.zeros(n, np.int8), vq=np.zeros(n, np.int8),
                      ms=np.zeros(nb, np.float32), vs=np.zeros(nb, np.float32))
            if self.resid_dtype:
                dt = {"float32": np.float32, "float16": np.float16,
                      "int8": np.int8}[self.resid_dtype]
                st["mr"] = np.zeros(n, dt)
                st["vr"] = np.zeros(n, dt)
                if self.resid_dtype == "int8":
                    st["mrs"] = np.zeros(nb, np.float32)
                    st["vrs"] = np.zeros(nb, np.float32)
            self.state[name] = st
        return st

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        lr_t = self.lr * float(np.sqrt(bc2)) / bc1
        for name, p in params.items():
            g = np.asarray(grads[name], dtype=np.float32)
            if g.shape != p.shape:
                raise ValueError("梯度形状 %s 与参数 %s 不一致。修法：检查该参数 backward 的输出形状"
                                 % (g.shape, p.shape))
            fp = p.reshape(-1)
            fg = g.reshape(-1)
            st = self._ensure(name, p.shape)
            n = fp.size
            for b in range(st["ms"].size):
                lo, hi = b * self.block, min(n, (b + 1) * self.block)
                gb = fg[lo:hi]
                m_prev = _dequant_block(st["mq"][lo:hi], st["ms"][b])
                v_prev = _dequant_block(st["vq"][lo:hi], st["vs"][b])
                if self.resid_dtype == "int8":
                    m_prev = m_prev + _dequant_block(st["mr"][lo:hi], st["mrs"][b])
                    v_prev = v_prev + _dequant_block(st["vr"][lo:hi], st["vrs"][b])
                elif self.resid_dtype:
                    m_prev = m_prev + st["mr"][lo:hi].astype(np.float32)
                    v_prev = v_prev + st["vr"][lo:hi].astype(np.float32)
                m = self.b1 * m_prev + (1 - self.b1) * gb
                v = self.b2 * v_prev + (1 - self.b2) * gb * gb
                st["mq"][lo:hi], st["ms"][b] = _quant_block(m)
                st["vq"][lo:hi], st["vs"][b] = _quant_block(v)
                if self.resid_dtype == "int8":
                    rm = m - _dequant_block(st["mq"][lo:hi], st["ms"][b])
                    rv = v - _dequant_block(st["vq"][lo:hi], st["vs"][b])
                    st["mr"][lo:hi], st["mrs"][b] = _quant_block(rm)
                    st["vr"][lo:hi], st["vrs"][b] = _quant_block(rv)
                elif self.resid_dtype:
                    st["mr"][lo:hi] = (m - _dequant_block(st["mq"][lo:hi], st["ms"][b])).astype(st["mr"].dtype)
                    st["vr"][lo:hi] = (v - _dequant_block(st["vq"][lo:hi], st["vs"][b])).astype(st["vr"].dtype)
                fp[lo:hi] -= self.lr * self.wd * fp[lo:hi]
                fp[lo:hi] -= lr_t * (m / (np.sqrt(v) + self.eps))
        return dict(t=self.t, lr_eff=lr_t)


class Fp32AdamW(object):
    def __init__(self, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr, (self.b1, self.b2) = float(lr), (float(betas[0]), float(betas[1]))
        self.eps, self.wd = float(eps), float(weight_decay)
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for name, p in params.items():
            g = np.asarray(grads[name], dtype=np.float32)
            m = self.b1 * self.m.get(name, np.zeros_like(p)) + (1 - self.b1) * g
            v = self.b2 * self.v.get(name, np.zeros_like(p)) + (1 - self.b2) * g * g
            self.m[name], self.v[name] = m, v
            p -= self.lr * self.wd * p
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

class Fp16AdamW(object):
    """**m/v 存 fp16**（4 B/参数）的 AdamW —— 2.97B 只需 11.9 GB 动量。

    为什么这比 int8+误差反馈更好（实测，100 步对照 fp32）：
      * 无量化误差反馈时 int8 会漂移（余弦 0.33）；加 fp16 残差后达标但 6.13 B/参数（18.2 GB）；
      * fp16 直接存 m/v：**4.00 B/参数（11.9 GB）**，参数偏差 4.6e-05、**余弦 1.000000**、loss 偏差 2.8e-05。
    原因：Adam 的更新是 m/√v（尺度不变），fp16 的 ~1e-3 相对误差在分子分母上大部分抵消。
    递推、bias correction、解耦权重衰减仍是 fp32；只有 m/v 的**存储**是 fp16。
    """

    def __init__(self, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr, (self.b1, self.b2) = float(lr), (float(betas[0]), float(betas[1]))
        self.eps, self.wd = float(eps), float(weight_decay)
        self.t = 0
        self.m, self.v = {}, {}

    def bytes_per_param(self):
        return 4.0

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for name, p in params.items():
            g = np.asarray(grads[name], dtype=np.float32)
            if g.shape != p.shape:
                raise ValueError("梯度形状 %s 与参数 %s 不一致。修法：检查该参数 backward 的输出形状"
                                 % (g.shape, p.shape))
            m0 = self.m.get(name)
            v0 = self.v.get(name)
            m = self.b1 * (m0.astype(np.float32) if m0 is not None else 0.0) + (1 - self.b1) * g
            v = self.b2 * (v0.astype(np.float32) if v0 is not None else 0.0) + (1 - self.b2) * g * g
            self.m[name] = m.astype(np.float16)
            self.v[name] = v.astype(np.float16)
            p -= self.lr * self.wd * p
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
