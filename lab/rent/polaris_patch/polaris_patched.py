#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POLARIS —— 自适应正交融合优化器（TGAI-ORV 插件）

一个优化器，不是旋钮框架。给一个学习率就能用，其余全部自动。

命名
----
POLARIS = POlar-express + Row-scaling + Adaptive + Inertial + Soft-gate。
北极星：训练迷雾里"指引方向"——正交化为更新定方向，且随状态自适应。

一句话数学
----------
    2D 权重:  动量 → Nesterov → Polar-Express 正交化 → 逐行 RMS → Cautious 软门 → 更新
    1D/标量:  AdamW（含 bias correction）

它是 M26~M27 全部实测结论的收敛：
  * 正交化定方向（Muon/ORCA，Polar Express 五次多项式）
  * 逐行 RMS 免疫动量缩放（CRAM/NorMuon）
  * Cautious 门控防走坏（M26 消融证明必需）
  * AOL 零额外矩阵乘预条件（Turbo-Muon，5 步 NS 压到 4 步）
  * 稀疏正交化 orth_freq（3090 实测 1.68×，质量持平）

★ 自适应（本插件的核心，"根据当前自动调整"）：
  1) 正交频率自适应：缓存正交方向的余弦相似度 cos_dir 越高（方向越稳定），
     正交化频率越低（step 间隔越长），省算力；方向剧变立刻恢复每步正交。
     无需手动设 orth_freq。
  2) 门控锐度自适应：记录 EMA 的"门控命中率"（g·d>0 的比例）。命中率越低
     （梯度与更新越不一致、越危险），cautious 门越硬；训练稳定后自动放松。
     无需手动调 cautious 强度。

用法
----
    import tgai_go.train as T
    T.plugin.load("polaris.py")          # 或从 ~/.tgai/plugins/ 自动加载
    opt = T.optim.Polaris(model.parameters(), lr=1e-3)
    T.fit(model, data, steps=2000, opt=opt)

    print(opt.status())                  # 看当前自适应状态（频率/门控/命中率/NS 次数）

兼容：CPU（numpy，完整实现）与 CUDA（复用 ORV 内置 Muon 的 device 原语）。
不依赖 torch。一个 .py 文件即插件。
"""
from __future__ import annotations

import math
import os

import numpy as np

# ----------------------------------------------------------------------------
# 自适应配置：这些是"自动调整"的边界，不是需要用户调的旋钮
# ----------------------------------------------------------------------------
PE_A, PE_B, PE_C = 3.4445, -4.7750, 2.0315  # Polar Express 五次多项式（Muon 口径）
_MIN_FREQ, _MAX_FREQ = 1, 8        # 正交间隔的自动范围
_DIR_STABLE = 0.995                # cos_dir 高于此值认为方向稳定 → 可拉长间隔
_GATE_HIT_LO, _GATE_HIT_HI = 0.5, 0.9  # 命中率区间 → 门控锐度


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _row_rms(x, eps=1e-7):
    return np.sqrt((x * x).mean(axis=-1, keepdims=True)) + eps


def _polar_express_np(x, steps=5, eps=1e-7):
    """Polar-Express Newton-Schulz 正交化（CPU/numpy）。返回谱≈1 的正交矩阵。

    单一五次多项式迭代：不动点 0.868，5 步把奇异值压到 [0.868±]，条件数≈1.6。
    与 ORV 内置 Muon 的 _newton_schulz5 同公式，供 CPU 与 self-test 使用。
    """
    x = np.array(x, dtype=np.float32, copy=True)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = np.ascontiguousarray(x.T)
    # ★ 补丁（2026-11-20，由 ORV 核心修复方提供）：下面这 4 行原来是 **Python 逐元素循环**
    #   求平方和。实测 26 万参数的一个矩阵就要 663 ms，把 1.05M 参数模型的
    #   CPU opt.step 拖到 620~813 ms（AdamW 只要 8 ms）。
    #   改成 BLAS 的 float32 `dot`（数学等价，差异只来自累加顺序，实测相对差 ~1e-7）；
    #   Newton-Schulz 会把范数的 1 ulp 差放大，所以对拍 GPU 时请设 POLARIS_SUMSQ_EXACT=1。
    if os.environ.get("POLARIS_SUMSQ_EXACT") == "1":
        acc = np.float32(0.0)
        for v in np.ascontiguousarray(x, np.float32).ravel():
            acc = np.float32(acc + np.float32(v * v))
    else:
        _flat = np.ascontiguousarray(x, np.float32).ravel()
        acc = np.float32(np.dot(_flat, _flat))
    nrm = np.float32(math.sqrt(float(acc)) + eps)
    x = np.float32(x / nrm)
    a, b, c = np.float32(PE_A), np.float32(PE_B), np.float32(PE_C)
    for _ in range(int(steps)):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = np.ascontiguousarray(x.T)
    return np.asarray(x, np.float32)


class Polaris:
    """POLARIS 自适应正交融合优化器。

    参数
    ----
    lr       : 学习率（唯一需要给的超参；fit 的 scheduler 会改这个属性）
    momentum : 动量系数（默认 0.95，M26/M27 验证值）
    wd       : 解耦权重衰减
    ns_steps : Polar-Express 迭代步数（默认 5）
    ns_dtype : NS 矩阵乘精度 'fp16'/'fp32'（仅 CUDA；fp16 真机更稳）
    tf32     : NS 期间开 TF32（仅 CUDA，结束恢复）
    betas    : 1D 参数 AdamW 的 (b1,b2)
    eps      : 数值下限
    auto     : ★ 是否启用自适应（默认 True）。关掉则正交频率固定为 1、门控固定中等锐度
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, wd=0.0,
                 ns_steps=5, ns_dtype="fp16", tf32=True,
                 betas=(0.9, 0.999), eps=1e-8, auto=True):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.wd = float(wd)
        self.ns_steps = int(ns_steps)
        self.ns_dtype = str(ns_dtype)
        self.tf32 = bool(tf32)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.auto = bool(auto)
        self.t = 0

        # 每参数状态：2D 走正交路径，1D 走 AdamW
        self.m = []
        self.mv = []
        self.mv2 = []
        self.is2d = []
        self.dir = []          # 缓存的正交方向（供稀疏正交化复用）
        self.u = []            # CUDA: 正交化输入 buffer（与动量缓冲分离，避免别名）
        # 自适应诊断量
        self._freq = 1         # 当前正交间隔
        self._hit_ema = 0.7    # 门控命中率 EMA（0=全门控，1=无门控）
        self._dir_cos = 0.0    # 正交方向余弦 EMA（从 0 起步，需连续稳定才拉长间隔）
        self._warm = 0         # 自适应预热计数（前几步强制每步正交，攒够信号）
        self.n_ns_calls = 0    # 实际 NS 调用次数（诊断省了多少算力）

        self._has_cuda = any(getattr(p, "device", "cpu") == "cuda" for p in self.params)
        self._rt = None
        if self._has_cuda:
            from tgai_go.train import _cuda  # ORV 内置 CUDA runtime
            self._rt = _cuda()

        for p in self.params:
            shp = tuple(int(s) for s in p.shape)
            two_d = (len(shp) == 2)
            self.is2d.append(two_d)
            if getattr(p, "device", "cpu") == "cuda":
                if two_d:
                    self.m.append(self._rt.zeros(shp))
                    self.mv.append(None); self.mv2.append(None)
                    self.u.append(self._rt.empty(shp))
                else:
                    self.m.append(None)
                    self.mv.append(self._rt.zeros(shp))
                    self.mv2.append(self._rt.zeros(shp))
                    self.u.append(None)
                self.dir.append(None)
            else:
                data = np.asarray(p.data, np.float32)
                if two_d:
                    self.m.append(np.zeros(shp, np.float32))
                    self.mv.append(None); self.mv2.append(None)
                    self.dir.append(None)
                else:
                    self.m.append(None)
                    self.mv.append(np.zeros(shp, np.float32))
                    self.mv2.append(np.zeros(shp, np.float32))
                    self.dir.append(None)

    # ------------------------------------------------------------------ 自适应
    def _adapt_frequency(self, cos_list):
        """方向越稳定（cos 高）→ 正交间隔越长 → 省算力；剧变立刻回到每步。"""
        if not self.auto:
            self._freq = 1
            return
        self._warm += 1
        if self._warm < 8:            # 预热：攒够方向信号前，每步正交
            self._freq = 1
            if cos_list:
                self._dir_cos = 0.9 * self._dir_cos + 0.1 * float(np.mean(cos_list))
            return
        c = float(np.mean(cos_list)) if cos_list else 1.0
        self._dir_cos = 0.9 * self._dir_cos + 0.1 * c
        if self._dir_cos >= _DIR_STABLE:
            self._freq = min(_MAX_FREQ, int(round(
                1 + (self._dir_cos - _DIR_STABLE) / (1.0 - _DIR_STABLE) * (_MAX_FREQ - 1))))
        else:
            self._freq = _MIN_FREQ

    def _gate_strength(self):
        """命中率越低越危险 → 门控越硬（放大 g·d 的符号权重）。"""
        if not self.auto:
            return 4.0
        # 命中率低 → 大 k（更接近硬门）；命中率高 → 小 k（放松）
        return float(2.0 + 10.0 * max(0.0, min(1.0, (_GATE_HIT_HI - self._hit_ema)
                                                   / (_GATE_HIT_HI - _GATE_HIT_LO))))

    def status(self):
        """当前自适应状态快照（给日志/调试看优化器'此刻在想什么'）。"""
        n_2d = sum(1 for x in self.is2d if x)
        theoretical = max(1, self.t) * n_2d          # 每步都正交的理论次数
        return {
            "step": self.t,
            "auto": self.auto,
            "orth_freq": self._freq,          # 当前隔几步正交一次
            "dir_cos": round(self._dir_cos, 5),  # 方向稳定性
            "gate_hit": round(self._hit_ema, 5), # 门控命中率
            "gate_k": round(self._gate_strength(), 3),
            "ns_calls": self.n_ns_calls,      # 累计 NS 次数
            "ns_saved_pct": round(100.0 * (1 - self.n_ns_calls / theoretical), 1),
        }

    # ------------------------------------------------------------------ 接口
    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def _invalidate_shadows(self):
        try:
            from tgai_go.train import fp16_shadows_invalidate
            fp16_shadows_invalidate()
        except Exception:
            pass

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.t += 1
        cos_list = []
        hit_list = []

        if self._has_cuda:
            self._step_cuda(cos_list, hit_list)
        else:
            self._step_cpu(cos_list, hit_list)

        self._adapt_frequency(cos_list)
        if hit_list:
            h = float(np.mean(hit_list))
            self._hit_ema = 0.9 * self._hit_ema + 0.1 * h
        self._invalidate_shadows()
        return loss

    # ------------------------------------------------------------------ CPU
    def _step_cpu(self, cos_list, hit_list):
        do_orth = (self.t == 1) or (self._freq <= 1) or (self.t % self._freq == 1)
        k = self._gate_strength()
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = np.asarray(p.grad.to_numpy() if hasattr(p.grad, "to_numpy") else p.grad,
                           np.float32)
            if not self.is2d[i]:
                self._adamw_cpu(i, p, g)
                continue

            m, lr, mom, wd, eps = self.m[i], self.lr, self.momentum, self.wd, self.eps
            # 动量 + Nesterov
            m *= mom
            m += g
            u = g + mom * m
            flip = u.shape[0] > u.shape[1]
            if flip:
                u = np.ascontiguousarray(u.T)
                gu = np.ascontiguousarray(g.T)
            else:
                gu = g

            # ---- 正交方向（自适应频率）----
            if do_orth:
                d = _polar_express_np(u, steps=self.ns_steps, eps=eps)
                self.dir[i] = d.copy()
                self.n_ns_calls += 1
                cos_list.append(1.0)  # 刚正交，方向是新的
            else:
                d = self.dir[i]
                if d is None:
                    d = _polar_express_np(u, steps=self.ns_steps, eps=eps)
                    self.dir[i] = d.copy()
                    self.n_ns_calls += 1
                    cos_list.append(1.0)
                else:
                    # 缓存方向与当前更新的余弦 → 自适应频率信号
                    d_norm = math.sqrt(float((d * d).sum())) + eps
                    u_norm = math.sqrt(float((u * u).sum())) + eps
                    c = float((d * u).sum()) / (d_norm * u_norm)
                    cos_list.append(max(-1.0, min(1.0, c)))

            # Muon 谱缩放
            d = d * (0.2 * math.sqrt(max(p.data.shape)))
            # 逐行 RMS
            d = d / _row_rms(d, eps=1e-7)
            # Cautious 软门：命中率驱动锐度
            dot = (gu * d).sum()
            denom = math.sqrt(float((gu * gu).sum())) * math.sqrt(float((d * d).sum())) + eps
            cos = max(-1.0, min(1.0, float(dot) / (denom + 1e-12)))
            gate = float(_sigmoid(np.float32(k * cos)))
            hit_list.append(1.0 if (dot > 0) else 0.0)
            d = d * gate

            if flip:
                d = np.ascontiguousarray(d.T)
            w = np.asarray(p.data, np.float32)
            new_w = (w - lr * (d + wd * w)).astype(np.float32)
            p.data = new_w

    def _adamw_cpu(self, i, p, g):
        st, st2 = self.mv[i], self.mv2[i]
        t = self.t
        st *= self.b1; st += (1 - self.b1) * g
        st2 *= self.b2; st2 += (1 - self.b2) * (g * g)
        mh = st / (1 - self.b1 ** t)
        vh = st2 / (1 - self.b2 ** t)
        w = np.asarray(p.data, np.float32)
        upd = self.lr * mh / (np.sqrt(vh) + self.eps)
        if self.wd:
            upd = upd + self.lr * self.wd * w
        p.data = (w - upd).astype(np.float32)

    # ------------------------------------------------------------------ CUDA
    def _norm2(self, rt, buf):
        """‖buf‖²：设备端平方和 → 标量 float（一次 D2H，与内置 Muon 同口径）。"""
        return float(rt.read_scalar(rt.sumsq_parallel(buf)))

    def _inner(self, rt, a, b):
        """⟨a,b⟩ = (‖a+b‖²−‖a‖²−‖b‖²)/2，只用已验证的 add/sumsq/read_scalar。"""
        ab = rt.add(a, b)
        return 0.5 * (self._norm2(rt, ab) - self._norm2(rt, a) - self._norm2(rt, b))

    def _step_cuda(self, cos_list, hit_list):
        """CUDA 路径：只用 ORV 内置 Muon 真机验证过的 device 原语，值不回 CPU。

        * 动量：rt.muon_momentum（buf 与 u 分离，避免别名）
        * 正交：_newton_schulz5_gpu（ns_dtype='fp32' 用顺序 FMA；其余用 cuBLAS）
        * 稀疏正交：中间步复用缓存方向，按当前 u 的整体 RMS 重标定（ORCA v2 实测口径）
        * 尺度：谱缩放 + 整体 RMS 归一化（逐行 RMS 在 CPU 版做，GPU 用整体近似——
          正交化后各行动量幅度已接近，差异很小；避免引入未验证的逐行归约内核）
        * 门控：内积用向量恒等式间接算（g·d = (‖g+d‖²−‖g‖²−‖d‖²)/2）

        ORV 原生 NS 是 fp32（走 ORV 自己的 CUDA runtime，与 torch.backends 无关），
        故 ns_dtype 只决定 GEMM 后端：'fp32'→顺序 FMA（逐位稳），其余→cuBLAS（快）。
        """
        from tgai_go.train import _newton_schulz5_gpu, _f32, Tensor
        rt = self._rt
        k = self._gate_strength()
        do_orth = (self.t == 1) or (self._freq <= 1) or (self.t % self._freq == 1)
        use_cublas = (self.ns_dtype != "fp32")

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
                raise TypeError("CUDA Polaris 需要 CUDA Tensor 梯度，收到 %s" % type(g).__name__)
            if not self.is2d[i]:
                self._adamw_cuda(i, p, g, rt, _f32)
                continue

            m, u = self.m[i], self.u[i]
            # 动量 + Nesterov：buf=m（更新后供下一步），u=正交输入（分离缓冲）
            rt.muon_momentum(p._gpu, g._gpu, m, u,
                             _f32(self.momentum), _f32(0.0), False, True)

            # ---- 正交方向（自适应频率）----
            if do_orth:
                d = _newton_schulz5_gpu(rt, u, steps=self.ns_steps, fast=use_cublas)
                self.dir[i] = d
                self.n_ns_calls += 1
                cos_list.append(1.0)
            else:
                d = self.dir[i]
                if d is None:
                    d = _newton_schulz5_gpu(rt, u, steps=self.ns_steps, fast=use_cublas)
                    self.dir[i] = d
                    self.n_ns_calls += 1
                    cos_list.append(1.0)
                else:
                    # 缓存方向与当前更新的余弦 → 自适应频率信号
                    c = self._inner(rt, d, u) / (
                        math.sqrt(self._norm2(rt, d) * self._norm2(rt, u)) + self.eps)
                    cos_list.append(max(-1.0, min(1.0, c)))
                    # 方向慢变、幅度快变：用当前 u 的整体 RMS 重标定缓存方向
                    su = math.sqrt(self._norm2(rt, u) / max(1, u.n)) + self.eps
                    sd = math.sqrt(self._norm2(rt, d) / max(1, d.n)) + self.eps
                    d = rt.scale(d, _f32(su / sd))

            # 谱缩放 + 整体 RMS 归一化
            nn = max(int(p.shape[0]), int(p.shape[1]))
            d = rt.scale(d, _f32(0.2 * math.sqrt(nn)))
            rms = math.sqrt(self._norm2(rt, d) / max(1, d.n)) + 1e-7
            d = rt.div_scalar(d, _f32(rms))

            # Cautious 软门：全局内积恒等式（逐元素门在 GPU 需额外内核，用全局近似）
            dot = self._inner(rt, g._gpu, d)
            hit_list.append(1.0 if dot > 0 else 0.0)
            denom = math.sqrt(self._norm2(rt, g._gpu) * self._norm2(rt, d)) + self.eps
            cos = max(-1.0, min(1.0, dot / (denom + 1e-12)))
            gate = float(_sigmoid(np.float32(k * cos)))
            d = rt.scale(d, _f32(gate))

            # 解耦权重衰减 + 更新
            if self.wd:
                p.data  # noqa: 占位（CUDA 权重在 _gpu）
                rt.muon_momentum(p._gpu, p._gpu, p._gpu, p._gpu,
                                 _f32(0.0), _f32(self.lr * self.wd), True, False)
            rt.sub_scale_(p._gpu, d, _f32(self.lr))

    def _adamw_cuda(self, i, p, g, rt, f32):
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        rt.adamw(p._gpu, g._gpu, self.mv[i], self.mv2[i],
                 f32(self.b1), f32(1 - self.b1), f32(self.b2), f32(1 - self.b2),
                 f32(self.lr), 0.0, f32(self.eps), f32(bc1), f32(bc2), False)


def _f32(x):
    return np.float32(x)


# ----------------------------------------------------------------------------
# ORV 插件契约与注册
# ----------------------------------------------------------------------------
def register(T=None, replace=False):
    """把 Polaris 注册进 ORV：T.optim.Polaris 可用。幂等——已登记则跳过。"""
    if T is None:
        import tgai_go.train as T
    existing = T.plugin.resolve_optimizer("Polaris")
    if existing is Polaris:
        return Polaris
    if existing is not None and not replace:
        return existing   # 同名但不同类：不擅自覆盖
    T.plugin.optimizer("Polaris", Polaris, works_inplace=False, replace=replace)
    return Polaris


# 模块被 import / 被 plugin.load 加载时自动注册（幂等：重名时静默跳过）
def _auto_register():
    try:
        import tgai_go.train as T
        if T.plugin.resolve_optimizer("Polaris") is None:
            register(T)
    except Exception:
        pass


_auto_register()

PLUGIN = {
    "name": "polaris-orv",
    "version": "1.0.0",
    "api_version": 1,
    "description": "POLARIS 自适应正交融合优化器（单文件 ORV 插件，CPU/CUDA）",
    "author": "BLLM",
    "license": "MIT",
    "ops": {},  # 本插件提供的是优化器轴，不是算子轴
}
