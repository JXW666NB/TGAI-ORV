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
import sys

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
        """每参数字节数。**注意 m 与 v 是两个缓冲**：
        fp16 动量 = 2 × 2 = 4 字节；fp32 动量 = 2 × 4 = 8 字节。
        （曾经写成 2.0 / 4.0，把 3B 的 11.1 GiB 算成了 5.5 GiB。）
        """
        return 8.0 if self.fp32_state else 4.0

    def state_bytes(self):
        """优化器状态总占用（字节）。3B 上用来判断"装不装得下"。"""
        n = sum(int(p.numel()) for p in self.params)
        return float(n) * self.bytes_per_param()

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        """一步更新。

        **融合路径**（默认，fp16 动量）：把所有有梯度的张量收进一次 `adamw_multi_f16`
        发射 —— 之前是每张量 5 次发射（cast×4 + adamw×1），V2.7 有 2550 个张量，
        实测占整步 58%（约 0.98 s）。
        算术序列与逐张量路径逐条相同，结果**逐位一致**（测试里对照）。

        `_step_scalar()` 保留为逐张量实现，用于对齐测试与 fp32_state 分支。
        """
        self.t += 1
        if self.fp32_state:
            return self._step_scalar()
        rt = _rt()
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        ents = []
        for i, p in enumerate(self.params):
            if p.grad is None or not getattr(p, "requires_grad", True):
                continue
            ents.append((p._gpu, p.grad._gpu, self.m16[i], self.v16[i]))
        rt.adamw_multi_f16(ents, b1, 1.0 - b1, b2, 1.0 - b2, self.lr, self.lr * self.wd,
                           self.eps, bc1, bc2, 1)
        # 参数已更新 ⇒ 旧的 fp16 权重影子作废（AMP 缓存，见 cuda.py f16_cache_*）
        try:
            rt.f16_cache_clear()
        except Exception:
            pass

    def _step_scalar(self):
        """逐张量更新（原实现，保留用于对照测试与 fp32_state）。"""
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
        try:
            rt.f16_cache_clear()
        except Exception:
            pass
    # ---------- 分层即时更新（0.9.28）：省掉"全部梯度同时驻留" ----------
    def set_layer_groups(self, groups, require_full=True, auto_rest=True):
        """登记"哪些参数属于哪一层"，供 `step_layers()` 使用。

        `groups` 形如 `{0: [param, ...], 1: [...], -1: [embed, norm, head]}`。

        `auto_rest=True`（默认）：没被任何组包含的参数会**自动**收进 `-1` 组，
        并打印一次提示。这样调用方只需写"每层的参数"即可，不会漏掉 embed/norm/head。

        `require_full=True` 时若 `auto_rest` 关掉且仍有未覆盖参数 → 直接报错。
        为什么必须拦住：实测踩过 —— 模型 24 个参数张量，`blocks[i]` 只覆盖 21 个，
        剩下 3 个没人更新，导致"分层更新 vs 统一更新"差 1.1e-02。
        这类错误不报错、只**悄悄少训练一部分参数**。
        """
        self._groups = {}
        # 绑定权重（tie_embedding 时 embed.weight 与 lm_head.weight 是**同一个对象**）
        # 会在 self.params 里出现两次。若按"每个出现的下标"更新，同一参数会被更新两遍
        # ⇒ 与统一更新不一致（实测差异 1.285e-03）。这里按对象身份去重，只取第一个下标。
        first = {}
        for i, q in enumerate(self.params):
            first.setdefault(id(q), i)
        for k, params in dict(groups or {}).items():
            idx = []
            for p in params:
                j = first.get(id(p))
                if j is not None and j not in idx:
                    idx.append(j)
            self._groups[int(k)] = idx
        covered = set()
        for idx in self._groups.values():
            covered.update(idx)
        # 只统计"唯一对象"里没被覆盖的
        unique_idx = sorted(set(first.values()))
        missing = [i for i in unique_idx if i not in covered]
        if missing and auto_rest:
            self._groups.setdefault(-1, [])
            self._groups[-1] = sorted(set(self._groups[-1]) | set(missing))
            sys.stderr.write(
                "[tgai] 分组未覆盖 %d 个参数（通常是 embed / final_norm / lm_head），"
                "已自动放进 -1 组，请记得 step_layers([... , -1])。\n" % len(missing))
            sys.stderr.flush()
            missing = []
        if require_full and missing:
            raise ValueError(
                "分层分组没有覆盖全部参数：%d / %d 个没被任何组包含。"
                "修法：打开 auto_rest，或把 embed / final_norm / lm_head 单独放进一个组，"
                "否则这些参数**永远不会被更新**（不报错，只静默少训练）"
                % (len(missing), len(self.params)))
        return self

    def all_groups(self):
        """已登记的全部组 key（调用 `step_layers` 时应当全部走一遍）。"""
        return sorted(self._groups.keys())

    def step_layers(self, layer_ids, t=None, release=True):
        """**只更新这些层**，并（默认）立刻释放它们的梯度。

        为什么在数学上等价于标准 AdamW：
        反向是逐层进行的，层 i 的 `dL/dx_i` 算完之后，**层 i-1 的反向不再需要层 i 的权重**。
        所以此时用同一个 `t` 更新层 i，与"全部反向后统一更新"得到的参数完全一致。
        这样同时驻留的梯度只需要一层的量（3B 模型上约 0.4 GiB，而不是 11.1 GiB）。

        `t` 必须由调用方在同一次外步里传同一个值（默认用 `self.t + 1`），
        否则各层会用不同的偏差校正系数。
        """
        if not getattr(self, "_groups", None):
            raise RuntimeError("还没登记分层参数" + "。修法：先调用 set_layer_groups({i: tr.params_of(i)})")
        tt = int(self.t + 1) if t is None else int(t)
        rt = _rt()
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1 ** tt
        bc2 = 1.0 - b2 ** tt
        ents = []
        for lid in layer_ids:
            for i in self._groups.get(int(lid), []):
                p = self.params[i]
                if p.grad is None or not getattr(p, "requires_grad", True):
                    continue
                ents.append((p._gpu, p.grad._gpu, self.m16[i], self.v16[i]))
        if ents:
            if self.fp32_state:
                rt.adamw_multi(ents, b1, 1.0 - b1, b2, 1.0 - b2, self.lr, self.lr * self.wd,
                               self.eps, bc1, bc2, 1)
            else:
                rt.adamw_multi_f16(ents, b1, 1.0 - b1, b2, 1.0 - b2, self.lr, self.lr * self.wd,
                                   self.eps, bc1, bc2, 1)
            try:
                rt.f16_cache_clear()
            except Exception:
                pass
        if release:
            for lid in layer_ids:
                for i in self._groups.get(int(lid), []):
                    self.params[i].grad = None
        if t is None:
            self.t = tt
        return len(ents)

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
