# -*- coding: utf-8 -*-
"""bf16 激活存储的数值可行性实验（按 docs/BF16_ACT_STORAGE_PLAN.md 的协议）。

两件事：
1) 验证往返误差量级 = 2^-9（bf16 只有 8 位尾数），确认量化通道正确
2) A/B 20 步：把 RMSNorm 的输出经 bf16 往返（直通梯度），对比 loss 曲线

判据（协议）：末步 loss 相对偏差 <= 1%，无 NaN，gnorm 不单调发散。
不达标就撤掉这条路，不靠调 lr 掩盖。

注意：这里**故意不走** `rt._cast_flat_ptr_bf16` —— 它带影子缓存，
会把"同一地址的上一次结果"直接返回，正是 0.9.87 修过的那个坑。
实验要求每次都是真实量化，所以直接发内核。
"""
import argparse
import ctypes
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import _acc

T.set_default_device("cuda")
rt = T._cuda()

N_QUANT = [0]


def to_bf16(buf):
    """fp32 -> bf16（uint16 位模式），**无缓存**。"""
    n = int(buf.n)
    out = rt.empty((n,), dtype=np.uint16)
    grid, block = rt._ewise_grid(n)
    rt.launch("cast_f32_to_bf16", grid, block, [
        (int(buf.ptr), ctypes.c_uint64), rt._ptr(out), (n, ctypes.c_int)])
    return out


def from_bf16(buf):
    n = int(buf.n)
    return rt._cast_bf16_to_f32_flat(buf, n)


def quant_tensor(t):
    """把 Tensor 经 bf16 往返；反向直通（STE）。"""
    if not isinstance(t, T.Tensor) or t.device != "cuda":
        return t
    f = from_bf16(to_bf16(t._gpu))
    f = rt.reshape(f, tuple(t.shape))     # 升回来是一维缓冲，形状要还原
    N_QUANT[0] += 1
    out = T.Tensor._from_gpu(f, requires_grad=t.requires_grad,
                             _parents=(t,) if t.requires_grad else ())
    if t.requires_grad:
        def bw(g, grads):
            _acc(grads, t, g)
        out._backward = bw
    return out


def check_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4096).astype(np.float32)
    g = rt.h2d(x)
    back = np.asarray(rt.d2h(from_bf16(to_bf16(g))), np.float32)
    rel = np.abs(back - x) / np.maximum(np.abs(x), 1e-30)
    # bf16 有效位 p=8（1 隐含 + 7 显式）⇒ RTNE 最大相对误差 = 2^-p = 2^-8 = 3.907e-3
    bound = 2 ** -8
    print("[往返] max_rel = %.3e  (RTNE 上界 2^-8 = %.3e)"
          % (float(rel.max()), bound))
    ok = float(rel.max()) <= bound * 1.0001
    print("[往返] %s" % ("OK" if ok else "**超界，通道有问题**"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()

    if not check_roundtrip():
        return 2

    if a.quant:
        import tgai_go.train.v3_ops as V
        orig = V.rmsnorm_orv
        V.rmsnorm_orv = lambda x, w: quant_tensor(orig(x, w))
        import tgai_go.train.v3_train_cli as CLI
        CLI.V.rmsnorm_orv = V.rmsnorm_orv
        print("[量化] 已把 rmsnorm_orv 的输出改为 bf16 往返（STE 梯度）")

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", "512", "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", "2816", "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "1", "--no-final-save"]
    try:
        from tgai_go.train import v3_pretrain as P
        P.main()
    except SystemExit:
        pass
    print("[量化次数] %d" % N_QUANT[0])


if __name__ == "__main__":
    sys.exit(main() or 0)
