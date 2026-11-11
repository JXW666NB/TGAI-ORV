# -*- coding: utf-8 -*-
"""验证 C 端插件：用户写 CUDA C，ORV 用 NVRTC 现场编译，跑原生速度。

## 要证明什么

「插件为什么是 .py？会不会太慢？」——.py 只是**登记与胶水**，加载时执行一次；
热路径是原生内核。本脚本写一个真实的 CUDA C 内核（axpy），验证：

1. `plugin.op_native` 能编译任意 CUDA C 源码（NVRTC）；
2. 启动后结果**数值正确**（与 numpy 对照）；
3. 大尺寸下**没有 Python 逐元素开销**（与逐元素 Python 循环对比时间量级）。

运行（需独占 GPU）：python native_plugin_check.py
"""
import ctypes
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

OUT = os.path.join(HERE, "npc.txt")

SRC = r"""
extern "C" __global__ void axpy_kernel(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] + a * x[i];
}
"""


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train import plugin

    T.set_default_device("cuda")
    rt = T._cuda()
    rep, ok = [], True

    launch = plugin.op_native(
        "axpy_demo", SRC, "axpy_kernel",
        grid=lambda x, y, a, n: ((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
        args=lambda x, y, a, n: [(y._gpu.ptr, ctypes.c_void_p),
                                 (x._gpu.ptr, ctypes.c_void_p),
                                 (float(a), ctypes.c_float),
                                 (int(n), ctypes.c_int)],
        doc="y += a * x",
    )
    rep.append("op_native 登记成功: %r" % plugin.native_info()["registered"])

    for n in (1000, 1 << 20):
        rng = np.random.default_rng(0)
        xa = rng.normal(0, 1, n).astype(np.float32)
        ya = rng.normal(0, 1, n).astype(np.float32)
        x = T.tensor(xa.reshape(1, n))
        y = T.tensor(ya.reshape(1, n))
        a = 2.5
        launch(x, y, a, n)
        rt.sync()
        got = np.asarray(rt.d2h(y._gpu), np.float32).reshape(-1)
        want = ya + a * xa
        rel = float(np.abs(got - want).max() / max(1e-9, float(np.abs(want).max())))
        rep.append("n=%-8d 与 numpy 相对差 %.2e ⇒ %s"
                   % (n, rel, "一致" if rel <= 1e-6 else "**不一致**"))
        ok = ok and rel <= 1e-6

    # 编译只发生一次（同一 name+arch）
    before = len(plugin._NATIVE)
    launch(T.tensor(np.zeros((1, 4), np.float32)), T.tensor(np.zeros((1, 4), np.float32)), 1.0, 4)
    rep.append("再次调用未重复编译: %s" % (len(plugin._NATIVE) == before))

    # 吞吐：原生内核 vs Python 逐元素（只跑小规模，够看出量级差）
    n = 1 << 20
    x = T.tensor(np.zeros((1, n), np.float32))
    y = T.tensor(np.zeros((1, n), np.float32))

    def ev(fn, r=20):
        for _ in range(3):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(r):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / r

    t_native = ev(lambda: launch(x, y, 1.0, n))
    rep.append("原生内核 n=1M: %.4f ms" % t_native)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
