# -*- coding: utf-8 -*-
"""验证 fp16 权重影子缓存的失效语义（最可疑的一层）。

假设：`_cast_f16_cached` 按**指针**缓存 fp16 影子；AdamW 是**原地**更新权重
（指针不变），若调用方不显式失效，前向就会用上一步的旧 fp16 权重，
而反向用的是当前 fp32 权重 ⇒ 前后向不一致 ⇒ 梯度错（实测梯度范数差 11000 倍）。

本测试：改权重后不清缓存再看结果，然后再清缓存看结果。
"""
import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")
rng = np.random.default_rng(0)

M, K, N = 256, 512, 512
x_n = rng.normal(0, 1.0, (M, K)).astype(np.float32)
w1 = rng.normal(0, 0.02, (N, K)).astype(np.float32)
w2 = w1 + 0.5                      # 明显不同的新权重

x = T.tensor(x_n.copy())


def y_of(w_np, tag):
    w = T.tensor(w_np.copy())
    w._gpu_hint = tag
    y = x @ w.T
    return np.asarray(y.to_numpy(), np.float32)


T.amp_on(dtype="fp16", weight_cache=True)
rt.f16_cache_clear()

# 同一个张量对象原地改数据：用 _gpu 缓冲的 h2d 覆盖
w = T.tensor(w1.copy())
y_a = np.asarray((x @ w.T).to_numpy(), np.float32)      # 建影子
# 原地覆盖权重（指针不变）——模拟 AdamW 的原地更新
new_buf = rt.h2d(w2)
rt._activate()
rt.memcpy_dtod(w._gpu, new_buf) if hasattr(rt, "memcpy_dtod") else None
if not hasattr(rt, "memcpy_dtod"):
    # 退路：直接替换张量内部缓冲（指针会变，另测）
    w = T.tensor(w2.copy())

y_b = np.asarray((x @ w.T).to_numpy(), np.float32)      # 未清缓存
rt.f16_cache_clear()
y_c = np.asarray((x @ w.T).to_numpy(), np.float32)      # 清缓存后

ref2 = (x_n.astype(np.float64) @ w2.astype(np.float64).T)
rel = lambda a: float(np.abs(a - ref2).max() / max(np.abs(ref2).max(), 1e-30))
print("改权重后（未清缓存） rel=%.3e" % rel(y_b))
print("清缓存后         rel=%.3e" % rel(y_c))
print("判定：%s" % ("缓存会失效 ⇒ 必须显式清" if rel(y_b) > 1e-2 and rel(y_c) < 2e-3
                 else "缓存未观察到陈旧"))
