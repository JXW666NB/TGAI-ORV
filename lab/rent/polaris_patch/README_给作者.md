# Polaris 性能反馈（来自 ORV 核心维护方）

作者：yyfdfs。以下全部基于**实测**，不是推测。测试环境：V100-PCIE-32GB / Win Server / ORV 0.9.97 / numpy 1.26。

---

## 一、最重要的一条：CUDA 路径上的 Newton-Schulz 其实跑在 **CPU** 上

`Polaris.step()` 里（`polaris.py`）：

| 行 | 代码 | 后果 |
| --- | --- | --- |
| L254 | `g = np.asarray(p.grad.to_numpy() ...)` | 每个参数 **D2H 梯度** |
| L161 / L306 / L317 | `np.asarray(p.data, np.float32)` | 每个参数 **D2H 权重** |
| L274 / L281 | `d = _polar_express_np(u, ...)` | **NS 在主机 numpy 上跑** |
| L308 / L321 | `p.data = new_w` | 每个参数 **H2D 写回** |

也就是说：**当参数在 CUDA 上时，整个更新仍然是把张量搬回主机、用 numpy 算完再搬回去。**
GPU 只承担了"存放参数"，没有参与 NS 的矩阵乘。

实测（9,441,536 参数 / 8 层 d=256 / B=2×S=64 / CUDA event / 预热+3 次中位数）：

| 优化器 | opt.step | 占整步 |
| --- | --- | --- |
| ORV `Int8AdamWGpu` | 0.261 ms | 1.3% |
| ORV `AdamW` | 0.341 ms | 1.7% |
| **Polaris** | **56.79 ms** | **75%** |
| ORV `Muon` | 40.04 ms | 67% |

对照：ORV 内置 Muon 的 NS **是全设备端的**（`cuda.py::_newton_schulz5_gpu`，值全程留在显存，
`X@X.T` 走自写的顺序 FMA 内核，与 numpy fp32 逐位相同）。它就是"同一公式的设备端参考实现"，
可以直接照搬：**把 `_polar_express_np` 换成设备端版本后，NS 的矩阵乘从 CPU 的 ~200 ms
（1024×1024）降到 GPU 的 ~5 ms 量级。**

---

## 二、可以直接应用的补丁：`_polar_express_np` 的平方和

`_polar_express_np` 里求范数用的是**Python 逐元素循环**：

```python
    # 顺序 float32 平方和，和 GPU sumsq 同一舍入口径（避免 NS 放大 ulp 差）
    acc = np.float32(0.0)
    for v in np.ascontiguousarray(x, np.float32).ravel():
        acc = np.float32(acc + np.float32(v * v))
    nrm = np.float32(math.sqrt(float(acc)) + eps)
```

26 万参数的一个矩阵就要 **663 ms**；1.05M 参数模型的 CPU `opt.step` 因此高达 620~813 ms
（AdamW 只要 8 ms）。改成 BLAS 的 float32 `dot`：

```diff
-    # 顺序 float32 平方和，和 GPU sumsq 同一舍入口径（避免 NS 放大 ulp 差）
-    acc = np.float32(0.0)
-    for v in np.ascontiguousarray(x, np.float32).ravel():
-        acc = np.float32(acc + np.float32(v * v))
-    nrm = np.float32(math.sqrt(float(acc)) + eps)
+    # 数学等价，差异只来自累加顺序（实测相对差 ~1e-7）；
+    # Newton-Schulz 会把范数的 1 ulp 差放大，需要与 GPU 逐位对拍时设 POLARIS_SUMSQ_EXACT=1
+    if os.environ.get("POLARIS_SUMSQ_EXACT") == "1":
+        acc = np.float32(0.0)
+        for v in np.ascontiguousarray(x, np.float32).ravel():
+            acc = np.float32(acc + np.float32(v * v))
+    else:
+        _flat = np.ascontiguousarray(x, np.float32).ravel()
+        acc = np.float32(np.dot(_flat, _flat))
+    nrm = np.float32(math.sqrt(float(acc)) + eps)
```

（需要在文件头 `import os`。）

### 实测效果（本地 CPU，同一输入，steps=5）

| 形状 | 原实现 | 补丁后 | 加速 | max\|Δ\| | 相对差 |
| --- | --- | --- | --- | --- | --- |
| 256×1024 | 439.67 ms | **17.24 ms** | **25.5×** | 5.6e-06 | 3.9e-05 |
| 512×512 | 440.91 ms | 39.60 ms | 11.1× | 8.1e-06 | 4.3e-05 |
| 1024×1024 | 1926.51 ms | **201.18 ms** | 9.6× | 3.6e-05 | 2.6e-04 |
| 64×4096 | 415.86 ms | 15.02 ms | 27.7× | 2.3e-06 | 4.5e-05 |

补丁后 1024×1024 剩下的 201 ms 就是 NS 的矩阵乘本身（10.7 GFLOP 跑在 CPU fp32 上），
这也正说明**真正的瓶颈是"NS 在 CPU 上"**，不只是那个循环。

---

## 三、另外两条（来自独立基准测试，供参考）

1. **稳定窗口窄**：`lr=3e-3` 先在 step 127 前后收敛到 <1e-6，**step 189 起进入极限环**
   （0.05~0.36 之间来回）；`lr=1e-2` 第 45 步崩到 1.16 不恢复。CPU/CUDA 都能复现 ⇒ 不是设备问题。
   推测原因：NS 按 `‖u‖+eps` 归一，梯度趋零时更新幅度仍是 O(lr)。建议加 lr 退火或梯度范数下限。
2. **`ns_saved_pct` 名不副实**：实测只有 **3.3~5.0%**（`orth_freq` 全程为 1，`dir_cos` 上限
   0.9587 < 阈值 0.995），文档宣称的"省算力"在这个模型上基本不成立。

---

## 四、一句话结论

Polaris 的**算法**没问题（在固定批过拟合任务上 5.5e-8，与母算法 Muon 同量级、远好于 AdamW 的 2.1e-4），
问题在**实现层**：CUDA 时仍走主机往返 + CPU 上的 NS。先应用上面的小补丁（9.6~27.7×），
再考虑把 NS 搬到设备端（参考 ORV 的 `_newton_schulz5_gpu`，预计再快一个数量级）。

补丁文件在同一目录：`polaris_patched.py`（`polaris_orig.py` 是原文件，便于 diff）。
