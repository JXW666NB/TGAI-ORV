# bf16/fp16 动量优化器的 checkpoint 续训 —— 实施规格

> 现状：`--opt fp16mom --opt-state bf16` 是跑 `1.5B / seq1024` 的推荐配置
> （省 5.6 GiB，实测 27.14 GiB / 855 tok/s），但**非 adamw 优化器明确拒绝
> checkpoint 与续训**（`v3_pretrain.py:1071` 的守卫），所以长跑会被挡住。
> 这份文档把实施要点与坑写清楚，避免下次重踩。

## 一、为什么不能直接复用现有的存取函数

```python
def _buffer_to_np(buf):                      # v3_pretrain.py:600
    return np.ascontiguousarray(T._cuda().d2h(buf), np.float32)   # ← 强制 fp32

def _load_into_buffer(buf, arr):             # v3_pretrain.py:606
    arr = np.ascontiguousarray(arr, np.float32)                   # ← 强制 fp32
    T._cuda().copy_h2d_into(buf, arr)                             # ← 与 uint16 缓冲冲突
```

`Fp16MomAdamW` 的 `m16` / `v16` 是 **float16**（`state="fp16"`）或 **uint16**（`state="bf16"`
把 bf16 位模式存在 uint16 里）的设备缓冲。用上面两个函数会：

* `state="fp16"`：静默转成 fp32 存盘（体积 ×2），读回再转 fp16 —— 精度虽不丢，但**白费一半磁盘**；
* `state="bf16"`：**直接坏**。uint16 的位模式被 `astype(np.float32)` 当成数值解释，
  读回来是垃圾 ⇒ 续训静默产出错误训练状态。**这是最危险的路径**。

## 二、需要新增的东西

1. `_buffer_to_np_raw(buf)`：`np.ascontiguousarray(T._cuda().d2h(buf))` —— **保留原 dtype**。
2. `_load_into_buffer_raw(buf, arr)`：
   - 校验 `arr.dtype == buf.dtype`，不符立刻报错（**不要静默转换**）；
   - 形状同样校验；
   - 设备侧用 `copy_h2d_into`，主机侧用 `np.copyto`。
3. `save_optimizer` / `load_optimizer` 按类型分派：
   ```python
   if hasattr(opt, "m16"):        # Fp16MomAdamW
       ... 存 m16/v16 + t.txt + state.txt（记 "fp16"/"bf16"，读回必须匹配）...
   ```
   注意 `Fp16MomAdamW` 没有 `.m` / `.v`，现有实现会直接 `AttributeError`。
4. 放宽 `v3_pretrain.py:1071` 的守卫：从 `a.opt != "adamw"` 改为
   `a.opt not in ("adamw", "fp16mom")`，其余（muon/adafactor/int8）继续拒绝。

## 三、验收（做不到就不许合并）

按"续训必须**逐位**一致"的标准，三步都要做：

1. **存取往返**：随机填 `m16/v16` → 存 → 读进新优化器 → 逐位相等
   （`np.array_equal`，不是 `allclose`）。
2. **续训一致性**：跑到第 3 步存盘 → 从该盘续训 → 与不中断跑到第 4 步的
   **loss 逐位相同**（参考已用同一方法验证过 adamw：`|d| = 0.000e+00`）。
3. **dtype 断言**：`state="bf16"` 下读出后 `dtype == uint16`，
   `state="fp16"` 下 `dtype == float16`；两者**不许互相转换成功**。

## 四、风险提示

* 存档路径一旦有 bug，会在**长跑恢复时静默**产出错误状态 —— 这是最坏的一类 bug。
  所以宁可一开始就拒绝（现状），也不要"看起来能存"。
* 现有 `tests/test_v3_pretrain.py` 有 `test_optimizer_roundtrip_cpu` 与
  `test_optimizer_roundtrip_fp32_exact` 两个用例可作模板（CPU 侧）。
