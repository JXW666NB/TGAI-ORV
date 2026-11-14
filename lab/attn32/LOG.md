# 实验日志

## #0 自检：实验场本身是否可信（2026-11-15）

把生产内核 python/tgai_go/train/kernels/tgai_attn_nop.cu **原样拷**成 kernel.cu，
用 `TGAI_ATTN_NOP_CU` 指向它跑对拍。

| | 与生产基线最大差 |
| --- | --- |
| y / dq / dk / dv | **全部 0.000e+00** |

**结论：实验场可信。** 后续所有内核实验都在这里做，不再碰生产代码。

基线数值（记住，任何实验都要和它比）：
- y 和 = 388.641815
- dq 和 = 0.166557

## 待做

- [ ] #1 只在实验场把 BT 降到 32（保留 8 行/warp 映射以外的部分），逐项对拍
## #1 行步长改 CUTLASS 式 padding（BT+4）—— 通过，逐位相同

**改动**：`kernel.cu` 里 `#define QSTR (DH + 8)` → `#define QSTR (BT + 4)`（72 → 68）。

| | 与基线最大差 |
| --- | --- |
| y / dq / dk / dv | **全部 0.000e+00** |

**意义**：

1. 证明**改 padding 步长在数值上安全**（步长只是布局，不影响结果）；
2. smem 从 69632 → **66048 B（64.5 KB）**；
3. `BT+4` 正是窗口化要用的步长 —— 这一步消掉了后续改动的一个不确定项。

**下一步**（三步清单第 2、3 步）：`smQ`/`smDO` 从"64 维常驻"改成"16 维窗口 + 外层 chunk 循环"，
预期再降到约 41 KB，进 48 KB 以内。

## 下一刀：Q/dO 改 head_dim 滑窗（方案已定，待执行）

**依据**：CUTLASS fp32/sm70 的 smem 里 Q/dO 都不常驻，只放 head_dim 的 16 宽滑窗
（见 DESIGN.md 第 1 节）。我们用 18+18=36 KB 常驻，这是 68 KB 的主要来源。

**为什么敢做**：head_dim 是**求和维**，沿它切分累加数学等价 ——
分块不成立只针对 key 维（softmax 归一化），与此无关。

**预期**：68 − 27 = **41 KB < 48 KB**，且**完全不动 BT/线程映射**（前三次全错在那里）。

**改动清单（只有这三处，逐个改逐个对拍）**

| # | 位置（kernel.cu）| 改什么 | 验证 |
| --- | --- | --- | --- |
| 1 | smem 切分（约 575–580 行）| smQ/smDO 从 64*QSTR 改成 16*(BT+4) 的窗口 | 编译 + 对拍 |
| 2 | lash_qk_dp 的内层 d 循环 | 外面套 or chunk in 0..3：先把 Q/dO 的该 16 宽窗口载入 smem，再跑现有 16 次 d | y 必须逐位相同 |
| 3 | dK/dV 写回循环里的 smQ[d*QSTR+q] / smDO[...] | 同样按 chunk 重载窗口 | dq/dk/dv 相对 < 1e-5 |

**逐步验证命令**（每一步都要跑）：

`at
python lab\attn32\run.py --cu lab\attn32\kernel.cu --out D:\_lab_tN
python lab\attn32\run.py --compare D:\_lab_base D:\_lab_tN
`

**基线**（任何一步偏离就要停）：y 和 = 388.641815、dq 和 = 0.166557；
y/dq/dk/dv 与基线相对差 < 1e-5。

**已知风险**
- 窗口化后 loat4 读是否还成立：窗口是 [dim][row] 的 16 行 dim × BT 行，行步长改用 BT+4
  （CUTLASS 的做法），不是 72 —— **bank 冲突可能变差，但数值不变**
- 第 3 步最容易漏：dK/dV 循环现在读的是**全 64 维**的 Q/dO，必须改造循环而不是只改下标
## #2 head_dim 滑窗 —— **通过**并已进生产

DCHUNK=16，新增 load_win / lash_qk_dp_win / ttn_bwd_flash_w（45056 B = 44 KB）。
对拍：y 逐位相同，dq/dk/dv 相对 1.2e-7 / 1.7e-7 / 1.0e-7。
48 KB 上限下自动启用；V100 仍走原内核（已用 probe_attn_kernel_choice 证明）。
全套测试 18/18 与基线一致；守卫测试 3 passed（新增 48 KB 可用用例）。
