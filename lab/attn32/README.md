# 注意力反向内核 · 隔离实验区

## 为什么要有这个目录

目标②（融合反向 smem 68 KB → ≤48 KB）在生产内核上试了三次，**三次都失败**，而且每次都是
"改了共享宏 → 连带别的函数 → 梯度错" 这种难以隔离的失败。前三次的教训：

| 失败 | 原因 |
| --- | --- |
| 行数 64→32（按行数缩容）| Q/dO 是 [dim][row] 布局，大小是 DH×步长，缩容导致写越界 |
| 加修正步长 | 仍然错（二分证明步长不是原因）|
| 二分 | 确认问题在"32 行"本身，但没定位到具体逻辑 |

**所以在生产内核上试错的方式是错的。** 这个目录提供一个完全隔离的实验场。

## 隔离机制：`TGAI_ATTN_NOP_CU`

运行时支持用环境变量覆盖内核源码路径（`cuda.py: attn_nop_source_path()`）：

```python
os.environ["TGAI_ATTN_NOP_CU"] = "D:/orv_src/lab/attn32/kernel.cu"
import tgai_go.train      # 之后加载的就是实验内核
```

⇒ **改实验内核不需要动生产代码，也不会影响任何人。**

## 实验规矩（必须遵守）

1. **先证明实验场本身是对的**：把生产内核原样拷进来，跑对拍，
   结果必须与生产**逐位相同**。做不到就先修实验场，别动算法。
2. **一次只改一件事**，每次改完都跑对拍（y 逐位 + dq/dk/dv 相对 < 1e-5）。
3. **失败要记录在案**：改了什么都写进 `LOG.md`，包括失败的。
4. 通过之后再谈"合回生产"，并且合回时必须是**独立函数/独立编译单元**，
   不能改共享宏（这是前三次失败的根源）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `kernel.cu` | 实验内核。初始为生产内核的**原样拷贝**（用于自检）|
| `run.py` | 跑一组数值并保存；或比较两组。支持 `--cu` 指定内核 |
| `LOG.md` | 实验日志（每次尝试 + 结果）|

## 用法

```bat
set PYTHONPATH=D:\orv_pip_0962\Lib\site-packages
:: 1) 自检：实验场拷出来的内核必须与生产逐位相同
python lab\attn32\run.py --out D:\_lab_base
python lab\attn32\run.py --cu lab\attn32\kernel.cu --out D:\_lab_copy
python lab\attn32\run.py --compare D:\_lab_base D:\_lab_copy

:: 2) 改实验内核，再跑
python lab\attn32\run.py --cu lab\attn32\kernel.cu --out D:\_lab_try1
python lab\attn32\run.py --compare D:\_lab_base D:\_lab_try1
```

> 注意：**一次进程只能测一个内核**（NVRTC 编译结果按名字缓存，
> 换内核必须换进程），所以上面每条命令都是独立进程。
