# 外部报告问题台账（2026-11-12）

> 来源：外部测试者用 `tgai-orv 0.9.53` 在 **GTX 1060 6GB（sm_61）** 上的完整报告。
> 这里逐条列问题、状态、处置方式。**未修的也写清**，不留"以为能用"的错觉。

## P0（会导致功能不可用）

| # | 问题 | 状态 |
| --- | --- | --- |
| 1 | **wheel 不含推理用的原生库**（`_native/*.dll`），`pip install` 后 `tgai_go.load()` 报 `FileNotFoundError` | **未修**。需要把 `_native` 纳入 package-data；但见第 2 条 —— 光打包会把坏 DLL 一起发出去 |
| 2 | **仓库里 Windows 引擎 DLL 的 PE 头错位 1 字节**（`libtgai_engine.dll`：`e_lfanew=0xF8` 而 PE 实际在 `0xF7`；VK 版 0x108 vs 0x107）。手工拷贝也 `WinError 193` | **未修**。要重编 DLL，属于独立任务；**当前不要把"训推一体"当成立** |

⇒ 结论：**推理端在 pip 形态下不可用**，这是已知状态。训练端不受影响。

## P1（影响正确使用或回归能力）

| # | 问题 | 状态 |
| --- | --- | --- |
| 3 | **Pascal 消费卡上融合注意力反向必失败**：需 68 KB（g2 变体 84 KB）动态 smem，sm_61 上限仅 48 KB 且不支持 opt-in ⇒ 旧版报 `cuFuncSetAttribute rc=1 invalid argument` | ✅ **已修（0.9.55）**：先查设备上限再设 opt-in，报错写明上限、卡型与替代方案。回归用例 `tests/test_attn_smem_guard.py`，V100 上 2 passed；dh=64 数值仍逐位一致 |
| 4 | **Windows 上"不用装 CUDA Toolkit"不成立**：驱动目录没有 NVRTC/cuBLAS，`tgai-orv[cuda]` extra 只对 Linux 生效 | ✅ **文档已修**（skill §2.1）|
| 5 | **测试在有 GPU 的机器上系统性变红（14 个）**：6 个顺序污染（`set_default_device("cuda")` 不还原 + 用 `.data` 读值）、5 个"假设本机无 CUDA"、2 个真缺陷（`plugin.op_native` 抛裸 `TypeError`；CUDA Graph 节点数断言 `>100` 过严）| **未修**。这是当前最该做的 —— **没有 GPU 回归门，Pascal 那类问题只能靠外部发现** |

## P2（工程卫生/数字不一致）

| # | 问题 | 状态 |
| --- | --- | --- |
| 6 | 文档数字互斥：README 徽章 656 / 正文 558 / `STATUS.md` 160 / 实际收集 710；版本徽章仍写 0.9.49 | **未修** |
| 7 | `tools/kernel_count.py` 等脚本硬编码作者本机路径（`D:\tgai_orv\...`、`D:/SFT语料库`）| **未修**（同类约十余个 `align_*.py`）|
| 8 | 仓库卫生：`release/` 40 MB exe、`backups/` 3.77 MB、`cuda.py.bak-*`、`ppo_trainer_part.txt` 与源码混放 | **未修** |

## 外部报告同时确认的事（不需要改）

* **速度与显存没有造假**：1060 上 dense 1.3–1.6×、MoE 2.2–2.6×；显存 1.2–1.94×，与 README 的 V100 数字同向。
* **数值对齐成立**：loss 相对偏差 3e-07–8.06e-06，AdamW 100 步 rel 1.57e-06 / cos 1.000000000。
* **1060 的容量边界**：约 **419M 参数**是 6 GB 可用上限（Fp16AdamW+检查点 4.85 GiB）；
  651M 时 WDDM 溢到共享内存，步时从 ~1 s 涨到 10.4 s。
* **CUDA 内核对拍从没进过 CI**（CI 只有 ubuntu + CPU torch）——第 5 条的直接后果。
