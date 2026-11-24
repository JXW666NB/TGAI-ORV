# 外部报告问题台账（2026-11-12）

> 来源：外部测试者用 `tgai-orv 0.9.53` 在 **GTX 1060 6GB（sm_61）** 上的完整报告。
> 这里逐条列问题、状态、处置方式。**未修的也写清**，不留"以为能用"的错觉。

## P0（会导致功能不可用）

| # | 问题 | 状态 |
| --- | --- | --- |
| 1 | **wheel 不含推理用的原生库**（`_native/*.dll`），`pip install` 后 `tgai_go.load()` 报 `FileNotFoundError` | ✅ **已修（0.9.57）**：`pyproject.toml` 补 `"tgai_go" = ["_native/*.dll", ...]`；从 PyPI 回读的 wheel 已确认含 `libomp.dll` / `libtgai_engine.dll` / `libtgai_engine_vk.dll`。同时把"缺库"和"加载失败"两种情形都改成**带修法的指引** |
| 2 | **仓库里 Windows 引擎 DLL 的 PE 头错位 1 字节**（`e_lfanew=0xF8` 而 PE 在 `0xF7`）。手工拷贝也 `WinError 193` | 🟡 **报错已改善、根因未修**：`ctypes.CDLL` 失败的 `OSError` 现在会被包成 `RuntimeError`，其中 WinError 193 明确指出"随包 DLL 损坏、需重编"，并给出三条替代路径。**根因需要重编 DLL，本机没有 C 编译器，无法完成** |

⇒ 结论：**打包路径已修通，但 Windows 上的推理仍然不可用**，直到 DLL 重编。
这是已知状态，不要对外宣称"训推一体"已在 pip 形态下成立。训练端不受影响。

## P1（影响正确使用或回归能力）

| # | 问题 | 状态 |
| --- | --- | --- |
| 3 | **Pascal 消费卡上融合注意力反向必失败**：需 68 KB（g2 变体 84 KB）动态 smem，sm_61 上限仅 48 KB 且不支持 opt-in ⇒ 旧版报 `cuFuncSetAttribute rc=1 invalid argument` | ✅ **已修（0.9.55）**：先查设备上限再设 opt-in，报错写明上限、卡型与替代方案。回归用例 `tests/test_attn_smem_guard.py`，V100 上 2 passed；dh=64 数值仍逐位一致 |
| 4 | **Windows 上"不用装 CUDA Toolkit"不成立**：驱动目录没有 NVRTC/cuBLAS，`tgai-orv[cuda]` extra 只对 Linux 生效 | ✅ **文档已修**（skill §2.1）|
| 5 | **测试在有 GPU 的机器上系统性变红（14 个）**：6 个顺序污染（`set_default_device("cuda")` 不还原 + 用 `.data` 读值）、5 个"假设本机无 CUDA"、2 个真缺陷（`plugin.op_native` 抛裸 `TypeError`；CUDA Graph 节点数断言 `>100` 过严）| **未修**。这是当前最该做的 —— **没有 GPU 回归门，Pascal 那类问题只能靠外部发现** |

## P2（工程卫生/数字不一致）

| # | 问题 | 状态 |
| --- | --- | --- |
| 6 | 文档数字互斥：README 徽章 656 / 正文 558 / `STATUS.md` 160 / 实际收集 710；版本徽章仍写 0.9.49 | ✅ **已修**：版本徽章改为 **shields 动态读 PyPI**（`img.shields.io/pypi/v/tgai-orv`），彻底不再手写；测试数与内核数不再写死，改为指向脚本生成的 `CAPABILITY_AUDIT.md`；正文 `# 0.9.49` 与 `558 passed` 一并改掉 |
| 7 | 工具脚本硬编码作者本机路径（`D:\tgai_orv\...`、`D:/SFT语料库`）| ✅ **已处理**：`kernel_count.py` 改为从 `V27_RUN` / `V27_TG` 环境变量读，缺失时明确说明"这是作者本机工具"并指向 `profile_block.py`；新增 `tools/README.md` 把「通用工具」与「作者本机一次性脚本」分开登记。其余 `align_*.py` 等属第二类，**保留可追溯、不再假装能直接跑** |
| 8 | 仓库卫生：`release/` 40 MB exe、`backups/` 3.77 MB、`cuda.py.bak-*`、`ppo_trainer_part.txt` 与源码混放 | 🟡 **部分修**：`backups/`（18 个）与源码内 `.bak`/空 txt **已从索引移除并加入 gitignore**（文件仍在磁盘上，历史可查）；`release/`（38 MB / 45 文件）**有意保留** —— 有 3 处文档引用它，是 ESP32 线的真实交付物 |

## 外部报告同时确认的事（不需要改）

* **速度与显存没有造假**：1060 上 dense 1.3–1.6×、MoE 2.2–2.6×；显存 1.2–1.94×，与 README 的 V100 数字同向。
* **数值对齐成立**：loss 相对偏差 3e-07–8.06e-06，AdamW 100 步 rel 1.57e-06 / cos 1.000000000。
* **1060 的容量边界**：约 **419M 参数**是 6 GB 可用上限（Fp16AdamW+检查点 4.85 GiB）；
  651M 时 WDDM 溢到共享内存，步时从 ~1 s 涨到 10.4 s。
* **CUDA 内核对拍从没进过 CI**（CI 只有 ubuntu + CPU torch）——第 5 条的直接后果。
