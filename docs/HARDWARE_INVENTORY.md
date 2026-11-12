# 硬件与能力清单（实测，2026-11-12）

> 用途：**任何关于「多卡 / P2P / 混用」的规划都必须先看这张表**。
> 2026-11-12 发生过两次判断错误，都是因为凭记忆假设硬件拓扑。本文只写实测到的。

## 1. 两台机器，各只有一半能力

| | **笔记本（agent 本机）** | **远端（SSH `Administrator@192.168.0.103`）** |
| --- | --- | --- |
| 系统 | Windows | Windows |
| 显卡 | **AMD Radeon (TM) Graphics**（iGPU） | **Tesla V100-PCIE-32GB** + **GeForce GT 730** |
| **CUDA** | ❌ 无 NVIDIA、无 `nvidia-smi` | ✅ **1 个**：`GPU 0: Tesla V100-PCIE-32GB`<br>UUID `GPU-6b615e44-f60e-f629-5337-849d1154239a` |
| **Vulkan** | ✅ **1 个设备**：`AMD Radeon (TM) Graphics`，`apiVersion 1.3.260`，`PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` | ❌ **0 个**：`vulkaninfo` 报 `Failed to detect any valid GPUs`（`vkEnumeratePhysicalDevices` → `ERROR_INITIALIZATION_FAILED`）|
| Vulkan ICD | 有（可用）| **`HKLM\SOFTWARE\Khronos\Vulkan\Drivers` 不存在** ⇒ 未注册 ICD。<br>`C:\Windows\System32\vulkan-1.dll` 与 `vulkan-1-999-0-0-0.dll` 在，但只是 loader |
| 驱动 | AMD `31.0.21924.4004` | V100 `32.0.15.8088`（2025-07-27）<br>GT730 `27.21.14.5671`（2020-09-30）|

## 2. 由此得出的硬结论

**① 任何一台机器都没有「CUDA + Vulkan 同时可用」。**

- 笔记本：只有 Vulkan（AMD iGPU），没有 NVIDIA。
- 远端：只有 CUDA（V100），Vulkan 枚举不到设备。

⇒ **`docs/VULKAN_BACKEND.md` 里那次 Vulkan 前向/反向实测（AMD 核显、1.3.260、对 numpy 逐位精确）
是在笔记本上做的**，与 V100 无关。

**② 没有任何一台机器有「两张可用卡」。**

- 笔记本只有 1 个 Vulkan 设备。
- 远端 `nvidia-smi -L` **只列 V100**；GT730 在 CUDA 侧不可用（现代驱动已淘汰 Kepler/Fermi）。

**③ 跨厂商组合在当前硬件上根本不存在。**

- AMD 在笔记本、NVIDIA 在远端，**两台机器之间没有 PCI 总线** ⇒ 物理上不可能 P2P，
  只能走网络（那是分布式训练，与 TG Link 的 P2P 无关）。

⇒ **TG Link 的核心假设（跨厂商 Vulkan P2P）目前【无法验证】**，不是"代码没写"，是**硬件不具备**。

## 3. GT730 现在是一块"两边都不可用"的卡

| 路径 | 状态 | 原因 |
| --- | --- | --- |
| CUDA | ❌ | `nvidia-smi -L` 不列它；驱动 `27.21.14.5671`（2020）与 V100 的 `32.0.15.8088` 属不同分支，两版 NVIDIA 驱动并存 |
| Vulkan | ❌ | 整机 Vulkan ICD 未注册（不是这块卡单独的问题）|

**理论上可尝试的修复**（未验证，仅供后续参考）：在 `DriverStore` 里找到 `nvoglv64.dll`
（NVIDIA 的 Vulkan ICD），手工在 `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` 注册其完整路径。
注册后 `vulkaninfo` 若能看到 V100 或 GT730，才有资格谈 TG Link 的实验。

## 4. 因此：什么能做、什么不能做

| 事项 | 可行性 |
| --- | --- |
| 探针的**静态能力层**（枚举设备、查扩展与句柄类型交集） | ✅ 两台都能跑；但两台各自只有 1 个设备 ⇒ 配对结论为空 |
| 探针的**实测层**（导出/导入/带宽） | ❌ 需要同一台机器上至少 2 个 Vulkan 设备 |
| 跨厂商 P2P 验证 | ❌ 硬件不存在 |
| 同厂商 P2P（V100↔GT730） | ❌ GT730 CUDA 不可用、Vulkan 未注册 |
| **host-staged 集合通信（tier 3）** | ✅ 只需多个进程，不需要特殊硬件 |
| 跨机 DP（笔记本 iGPU + 远端 V100，走网络） | ⚠️ 正确性可测、速度不可测；且**要先有 Vulkan 训练后端**（目前 0 个内核）|

## 5. 给未来的规划者

1. **要验证 TG Link P2P，必须有一台同时具备 ≥2 个 Vulkan 设备（最好跨厂商）的机器。**
   租机时选「同机双卡」并且确认两张卡都有 Vulkan ICD —— 云上常见的 4090×2 也满足
   "2 个 Vulkan 设备"，可用于验证**机制**（跨厂商仍需 NVIDIA+AMD 同机，很难租到）。
2. **不要在笔记本上做 CUDA 实验**，也不要在远端假设 Vulkan 可用。
3. 改这份表之前先实测，不要凭 `docs/` 里的旧描述推断 —— 这次两次错误都是这么来的。
