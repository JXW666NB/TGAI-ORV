# -*- coding: utf-8 -*-
"""Vulkan 后端：能力探测 + SPIR-V 计算内核加载（预编译，无运行时 GLSL JIT）。

## 现实约束（先说清）

Vulkan 没有"把 GLSL 字符串丢进去就出机器码"的标准 API（`glslangValidator`/`glslc` 是
独立工具）。所以 Vulkan 插件只能**预编译**：用户交 `.spv`（SPIR-V 二进制），
引擎负责建 instance/device/pipeline 并 dispatch。

## 当前状态

* `available()` / `devices()`：**已实现**，纯 ctypes，用来判断本机核显/独显有没有 Vulkan
* `load_spirv()`：骨架已搭，但**还没做完整 dispatch**（buffer 绑定、descriptor set、
  pipeline barrier 全套）。真跑起来需要一套内存管理，属于独立工作量；
  在没做完之前**明确抛错并说明**，不会静默返回错结果。

## 为什么值得留这一档

核显（Intel/AMD iGPU）通常有 Vulkan 但没 CUDA。对"插件也能在核显上跑"这个目标，
Vulkan 是唯一通道 —— 所以先把它探测清楚，再决定投入。
"""
import ctypes
import os
import sys

VK_SUCCESS = 0
VK_TYPE_NAMES = {0: "other", 1: "integrated-gpu", 2: "discrete-gpu", 3: "virtual-gpu", 4: "cpu"}


class _VkAppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_VkAppInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class _VkPhysicalDeviceProperties(ctypes.Structure):
    # 只需要前两个字段（apiVersion / driverVersion）与后面的 deviceType 位置不连续，
    # 所以这里只取前缀，deviceName/deviceType 用单独的小结构读。
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32),
                ("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("deviceType", ctypes.c_int), ("deviceName", ctypes.c_char * 256),
                ("pipelineCacheUUID", ctypes.c_uint8 * 16)]


def _load():
    names = (["vulkan-1.dll"] if sys.platform == "win32"
             else ["libvulkan.so.1", "libvulkan.so"])
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


def available():
    """本机有没有可用的 Vulkan loader。"""
    lib = _load()
    if lib is None:
        return False
    try:
        v = ctypes.c_uint32()
        if hasattr(lib, "vkEnumerateInstanceVersion"):
            return lib.vkEnumerateInstanceVersion(ctypes.byref(v)) == VK_SUCCESS
        return True
    except Exception:                                  # noqa: BLE001
        return True


def devices():
    """枚举物理设备 → [{'name','type','api'}...]；不可用返回空列表。"""
    lib = _load()
    if lib is None:
        return []
    try:
        app = _VkAppInfo(0, None, b"tgai", 1, b"tgai", 1, 0x00400000)
        ci = _VkInstanceCreateInfo(0, None, 0, ctypes.pointer(app), 0, None, 0, None)
        inst = ctypes.c_void_p()
        rc = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
        if rc != VK_SUCCESS:
            return []
        n = ctypes.c_uint32(0)
        rc = lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        if rc != VK_SUCCESS or n.value == 0:
            lib.vkDestroyInstance(inst, None)
            return []
        arr = (ctypes.c_void_p * n.value)()
        rc = lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), arr)
        out = []
        if rc == VK_SUCCESS:
            lib.vkGetPhysicalDeviceProperties.restype = None
            lib.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p,
                                                          ctypes.POINTER(_VkPhysicalDeviceProperties)]
            for i in range(n.value):
                props = _VkPhysicalDeviceProperties()
                lib.vkGetPhysicalDeviceProperties(arr[i], ctypes.byref(props))
                out.append({
                    "name": props.deviceName.decode("utf-8", "replace"),
                    "type": VK_TYPE_NAMES.get(int(props.deviceType), "other"),
                    "api": "%d.%d.%d" % (props.apiVersion >> 22,
                                         (props.apiVersion >> 12) & 0x3FF,
                                         props.apiVersion & 0xFFF),
                })
        lib.vkDestroyInstance(inst, None)
        return out
    except Exception:                                  # noqa: BLE001
        return []


def load_spirv(path, entry="main"):
    """加载预编译 SPIR-V 计算内核。

    **当前未实现 dispatch**：Vulkan 的 buffer/descriptor/pipeline 管理是独立工作量，
    在没做完之前明确报错，避免"以为能跑其实算错"。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到 SPIR-V：%s。修法：用 glslc 把 .comp 编成 .spv" % path)
    devs = devices()
    if not devs:
        raise RuntimeError(
            "本机没有可用的 Vulkan 设备（或 loader 缺失）。修法：装显卡驱动里的 Vulkan 运行时；"
            "核显通常自带 Vulkan，但没有 CUDA，此时可用 python 档兜底")
    raise NotImplementedError(
        "Vulkan 的 dispatch 还没实现（已探测到设备：%s）。"
        "当前可用后端：cuda（NVIDIA）与 cpu。修法：插件包里加 python 或 cpu 档"
        % ", ".join(d["name"] for d in devs))
