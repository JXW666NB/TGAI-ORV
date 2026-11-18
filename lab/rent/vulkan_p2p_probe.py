# -*- coding: utf-8 -*-
"""Vulkan P2P 能力探针（不依赖第三方库，直接 ctypes 调 libvulkan）。

回答三个问题：
  1. 系统有几张 Vulkan 物理设备，它们是否组成**设备组**（device group）？
     Vulkan 没有 CUDA 那样的"peer access"概念，最接近的就是 device group +
     `vkGetDeviceGroupPeerMemoryFeatures`（能不能做 peer copy）。
  2. 暴露了哪些与跨设备/跨 API 共享内存相关的扩展
     （external_memory_fd / dma_buf / buffer_device_address）。
  3. 若构成设备组，直接查 peer memory features 的位。

用法：python3 vulkan_p2p_probe.py
"""
import ctypes
import ctypes.util

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO = 1000044000

# VkDeviceGroupPeerMemoryFeatures 位
PEER_BITS = {1: "PEER_COPY_LOCAL", 2: "PEER_COPY_REMOTE", 4: "PEER_COPY_TRANSFER_ONLY",
             8: "PEER_COPY_BIDIRECTIONAL", 16: "PEER_COPY_SAME_DEVICE",
             31: "PEER_COPY_DEVICE_MASK"}

INTERESTING = ("device_group", "external_memory", "dma_buf", "buffer_device_address",
               "p2p", "peer", "external_semaphore", "memory_budget")


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32), ("pApplicationInfo", ctypes.c_void_p),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32),
                ("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("deviceType", ctypes.c_int), ("deviceName", ctypes.c_char * 256),
                ("pipelineCacheUUID", ctypes.c_uint8 * 16)]


class VkPhysicalDeviceGroupProperties(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("physicalDeviceCount", ctypes.c_uint32),
                ("physicalDevices", ctypes.c_void_p * 32),
                ("subsetAllocation", ctypes.c_uint32)]


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", ctypes.c_uint32)]


def main():
    name = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
    vk = ctypes.CDLL(name)
    print("libvulkan:", name)

    ai = VkApplicationInfo()
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO
    ai.pApplicationName = b"orv-p2p-probe"
    ai.apiVersion = (1 << 22) | (1 << 12)      # 1.1

    want = [b"VK_KHR_device_group_creation", b"VK_KHR_external_memory_capabilities",
            b"VK_KHR_external_semaphore_capabilities", b"VK_KHR_get_physical_device_properties2"]
    arr = (ctypes.c_char_p * len(want))(*want)
    ci = VkInstanceCreateInfo()
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    ci.pApplicationInfo = ctypes.cast(ctypes.byref(ai), ctypes.c_void_p)
    ci.enabledExtensionCount = len(want)
    ci.ppEnabledExtensionNames = ctypes.cast(arr, ctypes.c_void_p)

    inst = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    if rc != VK_SUCCESS:
        print("vkCreateInstance 失败 rc=%d（可能缺 libvulkan / ICD）" % rc)
        return
    print("vkCreateInstance OK")

    n = ctypes.c_uint32()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
    devs = (ctypes.c_void_p * n.value)()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
    print("\n物理设备数：%d" % n.value)
    for i in range(n.value):
        p = VkPhysicalDeviceProperties()
        vk.vkGetPhysicalDeviceProperties(devs[i], ctypes.byref(p))
        print("  [%d] %-28s type=%d vendor=0x%04x api=%d.%d.%d"
              % (i, p.deviceName.decode("utf-8", "replace"), p.deviceType, p.vendorID,
                 p.apiVersion >> 22, (p.apiVersion >> 12) & 0x3ff, p.apiVersion & 0xfff))

    # ---- 设备组（P2P 的基础）----
    print("\n=== 设备组 / device group ===")
    fn = getattr(vk, "vkEnumeratePhysicalDeviceGroups", None)
    if fn is None:
        print("  没有 vkEnumeratePhysicalDeviceGroups（Vulkan < 1.1）")
    else:
        ng = ctypes.c_uint32()
        fn(inst, ctypes.byref(ng), None)
        print("  设备组数：%d" % ng.value)
        groups = (VkPhysicalDeviceGroupProperties * max(1, ng.value))()
        for g in groups:
            g.sType = 1000044001        # VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES
        if ng.value:
            fn(inst, ctypes.byref(ng), groups)
            for gi in range(ng.value):
                g = groups[gi]
                print("  组[%d]：%d 张卡，subsetAllocation=%d%s"
                      % (gi, g.physicalDeviceCount, g.subsetAllocation,
                         "  ← 允许把一个逻辑设备拆到多卡" if g.subsetAllocation else ""))

    # ---- 每张卡的相关扩展 ----
    print("\n=== 与跨设备/跨 API 共享内存相关的**设备**扩展 ===")
    for i in range(n.value):
        cnt = ctypes.c_uint32()
        vk.vkEnumerateDeviceExtensionProperties(devs[i], None, ctypes.byref(cnt), None)
        props = (VkExtensionProperties * cnt.value)()
        vk.vkEnumerateDeviceExtensionProperties(devs[i], None, ctypes.byref(cnt), props)
        hits = []
        for e in props:
            nm = e.extensionName.decode("utf-8", "replace")
            if any(k in nm.lower() for k in INTERESTING):
                hits.append(nm)
        print("  [%d] %d 个扩展，命中 %d 个：" % (i, cnt.value, len(hits)))
        for h in sorted(hits):
            print("        %s" % h)

    vk.vkDestroyInstance(inst, None)


if __name__ == "__main__":
    main()
