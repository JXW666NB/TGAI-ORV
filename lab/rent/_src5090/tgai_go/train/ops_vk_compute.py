# -*- coding: utf-8 -*-
"""Vulkan compute dispatch（最小可用）：在核显/独显上跑 SPIR-V 计算内核。

设计取舍（先跑通，再优化）：
* 缓冲全部 **HOST_VISIBLE | HOST_COHERENT** ⇒ 无需 staging、无需显式 flush，
  读写直接映射（核显上本来就和系统内存同体，几乎无损失）。
* 每次 dispatch 重建 pipeline 太慢，所以按 SPIR-V 字节做缓存。
* 只支持"若干 storage buffer + push constant 无"的简单算子 —— 正好是插件算子的形状。

用法：
    dev = VulkanCompute()                       # 建 instance/device/queue
    out = dev.run(spv_bytes, [a, b], out_bytes=n*4)
    dev.close()
"""
import ctypes
import os
import struct

from tgai_go.train import ops_vk

# ---- sType / enum 常量（Vulkan 1.1）----
ST_APP = 0
ST_INSTANCE = 1
ST_DEVICE_QUEUE = 2
ST_DEVICE = 3
ST_SUBMIT = 4
ST_MEMORY_ALLOC = 5
ST_FENCE = 8
ST_BUFFER = 12
ST_SHADER_MODULE = 16
ST_PIPELINE_SHADER_STAGE = 18
ST_COMPUTE_PIPELINE = 29
ST_PIPELINE_LAYOUT = 30
ST_DESCRIPTOR_SET_LAYOUT = 32
ST_DESCRIPTOR_POOL = 33
ST_DESCRIPTOR_SET_ALLOC = 34
ST_WRITE_DESCRIPTOR_SET = 35
ST_COMMAND_POOL = 39
ST_COMMAND_BUFFER_ALLOC = 40
ST_COMMAND_BUFFER_BEGIN = 42
ST_MEMORY_BARRIER = 46

USAGE_STORAGE = 0x20
USAGE_TRANSFER_DST = 0x2
SHARING_EXCLUSIVE = 0
DESC_STORAGE_BUFFER = 7
STAGE_COMPUTE = 0x20
BIND_COMPUTE = 1  # VK_PIPELINE_BIND_POINT_COMPUTE（之前错写成 4，导致绑定非法、写被丢弃）
CMD_ONE_TIME_SUBMIT = 0x1
QUEUE_COMPUTE = 0x2
MEM_HOST_VISIBLE = 0x2
MEM_HOST_COHERENT = 0x4
VK_SUCCESS = 0


class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class _InstanceCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_AppInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class _QueueFamilyProps(ctypes.Structure):
    _fields_ = [("queueFlags", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("timestampValidBits", ctypes.c_uint32),
                ("minImageTransferGranularity", ctypes.c_uint32 * 3)]


class _DeviceQueueCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("queueFamilyIndex", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class _DeviceCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("queueCreateInfoCount", ctypes.c_uint32),
                ("pQueueCreateInfos", ctypes.POINTER(_DeviceQueueCI)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p),
                ("pEnabledFeatures", ctypes.c_void_p)]


class _BufferCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("usage", ctypes.c_uint32),
                ("sharingMode", ctypes.c_int), ("queueFamilyIndexCount", ctypes.c_uint32),
                ("pQueueFamilyIndices", ctypes.c_void_p)]


class _MemoryReq(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("alignment", ctypes.c_uint64),
                ("memoryTypeBits", ctypes.c_uint32)]


class _MemoryAlloc(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("allocationSize", ctypes.c_uint64), ("memoryTypeIndex", ctypes.c_uint32)]


class _MemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _MemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _MemoryProps(ctypes.Structure):
    _fields_ = [("memoryTypeCount", ctypes.c_uint32), ("memoryTypes", _MemoryType * 32),
                ("memoryHeapCount", ctypes.c_uint32), ("memoryHeaps", _MemoryHeap * 16)]


class _DescSetLayoutBinding(ctypes.Structure):
    _fields_ = [("binding", ctypes.c_uint32), ("descriptorType", ctypes.c_int),
                ("descriptorCount", ctypes.c_uint32), ("stageFlags", ctypes.c_uint32),
                ("pImmutableSamplers", ctypes.c_void_p)]


class _DescSetLayoutCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("bindingCount", ctypes.c_uint32),
                ("pBindings", ctypes.POINTER(_DescSetLayoutBinding))]


class _PipelineLayoutCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("setLayoutCount", ctypes.c_uint32),
                ("pSetLayouts", ctypes.POINTER(ctypes.c_void_p)),
                ("pushConstantRangeCount", ctypes.c_uint32),
                ("pPushConstantRanges", ctypes.c_void_p)]


class _ShaderModuleCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("codeSize", ctypes.c_size_t), ("pCode", ctypes.c_void_p)]


class _PipelineShaderStageCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("stage", ctypes.c_uint32), ("module", ctypes.c_void_p),
                ("pName", ctypes.c_char_p), ("pSpecializationInfo", ctypes.c_void_p)]


class _ComputePipelineCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("stage", _PipelineShaderStageCI), ("layout", ctypes.c_void_p),
                ("basePipelineHandle", ctypes.c_void_p), ("basePipelineIndex", ctypes.c_int32)]


class _DescriptorPoolSize(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("descriptorCount", ctypes.c_uint32)]


class _DescriptorPoolCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("maxSets", ctypes.c_uint32), ("poolSizeCount", ctypes.c_uint32),
                ("pPoolSizes", ctypes.POINTER(_DescriptorPoolSize))]


class _DescriptorSetAllocI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("descriptorPool", ctypes.c_void_p), ("descriptorSetCount", ctypes.c_uint32),
                ("pSetLayouts", ctypes.POINTER(ctypes.c_void_p))]


class _DescriptorBufferInfo(ctypes.Structure):
    _fields_ = [("buffer", ctypes.c_void_p), ("offset", ctypes.c_uint64),
                ("range", ctypes.c_uint64)]


class _WriteDescriptorSet(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("dstSet", ctypes.c_void_p), ("dstBinding", ctypes.c_uint32),
                ("dstArrayElement", ctypes.c_uint32), ("descriptorCount", ctypes.c_uint32),
                ("descriptorType", ctypes.c_int),
                ("pImageInfo", ctypes.c_void_p),
                ("pBufferInfo", ctypes.POINTER(_DescriptorBufferInfo)),
                ("pTexelBufferView", ctypes.c_void_p)]


class _CommandPoolCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("queueFamilyIndex", ctypes.c_uint32)]


class _CommandBufferAllocI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("commandPool", ctypes.c_void_p), ("level", ctypes.c_int),
                ("commandBufferCount", ctypes.c_uint32)]


class _CommandBufferBeginI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pInheritanceInfo", ctypes.c_void_p)]


class _MemoryBarrier(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("srcAccessMask", ctypes.c_uint32), ("dstAccessMask", ctypes.c_uint32)]


class _SubmitInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("waitSemaphoreCount", ctypes.c_uint32), ("pWaitSemaphores", ctypes.c_void_p),
                ("pWaitDstStageMask", ctypes.c_void_p),
                ("commandBufferCount", ctypes.c_uint32),
                ("pCommandBuffers", ctypes.POINTER(ctypes.c_void_p)),
                ("signalSemaphoreCount", ctypes.c_uint32),
                ("pSignalSemaphores", ctypes.c_void_p)]


class _FenceCI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]




_V = ctypes.c_void_p
_U = ctypes.c_uint32
_U64 = ctypes.c_uint64
# (restype, argtypes) —— 64 位句柄必须显式声明，否则 ctypes 会按 int 传而溢出
_SIGS = {
    "vkCreateInstance": (ctypes.c_int, [_V, _V, _V]),
    "vkEnumeratePhysicalDevices": (ctypes.c_int, [_V, _V, _V]),
    "vkGetPhysicalDeviceQueueFamilyProperties": (None, [_V, _V, _V]),
    "vkCreateDevice": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkGetDeviceQueue": (None, [_V, _U, _U, _V]),
    "vkGetPhysicalDeviceMemoryProperties": (None, [_V, _V]),
    "vkCreateCommandPool": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkCreateBuffer": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkGetBufferMemoryRequirements": (None, [_V, _V, _V]),
    "vkAllocateMemory": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkBindBufferMemory": (ctypes.c_int, [_V, _V, _V, _U64]),
    "vkMapMemory": (ctypes.c_int, [_V, _V, _U64, _U64, _U, _V]),
    "vkUnmapMemory": (None, [_V, _V]),
    "vkCreateDescriptorSetLayout": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkCreatePipelineLayout": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkCreateShaderModule": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkCreateComputePipelines": (ctypes.c_int, [_V, _V, _U, _V, _V, _V]),
    "vkCreateDescriptorPool": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkAllocateDescriptorSets": (ctypes.c_int, [_V, _V, _V]),
    "vkUpdateDescriptorSets": (None, [_V, _U, _V, _U, _V]),
    "vkAllocateCommandBuffers": (ctypes.c_int, [_V, _V, _V]),
    "vkBeginCommandBuffer": (ctypes.c_int, [_V, _V]),
    "vkCmdBindPipeline": (None, [_V, ctypes.c_int, _V]),
    "vkCmdBindDescriptorSets": (None, [_V, ctypes.c_int, _V, _U, _U, _V, _U, _V]),
    "vkCmdDispatch": (None, [_V, _U, _U, _U]),
    "vkCmdPipelineBarrier": (None, [_V, _U, _U, _U, _U, _V, _U, _V, _U, _V]),
    "vkEndCommandBuffer": (ctypes.c_int, [_V]),
    "vkCreateFence": (ctypes.c_int, [_V, _V, _V, _V]),
    "vkQueueSubmit": (ctypes.c_int, [_V, _U, _V, _V]),
    "vkWaitForFences": (ctypes.c_int, [_V, _U, _V, _U, _U64]),
    "vkDestroyFence": (None, [_V, _V, _V]),
    "vkFreeCommandBuffers": (None, [_V, _V, _U, _V]),
    "vkDestroyBuffer": (None, [_V, _V, _V]),
    "vkFreeMemory": (None, [_V, _V, _V]),
    "vkDestroyDescriptorPool": (None, [_V, _V, _V]),
    "vkDestroyPipeline": (None, [_V, _V, _V]),
    "vkDestroyShaderModule": (None, [_V, _V, _V]),
    "vkDestroyPipelineLayout": (None, [_V, _V, _V]),
    "vkDestroyDescriptorSetLayout": (None, [_V, _V, _V]),
    "vkDestroyCommandPool": (None, [_V, _V, _V]),
    "vkDestroyDevice": (None, [_V, _V]),
    "vkDestroyInstance": (None, [_V, _V]),
}


def _bind(lib):
    for name, (rt, at) in _SIGS.items():
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.restype = rt
        fn.argtypes = at


class VulkanComputeError(RuntimeError):
    """Vulkan 计算不可用或失败（消息里带修法）。"""


class VulkanCompute:
    def __init__(self):
        self.lib = ops_vk._load()
        if self.lib is None:
            raise VulkanComputeError(
                "找不到 Vulkan loader（vulkan-1.dll / libvulkan.so.1）。"
                "修法：装显卡驱动自带的 Vulkan 运行时")
        _bind(self.lib)
        self._init_instance()
        self._init_device()
        self._init_pool()
        self._cache = {}

    # ---------- 初始化 ----------
    def _chk(self, rc, what):
        if rc != VK_SUCCESS:
            raise VulkanComputeError("%s 失败 rc=%d。修法：把这段输出发出来，"
                                     "或先用 cpu/python 后端" % (what, rc))

    def _init_instance(self):
        L = self.lib
        L.vkCreateInstance.restype = ctypes.c_int
        app = _AppInfo(ST_APP, None, b"tgai", 1, b"tgai", 1, 0x00400000)
        ci = _InstanceCI(ST_INSTANCE, None, 0, ctypes.pointer(app), 0, None, 0, None)
        self.inst = ctypes.c_void_p()
        self._chk(L.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(self.inst)),
                  "vkCreateInstance")
        n = ctypes.c_uint32(0)
        L.vkEnumeratePhysicalDevices.restype = ctypes.c_int
        self._chk(L.vkEnumeratePhysicalDevices(self.inst, ctypes.byref(n), None),
                  "vkEnumeratePhysicalDevices(count)")
        if n.value == 0:
            raise VulkanComputeError("本机没有 Vulkan 物理设备。修法：装驱动里的 Vulkan 运行时")
        arr = (ctypes.c_void_p * n.value)()
        self._chk(L.vkEnumeratePhysicalDevices(self.inst, ctypes.byref(n), arr),
                  "vkEnumeratePhysicalDevices(list)")
        self.pdev = arr[0]
        # 找支持 compute 的队列族
        fc = ctypes.c_uint32(0)
        L.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
        L.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(fc), None)
        props = (_QueueFamilyProps * max(1, fc.value))()
        L.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(fc), props)
        self.qfam = None
        for i in range(fc.value):
            if props[i].queueFlags & QUEUE_COMPUTE and props[i].queueCount > 0:
                self.qfam = i
                break
        if self.qfam is None:
            raise VulkanComputeError("该设备的 Vulkan 驱动没有 compute 队列。修法：换设备或更新驱动")

    def _init_device(self):
        L = self.lib
        prio = ctypes.c_float(1.0)
        qci = _DeviceQueueCI(ST_DEVICE_QUEUE, None, 0, self.qfam, 1, ctypes.pointer(prio))
        dci = _DeviceCI(ST_DEVICE, None, 0, 1, ctypes.pointer(qci), 0, None, 0, None, None)
        self.dev = ctypes.c_void_p()
        L.vkCreateDevice.restype = ctypes.c_int
        self._chk(L.vkCreateDevice(self.pdev, ctypes.byref(dci), None, ctypes.byref(self.dev)),
                  "vkCreateDevice")
        q = ctypes.c_void_p()
        L.vkGetDeviceQueue.restype = None
        L.vkGetDeviceQueue(self.dev, self.qfam, 0, ctypes.byref(q))
        self.queue = q
        mp = _MemoryProps()
        L.vkGetPhysicalDeviceMemoryProperties.restype = None
        L.vkGetPhysicalDeviceMemoryProperties(self.pdev, ctypes.byref(mp))
        self.mem_type = None
        for i in range(mp.memoryTypeCount):
            f = mp.memoryTypes[i].propertyFlags
            if (f & MEM_HOST_VISIBLE) and (f & MEM_HOST_COHERENT):
                self.mem_type = i
                break
        if self.mem_type is None:
            raise VulkanComputeError(
                "找不到 HOST_VISIBLE|HOST_COHERENT 内存类型（核显通常有）。"
                "修法：这条后端目前只支持可直接映射的内存")

    def _init_pool(self):
        L = self.lib
        pci = _CommandPoolCI(ST_COMMAND_POOL, None, 0, self.qfam)
        self.cmd_pool = ctypes.c_void_p()
        L.vkCreateCommandPool.restype = ctypes.c_int
        self._chk(L.vkCreateCommandPool(self.dev, ctypes.byref(pci), None,
                                        ctypes.byref(self.cmd_pool)), "vkCreateCommandPool")
        # 复用：命令缓冲与 fence 只建一次（每次 dispatch 重建成 1.8 ms 固定开销）
        cba = _CommandBufferAllocI(ST_COMMAND_BUFFER_ALLOC, None, self.cmd_pool, 0, 1)
        self._cmd = ctypes.c_void_p()
        L.vkAllocateCommandBuffers.restype = ctypes.c_int
        self._chk(L.vkAllocateCommandBuffers(self.dev, ctypes.byref(cba),
                                             ctypes.byref(self._cmd)), "vkAllocateCommandBuffers")
        fci = _FenceCI(ST_FENCE, None, 0)
        self._fence = ctypes.c_void_p()
        L.vkCreateFence.restype = ctypes.c_int
        self._chk(L.vkCreateFence(self.dev, ctypes.byref(fci), None,
                                  ctypes.byref(self._fence)), "vkCreateFence")
        self._buf_pool = {}          # nbytes -> [buffer dict, ...]（空闲可复用）
        L.vkResetCommandBuffer.restype = ctypes.c_int
        L.vkResetFences.restype = ctypes.c_int

    # ---------- 资源（带池化）----------
    def _acquire(self, nbytes):
        pool = self._buf_pool.setdefault(int(nbytes), [])
        if pool:
            return pool.pop()
        return self._buffer(int(nbytes))

    def _release(self, b):
        self._buf_pool.setdefault(b["n"], []).append(b)

    def _buffer(self, nbytes, usage=USAGE_STORAGE):
        L = self.lib
        ci = _BufferCI(ST_BUFFER, None, 0, int(nbytes), usage, SHARING_EXCLUSIVE, 0, None)
        buf = ctypes.c_void_p()
        L.vkCreateBuffer.restype = ctypes.c_int
        self._chk(L.vkCreateBuffer(self.dev, ctypes.byref(ci), None, ctypes.byref(buf)),
                  "vkCreateBuffer")
        req = _MemoryReq()
        L.vkGetBufferMemoryRequirements.restype = None
        L.vkGetBufferMemoryRequirements(self.dev, buf, ctypes.byref(req))
        ai = _MemoryAlloc(ST_MEMORY_ALLOC, None, req.size, self.mem_type)
        mem = ctypes.c_void_p()
        L.vkAllocateMemory.restype = ctypes.c_int
        self._chk(L.vkAllocateMemory(self.dev, ctypes.byref(ai), None, ctypes.byref(mem)),
                  "vkAllocateMemory")
        L.vkBindBufferMemory.restype = ctypes.c_int
        self._chk(L.vkBindBufferMemory(self.dev, buf, mem, 0), "vkBindBufferMemory")
        ptr = ctypes.c_void_p()
        L.vkMapMemory.restype = ctypes.c_int
        self._chk(L.vkMapMemory(self.dev, mem, 0, int(nbytes), 0, ctypes.byref(ptr)),
                  "vkMapMemory")
        return {"buf": buf, "mem": mem, "ptr": ptr, "n": int(nbytes)}

    def _pipeline(self, spv, nbuf):
        key = (spv, nbuf)
        if key in self._cache:
            return self._cache[key]
        L = self.lib
        bindings = (_DescSetLayoutBinding * nbuf)()
        for i in range(nbuf):
            bindings[i] = _DescSetLayoutBinding(i, DESC_STORAGE_BUFFER, 1, STAGE_COMPUTE, None)
        dslci = _DescSetLayoutCI(ST_DESCRIPTOR_SET_LAYOUT, None, 0, nbuf, bindings)
        dsl = ctypes.c_void_p()
        L.vkCreateDescriptorSetLayout.restype = ctypes.c_int
        self._chk(L.vkCreateDescriptorSetLayout(self.dev, ctypes.byref(dslci), None,
                                                ctypes.byref(dsl)),
                  "vkCreateDescriptorSetLayout")
        sets = (ctypes.c_void_p * 1)(dsl)
        plci = _PipelineLayoutCI(ST_PIPELINE_LAYOUT, None, 0, 1, sets, 0, None)
        pl = ctypes.c_void_p()
        L.vkCreatePipelineLayout.restype = ctypes.c_int
        self._chk(L.vkCreatePipelineLayout(self.dev, ctypes.byref(plci), None, ctypes.byref(pl)),
                  "vkCreatePipelineLayout")
        code = ctypes.create_string_buffer(spv, len(spv))
        smci = _ShaderModuleCI(ST_SHADER_MODULE, None, 0, len(spv),
                               ctypes.cast(code, ctypes.c_void_p))
        sm = ctypes.c_void_p()
        L.vkCreateShaderModule.restype = ctypes.c_int
        self._chk(L.vkCreateShaderModule(self.dev, ctypes.byref(smci), None, ctypes.byref(sm)),
                  "vkCreateShaderModule")
        stage = _PipelineShaderStageCI(ST_PIPELINE_SHADER_STAGE, None, 0, STAGE_COMPUTE, sm,
                                       b"main", None)
        cpci = _ComputePipelineCI(ST_COMPUTE_PIPELINE, None, 0, stage, pl, None, -1)
        pipe = ctypes.c_void_p()
        L.vkCreateComputePipelines.restype = ctypes.c_int
        self._chk(L.vkCreateComputePipelines(self.dev, None, 1, ctypes.byref(cpci), None,
                                             ctypes.byref(pipe)),
                  "vkCreateComputePipelines")
        psi = _DescriptorPoolSize(DESC_STORAGE_BUFFER, max(1, nbuf))
        # flags 用 0：之前传的 0x2 不是合法 flag（FREE_DESCRIPTOR_SET 是 0x1）
        dpci = _DescriptorPoolCI(ST_DESCRIPTOR_POOL, None, 0, 1, 1, ctypes.pointer(psi))
        pool = ctypes.c_void_p()
        L.vkCreateDescriptorPool.restype = ctypes.c_int
        self._chk(L.vkCreateDescriptorPool(self.dev, ctypes.byref(dpci), None,
                                           ctypes.byref(pool)), "vkCreateDescriptorPool")
        dsai = _DescriptorSetAllocI(ST_DESCRIPTOR_SET_ALLOC, None, pool, 1, sets)
        dset = ctypes.c_void_p()
        L.vkAllocateDescriptorSets.restype = ctypes.c_int
        self._chk(L.vkAllocateDescriptorSets(self.dev, ctypes.byref(dsai), ctypes.byref(dset)),
                  "vkAllocateDescriptorSets")
        info = {"dsl": dsl, "pl": pl, "sm": sm, "pipe": pipe, "pool": pool, "dset": dset}
        self._cache[key] = info
        return info

    # ---------- 执行 ----------
    def run(self, spv, inputs, out_bytes, n_elem, local_size=64, prefill=None):
        """跑一个计算内核：inputs 是字节串列表，返回 out_bytes 字节的输出。

        `prefill`：先把输出缓冲填成这个字节串（诊断用，验证内核到底有没有写）。
        """
        L = self.lib
        p = self._pipeline(spv, len(inputs) + 1)
        bufs = [self._acquire(len(b)) for b in inputs]
        obuf = self._acquire(out_bytes)
        for b, data in zip(bufs, inputs):
            ctypes.memmove(b["ptr"], data, len(data))
        if prefill is not None:
            ctypes.memmove(obuf["ptr"], prefill, min(len(prefill), out_bytes))
        # 每个 write 需要自己的 VkDescriptorBufferInfo 指针；用 list 持有引用防 GC
        infos = [_DescriptorBufferInfo(b["buf"], 0, b["n"]) for b in bufs + [obuf]]
        writes = (_WriteDescriptorSet * (len(bufs) + 1))()
        for i in range(len(bufs) + 1):
            writes[i] = _WriteDescriptorSet(ST_WRITE_DESCRIPTOR_SET, None, p["dset"], i, 0, 1,
                                            DESC_STORAGE_BUFFER, None,
                                            ctypes.pointer(infos[i]), None)
        L.vkUpdateDescriptorSets.restype = None
        L.vkUpdateDescriptorSets(self.dev, len(bufs) + 1, writes, 0, None)

        # 复用同一个命令缓冲 + fence（每次重建约 1.8 ms，是之前慢 numpy 几十倍的主因）
        cmd = self._cmd
        fence = self._fence
        self._chk(L.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer")
        begin = _CommandBufferBeginI(ST_COMMAND_BUFFER_BEGIN, None, CMD_ONE_TIME_SUBMIT, None)
        L.vkBeginCommandBuffer.restype = ctypes.c_int
        self._chk(L.vkBeginCommandBuffer(cmd, ctypes.byref(begin)), "vkBeginCommandBuffer")
        L.vkCmdBindPipeline(cmd, BIND_COMPUTE, p["pipe"])
        dsets = (ctypes.c_void_p * 1)(p["dset"])
        L.vkCmdBindDescriptorSets(cmd, BIND_COMPUTE, p["pl"], 0, 1, dsets, 0, None)
        groups = (int(n_elem) + local_size - 1) // local_size
        L.vkCmdDispatch(cmd, max(1, groups), 1, 1)
        # 让 compute 的写对 HOST 可见。之前这里掩码写错（用了 TRANSFER/HOST 的旧值），
        # GPU 写了但主机读到旧值 —— 表现就是"输出全 0"。
        # srcAccess=SHADER_WRITE(0x40) dstAccess=HOST_READ(0x2000)
        # srcStage=COMPUTE_SHADER(0x800) dstStage=HOST(0x4000)
        bar = _MemoryBarrier(ST_MEMORY_BARRIER, None, 0x40, 0x2000)
        L.vkCmdPipelineBarrier(cmd, 0x800, 0x4000, 0, 1, ctypes.byref(bar), 0, None, 0, None)
        L.vkEndCommandBuffer.restype = ctypes.c_int
        self._chk(L.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")
        self._chk(L.vkResetFences(self.dev, 1, ctypes.byref(fence)), "vkResetFences")
        cmds = (ctypes.c_void_p * 1)(cmd)
        si = _SubmitInfo(ST_SUBMIT, None, 0, None, None, 1, cmds, 0, None)
        L.vkQueueSubmit.restype = ctypes.c_int
        self._chk(L.vkQueueSubmit(self.queue, 1, ctypes.byref(si), fence), "vkQueueSubmit")
        L.vkWaitForFences.restype = ctypes.c_int
        self._chk(L.vkWaitForFences(self.dev, 1, ctypes.byref(fence), 1, 10 ** 10),
                  "vkWaitForFences")
        L.vkDeviceWaitIdle.restype = ctypes.c_int
        L.vkDeviceWaitIdle(self.dev)
        out = ctypes.string_at(obuf["ptr"], out_bytes)
        # 缓冲归还池子，下次同尺寸直接复用（不再每次新建/销毁）
        for b in bufs + [obuf]:
            self._release(b)
        return out

    def close(self):
        """幂等销毁。Vulkan 对象必须在进程退出前显式释放，否则驱动 teardown 会段错误。"""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        L = self.lib
        for pool in getattr(self, "_buf_pool", {}).values():
            for b in pool:
                L.vkUnmapMemory(self.dev, b["mem"])
                L.vkDestroyBuffer(self.dev, b["buf"], None)
                L.vkFreeMemory(self.dev, b["mem"], None)
        if getattr(self, "_buf_pool", None):
            self._buf_pool.clear()
        if getattr(self, "_fence", None):
            L.vkDestroyFence(self.dev, self._fence, None)
        if getattr(self, "_cmd", None):
            L.vkFreeCommandBuffers(self.dev, self.cmd_pool, 1, ctypes.byref(self._cmd))
        for info in self._cache.values():
            L.vkDestroyDescriptorPool(self.dev, info["pool"], None)
            L.vkDestroyPipeline(self.dev, info["pipe"], None)
            L.vkDestroyShaderModule(self.dev, info["sm"], None)
            L.vkDestroyPipelineLayout(self.dev, info["pl"], None)
            L.vkDestroyDescriptorSetLayout(self.dev, info["dsl"], None)
        self._cache.clear()
        L.vkDestroyCommandPool(self.dev, self.cmd_pool, None)
        L.vkDestroyDevice(self.dev, None)
        L.vkDestroyInstance(self.inst, None)
