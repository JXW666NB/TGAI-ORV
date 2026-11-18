# -*- coding: utf-8 -*-
"""查两张卡之间能不能 P2P（cuDeviceCanAccessPeer / cuDeviceGetP2PAttribute）。

为什么关键：DP 的梯度归约现在走「主机中转」（d2h → 共享内存 → h2d），
3.16 GiB 要 8~9 秒。若支持 P2P，就能让 GPU 直接读写对端显存，省掉两次 PCIe 搬运。
消费级 3090 通常**不支持** P2P，这里实测确认。
"""
import ctypes

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
cu = rt.cu

n = ctypes.c_int()
cu.cuDeviceGetCount(ctypes.byref(n))
print("设备数：%d" % n.value)

devs = []
for i in range(n.value):
    d = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(d), ctypes.c_int(i))
    devs.append(d)
    name = ctypes.create_string_buffer(256)
    cu.cuDeviceGetName(name, 256, d)
    print("  [%d] %s" % (i, name.value.decode("utf-8", "replace")))

print("\n=== cuDeviceCanAccessPeer ===")
for i in range(n.value):
    for j in range(n.value):
        if i == j:
            continue
        can = ctypes.c_int(-1)
        rc = cu.cuDeviceCanAccessPeer(ctypes.byref(can), devs[i], devs[j])
        print("  %d -> %d : rc=%d can=%s" % (i, j, rc, bool(can.value)))

# 若有 1.6+ 的 cuDeviceGetP2PAttribute，查性能档位（0=不支持）
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 1
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 2
fn = getattr(cu, "cuDeviceGetP2PAttribute", None)
print("\n=== cuDeviceGetP2PAttribute ===")
if fn is None:
    print("  驱动未导出该函数")
else:
    for i in range(n.value):
        for j in range(n.value):
            if i == j:
                continue
            for att, nm in ((CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, "性能档位"),
                            (CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, "可访问"),
                            (CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, "原生原子")):
                v = ctypes.c_int(-1)
                rc = fn(ctypes.byref(v), ctypes.c_int(att), devs[i], devs[j])
                print("  %d -> %d %-8s rc=%d val=%d" % (i, j, nm, rc, v.value))
