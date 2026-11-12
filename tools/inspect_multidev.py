# -*- coding: utf-8 -*-
"""确认 ORV 现有的多卡代码是按进程还是按设备组织的。

这决定「把脏活交给弱卡」属于：
  A) 多设备（同后端、不同 device_index）—— 小工作量
  B) 后端混用（不同 API / 不同厂商）—— 大工程
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "tgai_go", "train")


def read(name):
    p = os.path.join(PKG, name)
    return io.open(p, encoding="utf-8").read() if os.path.isfile(p) else ""


print("=== dist.py 公开接口 ===")
s = read("dist.py")
print("  ", re.findall(r"(?m)^(?:def |class )(\w+)", s))
i = s.find('"""')
if i >= 0:
    j = s.find('"""', i + 3)
    print("  模块说明:", s[i + 3:j].strip().replace("\n", " ")[:420])

print()
print("=== 是否出现 device index / 多上下文 ===")
for kw in ("device_index", "set_device", "cuCtxCreate", "cuDeviceGet",
           "rank", "world_size", "nccl", "NCCL", "p2p", "peer"):
    c = s.count(kw)
    if c:
        print("  dist.py  %-14s %d" % (kw, c))

print()
print("=== cuda.py 里上下文/设备的选择方式 ===")
cu = read("cuda.py")
for kw in ("cuCtxCreate", "cuDeviceGet(", "cuDevicePrimaryCtxRetain",
           "CU_DEVICE_ATTRIBUTE_MULTI_GPU", "self.ctx", "TGAI_DEVICE"):
    print("  %-32s %d" % (kw, cu.count(kw)))

print()
print("=== 全局单例假设 ===")
for m in re.finditer(r"(?m)^(_CUDA\w*|_INSTANCE\w*|_RT\w*)\s*=.*$", cu):
    print("  ", m.group(0).strip()[:90])
