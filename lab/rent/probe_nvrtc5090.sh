#!/bin/bash
# NVRTC probe (ASCII only): version, whether it knows compute_120, pip-installed newer one
echo "=== nvrtc on system ==="
ldconfig -p 2>/dev/null | grep -i nvrtc || echo "  none in ldconfig"
ls -1 /usr/local/cuda*/lib64/libnvrtc.so* 2>/dev/null | head -5
PY=/root/miniconda3/bin/python
$PY - <<'PY'
import ctypes, glob, os
pats = ("/usr/local/cuda*/lib64/libnvrtc.so*",
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
        "/root/miniconda3/lib/libnvrtc.so*",
        "/root/miniconda3/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*")
cands = []
for p in pats:
    cands += sorted(glob.glob(p))
print("  candidates:", cands or "none")
for p in cands[:6]:
    try:
        lib = ctypes.CDLL(p)
        ma, mi = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(ma), ctypes.byref(mi))
        has = b"compute_120" in open(p, "rb").read()
        print("   %-72s nvrtc %d.%d  knows_120=%s" % (p, ma.value, mi.value, has))
    except Exception as e:
        print("   %-72s open failed: %s" % (p, type(e).__name__))
PY

echo "=== pip nvidia/torch ==="
$PY -m pip list 2>/dev/null | grep -i -E "nvidia|torch" | head -8

echo "=== install nvrtc 12.8 into /root/nvrtc128 ==="
$PY -m pip install -q --no-deps --target /root/nvrtc128 \
    --index-url https://pypi.org/simple --no-cache-dir \
    "nvidia-cuda-nvrtc-cu12==12.8.93" 2>&1 | tail -3
find /root/nvrtc128 -name 'libnvrtc.so*' 2>/dev/null | head -3
