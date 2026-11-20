#!/bin/bash
# Get ORV running on sm_120: install CUDA 12.8 nvrtc + cublas into /root/cuda128, smoke test
PY=/root/miniconda3/bin/python
echo "=== install cuda 12.8 runtime pieces ==="
$PY -m pip install -q --no-deps --target /root/cuda128 \
    --index-url https://pypi.org/simple --no-cache-dir \
    "nvidia-cuda-nvrtc-cu12==12.8.93" "nvidia-cublas-cu12==12.8.4.1" 2>&1 | tail -3

echo "=== what got installed ==="
find /root/cuda128 -name 'libnvrtc.so*' -o -name 'libcublas.so*' -o -name 'libcublasLt.so*' 2>/dev/null | head -6

echo "=== does 12.8 nvrtc know compute_120? ==="
$PY - <<'PY'
import ctypes, glob
for p in sorted(glob.glob("/root/cuda128/nvidia/cuda_nvrtc/lib/libnvrtc.so*")):
    try:
        lib = ctypes.CDLL(p)
        ma, mi = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(ma), ctypes.byref(mi))
        has = b"compute_120" in open(p, "rb").read()
        print("  %-64s nvrtc %d.%d knows_120=%s" % (p, ma.value, mi.value, has))
    except Exception as e:
        print("  %-64s failed %s" % (p, type(e).__name__))
PY

echo "=== smoke with TGAI_CUDA_BIN pointing at 12.8 ==="
export TGAI_CUDA_BIN="/root/cuda128/nvidia/cuda_nvrtc/lib:/root/cuda128/nvidia/cublas/lib"
$PY - <<'PY'
import numpy as np
import tgai_go.train as T
try:
    T.set_default_device("cuda")
    x = T.ones((64, 64))
    y = x @ x
    print("  matmul OK, sum =", float(y.sum().to_numpy()))
    a = T.ones((256, 256)) * 2.0
    b = T.ones((256, 256)) * 3.0
    print("  mul OK, sum =", float((a * b).sum().to_numpy()))
except Exception:
    import traceback
    traceback.print_exc()
PY
