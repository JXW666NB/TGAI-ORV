#!/bin/bash
# Verify arch fallback on the 5090: NO TGAI_CUDA_BIN (system nvrtc 12.4), patched cuda.py
PY=/root/miniconda3/bin/python
echo "=== prepare patched package ==="
rm -rf /root/p5090
mkdir -p /root/p5090
cp -r "$($PY -c 'import site;print(site.getsitepackages()[0])')/tgai_go" /root/p5090/tgai_go
# 打补丁：用带 arch 降级链的 cuda.py 覆盖（否则上面的 cp -r 会把它冲掉）
if [ -f /root/p5090_new_cuda.py ]; then
  cp /root/p5090_new_cuda.py /root/p5090/tgai_go/train/cuda.py
  echo "  patched cuda.py applied"
fi
echo "  copied"

echo "=== smoke WITHOUT TGAI_CUDA_BIN (nvrtc 12.4 -> expect arch fallback) ==="
unset TGAI_CUDA_BIN
PYTHONPATH=/root/p5090 $PY - <<'PY'
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
    rt = T._cuda()
    print("  arch_used =", getattr(rt, "_arch_used", "?"),
          "| module_kind =", getattr(rt, "module_kind", "?"))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("  FAILED:", type(e).__name__, str(e)[:300])
PY
