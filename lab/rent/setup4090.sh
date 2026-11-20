#!/bin/bash
# 4090 ?????? ORV ??? + GPU ???
set -e
PY=/root/miniconda3/bin/python

echo "=== install ==="
$PY -m pip install -q -U 'tgai-orv[cuda]' 2>&1 | tail -5 || true

echo "=== version ==="
$PY -c "import tgai_go; print('pkg', tgai_go.__version__)"

echo "=== gpu smoke ==="
$PY - <<'PYEOF'
import numpy as np
import tgai_go.train as T

rt = None
for nm in ("cuda", "_cuda"):
    f = getattr(T, nm, None)
    if f is not None:
        try:
            rt = f()
            print("runtime via", nm, "->", type(rt).__name__)
            break
        except Exception as e:
            print(nm, "???:", type(e).__name__, str(e)[:160])

if rt is None:
    print("!! ?????? CUDA runtime")
    raise SystemExit(1)

a = T.tensor(np.random.rand(64, 64).astype(np.float32))
b = T.tensor(np.random.rand(64, 64).astype(np.float32))
c = a @ b
print("matmul ok, sum =", float(c.sum().item()) if hasattr(c.sum(), "item") else "n/a")

# ?????try:
    print("pool:", rt.pool_stats())
except Exception as e:
    print("pool_stats ?????", type(e).__name__)
PYEOF

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
