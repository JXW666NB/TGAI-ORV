#!/bin/bash
# 5090 鎺㈤拡锛氱‖浠?/ 宸ュ叿閾?/ 瑁?ORV / 鍐掔儫锛圢VRTC 鑳藉惁涓?sm_120 缂栧唴鏍革級
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null \
  || nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Python ==="
PY=/root/miniconda3/bin/python
if [ ! -x "$PY" ]; then PY=$(command -v python3); fi
echo "  python: $PY"
$PY -V

echo "=== torch锛堝彲鑳芥病瑁咃級==="
$PY - <<'PY'
try:
    import torch
    print("  torch", torch.__version__, "| cuda", torch.version.cuda,
          "| avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  cap", torch.cuda.get_device_capability())
except Exception as e:
    print("  鏃?torch:", type(e).__name__, e)
PY

echo "=== 纾佺洏 ==="
df -h /root | tail -1

echo "=== 瑁?tgai-orv 1.0.12锛堜紭鍏堝畼鏂规簮锛岀粫寮€婊炲悗闀滃儚锛?=="
$PY -m pip install -q -U --index-url https://pypi.org/simple \
    --no-cache-dir "tgai-orv==1.0.12" 2>&1 | tail -3
$PY -c "import tgai_go; print('  tgai_go', tgai_go.__version__)" 2>&1 | tail -1

echo "=== 鍐掔儫锛氳繍琛屾椂缂栧唴鏍?+ 绠椾竴娆?matmul锛坰m_120 鍏抽敭楠岃瘉锛?=="
$PY - <<'PY'
import numpy as np
import tgai_go.train as T
try:
    T.set_default_device("cuda")
    x = T.ones((64, 64))
    y = x @ x
    s = float(y.sum().to_numpy())
    print("  matmul OK, sum =", s)
    a = T.ones((64, 64)) * 2.0
    b = T.ones((64, 64)) * 3.0
    print("  add/mul OK, sum =", float((a * b).sum().to_numpy()))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("  鍐掔儫澶辫触:", type(e).__name__, str(e)[:400])
PY
