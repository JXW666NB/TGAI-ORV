#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 8 --log-every 99 --no-final-save"
export PROBE_MODE=host
for amp in gemm off; do
  export PROBE_OUT=/root/ph_$amp.json
  $PY /root/probe_host.py $COMMON --amp $amp >/root/ph_$amp.log 2>&1
  echo "  $amp -> $(grep -o '"step_s_steady": [0-9.]*' /root/ph_$amp.log)"
done
$PY /root/show_host.py
