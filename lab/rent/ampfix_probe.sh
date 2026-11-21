#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 6 --log-every 1 --no-final-save"
export PROBE_MODE=count
export PROBE_OUT=/root/probe_gemm.json
echo "=== count: amp gemm ==="
$PY /root/probe_amp.py $COMMON --amp gemm 2>&1 | grep -E "^step|probe|RESULT" | tail -12
export PROBE_OUT=/root/probe_off.json
echo "=== count: amp off ==="
$PY /root/probe_amp.py $COMMON --amp off 2>&1 | grep -E "^step|probe|RESULT" | tail -12
ls -la /root/probe_gemm.json /root/probe_off.json
