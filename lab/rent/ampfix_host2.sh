#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 8 --log-every 99 --no-final-save"
export PROBE_MODE=host
export PYTHONPATH=/root/orv_src/python
export PROBE_OUT=/root/ph_gemm.json
$PY /root/probe_host.py $COMMON --amp gemm >/root/ph_gemm.log 2>&1
echo "  gemm -> $(grep -o '\"step_s_steady\": [0-9.]*' /root/ph_gemm.log)"
export PYTHONPATH=/root/orv_base/python
export PROBE_OUT=/root/ph_off.json
$PY /root/probe_host.py $COMMON --amp off >/root/ph_off.log 2>&1
echo "  off -> $(grep -o '\"step_s_steady\": [0-9.]*' /root/ph_off.log)"
$PY /root/show_host.py gemm off
