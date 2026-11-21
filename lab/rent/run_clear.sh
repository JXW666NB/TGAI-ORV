#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_a/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 3 --log-every 99 --no-final-save"
export PROBE_OUT=/root/clear.json
$PY /root/probe_clear.py $C --amp gemm >/root/clear.log 2>&1
$PY /root/show_clear.py /root/clear.json
