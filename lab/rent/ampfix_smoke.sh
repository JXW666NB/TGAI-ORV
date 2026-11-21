#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
echo "=== amp off ==="
$PY -m tgai_go.train.v3_pretrain $COMMON --amp off 2>&1 | tail -14
echo "=== amp gemm ==="
$PY -m tgai_go.train.v3_pretrain $COMMON --amp gemm 2>&1 | tail -14
