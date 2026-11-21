#!/bin/bash
# Snapshot the current tree as A-baseline, then attribute every cast to a model call site.
set -e
rm -rf /root/orv_a
cp -a /root/orv_src /root/orv_a
md5sum /root/orv_a/python/tgai_go/train/cuda.py

PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_a/python
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 3 --log-every 99 --no-final-save"
export PROBE_MODE=sites
export PROBE_OUT=/root/sites_gemm.json
$PY /root/probe_cast_sites.py $COMMON --amp gemm >/root/sites_gemm.log 2>&1
echo "  gemm -> $(grep -o '"step_s_steady": [0-9.]*' /root/sites_gemm.log)"
export PYTHONPATH=/root/orv_a/python
$PY /root/show_sites.py /root/sites_gemm.json
