#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
$PY /root/mb_gemm_shapes.py 2>&1 | tail -25
