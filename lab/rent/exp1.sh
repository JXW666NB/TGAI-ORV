#!/bin/bash
# Experiment 1 launcher: NCCL allreduce vs host-staged, 2 ranks on the dual 5090
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=8
cd /root || exit 1
echo "=== 2 ranks ==="
$PY -m torch.distributed.run --nproc_per_node=2 --standalone /root/exp1_comm.py 2>&1 | grep -vE "^\s*$" | tail -14
echo "=== 1 rank (host-staged baseline only) ==="
$PY /root/exp1_comm.py 2>&1 | tail -6
