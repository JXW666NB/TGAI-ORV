#!/bin/bash
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=120
cd /root/bench || exit 1
STAMP=$(date +%s)
echo "== stamp $STAMP =="
df -h /dev/shm | tail -1
echo "== nvidia compute apps =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "== host backend =="
JOB=arbh_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=host \
  $PY /root/bench/_dp_arbench.py --iters 5 > /root/bench/arb_h0.log 2>&1 &
P0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=host \
  $PY /root/bench/_dp_arbench.py --iters 5 > /root/bench/arb_h1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "rc0=$R0 rc1=$R1"
grep -E "^ARB|Traceback|Error" /root/bench/arb_h0.log | tail -3
echo "== nccl backend =="
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
JOB=arbn_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY /root/bench/_dp_arbench.py --iters 5 > /root/bench/arb_n0.log 2>&1 &
Q0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY /root/bench/_dp_arbench.py --iters 5 > /root/bench/arb_n1.log 2>&1 &
Q1=$!
wait $Q0; A0=$?
wait $Q1; A1=$?
echo "rc0=$A0 rc1=$A1"
grep -E "^ARB|Traceback|Error" /root/bench/arb_n0.log | tail -3
