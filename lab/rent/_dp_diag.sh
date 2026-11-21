#!/bin/bash
# 诊断：DP 每步多出来的时间到底花在哪（host vs nccl，逐相位）
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
export TG_DP_TIMING=1
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 8 --log-every 1 --no-final-save --amp off"
STAMP=$(date +%s)

echo "===== single 8 steps ====="
$PY -m tgai_go.train.v3_pretrain $C 2>&1 | grep -E "^step|^RESULT" | tail -10

echo "===== host 8 steps ====="
JOB=dh_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/d_h0.log 2>&1 &
P0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/d_h1.log 2>&1 &
P1=$!
wait $P0; wait $P1
grep -E "^step|^\[dp-phases\]" /root/bench/d_h0.log | tail -12

echo "===== nccl 8 steps ====="
JOB=dn_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/d_n0.log 2>&1 &
Q0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/d_n1.log 2>&1 &
Q1=$!
wait $Q0; wait $Q1
grep -E "^step|^\[dp-phases\]" /root/bench/d_n0.log | tail -12
