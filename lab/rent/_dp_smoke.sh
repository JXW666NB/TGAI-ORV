#!/bin/bash
# DP correctness smoke: world=1(batch2) vs world=2(host) vs world=2(nccl)
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 256 --layers 2 --E 2 --n_q 4 --d_ff 512 --top_k 1 --opt adamw --batch 1 --steps 3 --log-every 1 --no-final-save --amp off"

echo "===== A: world=1 batch=2 ====="
$PY -m tgai_go.train.v3_pretrain $C --batch 2 2>&1 | grep -E "^\[dp|^\[model|^\[data\]|^step|^losses|Traceback|Error" | head -20
echo "rc=$?"

echo "===== B: world=2 host (batch=1 each) ====="
JOB=sm1
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/s_r0.log 2>&1 &
P0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/s_r1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "rc0=$R0 rc1=$R1"
grep -E "^\[dp|^step|^losses|Traceback|Error" /root/bench/s_r0.log | head -12
echo "--- rank1 (head) ---"
grep -E "^\[dp|^\[r1\]|Traceback|Error" /root/bench/s_r1.log | head -8

echo "===== C: world=2 nccl ====="
JOB=sm2
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/n_r0.log 2>&1 &
Q0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 > /root/bench/n_r1.log 2>&1 &
Q1=$!
wait $Q0; A0=$?
wait $Q1; A1=$?
echo "rc0=$A0 rc1=$A1"
grep -E "^\[dp|^step|^losses|Traceback|Error" /root/bench/n_r0.log | head -12

echo "===== D: nccl selftest ====="
JOB=sm3
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.dist_nccl --selftest > /root/bench/t_r0.log 2>&1 &
S0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.dist_nccl --selftest > /root/bench/t_r1.log 2>&1 &
S1=$!
wait $S0; T0=$?
wait $S1; T1=$?
echo "rc0=$T0 rc1=$T1"
cat /root/bench/t_r0.log | tail -4
cat /root/bench/t_r1.log | tail -4
