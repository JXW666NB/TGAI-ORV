#!/bin/bash
# 1) 正确性（--opt adamw：没有 int8 量化，梯度平均应只在 fp32 舍入级别不同）
# 2) seq=512 的 3 轮 A/B（分块打包后）
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt adamw --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
STAMP=$(date +%s)

echo "##### CORRECTNESS (adamw) #####"
rm -rf /root/bench/a_s /root/bench/a_h /root/bench/a_n
$PY -m tgai_go.train.v3_pretrain $C --batch 2 --out /root/bench/a_s > /root/bench/a_s.log 2>&1
echo "single rc=$?"
JOB=ah_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/a_h > /root/bench/a_h0.log 2>&1 &
P0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=host \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/a_h > /root/bench/a_h1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "host rc0=$R0 rc1=$R1"
$PY /root/bench/_dp_correct.py /root/bench/a_s/train_log.jsonl \
    /root/bench/a_h/train_log.jsonl /root/bench/a_h/train_log.r1.jsonl

JOB=an_$STAMP
TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/a_n > /root/bench/a_n0.log 2>&1 &
Q0=$!
TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB \
  $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/a_n > /root/bench/a_n1.log 2>&1 &
Q1=$!
wait $Q0; A0=$?
wait $Q1; A1=$?
echo "nccl rc0=$A0 rc1=$A1"
$PY /root/bench/_dp_correct.py /root/bench/a_s/train_log.jsonl \
    /root/bench/a_n/train_log.jsonl /root/bench/a_n/train_log.r1.jsonl

echo "##### BENCH 512 (chunked pack) #####"
bash /root/bench/_dp_bench.sh 512
