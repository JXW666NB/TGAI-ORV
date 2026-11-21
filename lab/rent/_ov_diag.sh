#!/bin/bash
# 诊断：真实配置下 4 步，看 bwd / 每桶通信时长 / 重叠情况
#   A: 单卡 4 步（对照 bwd 时间）
#   B: 2-rank NCCL overlap ON（TG_DP_OV_DEBUG=1）
#   C: 2-rank NCCL overlap OFF（老路径对照）
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=180
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 4 --log-every 1 --no-final-save --amp off"
STAMP=$(date +%s)
rm -rf /root/bench/ovd_* /tmp/tgai_dp_* 2>/dev/null || true

echo "===== A: 单卡 ====="
$PY -m tgai_go.train.v3_pretrain $C --out /root/bench/ovd_s > /root/bench/ovd_s.log 2>&1
echo "single rc=$?"
grep -E "step |RESULT" /root/bench/ovd_s.log | head -6 | cut -c1-200

echo "===== B: dp overlap ON ====="
JOB=ovd_on_$STAMP
env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 \
    TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 TG_DP_OV_DEBUG=1 \
    $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovd_on > /root/bench/ovd_on.r0.log 2>&1 &
P0=$!
env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 \
    TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 TG_DP_OV_DEBUG=1 \
    $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovd_on > /root/bench/ovd_on.r1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "dp on rc0=$R0 rc1=$R1"
grep -E "dp-ov|step " /root/bench/ovd_on.r0.log | head -12 | cut -c1-400

echo "===== C: dp overlap OFF（老路径） ====="
JOB=ovd_off_$STAMP
env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=0 \
    TG_DP_BACKEND=nccl TG_DP_OVERLAP=0 TG_NCCL_LIB=$NCCLLIB \
    $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovd_off > /root/bench/ovd_off.r0.log 2>&1 &
Q0=$!
env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB CUDA_VISIBLE_DEVICES=1 \
    TG_DP_BACKEND=nccl TG_DP_OVERLAP=0 TG_NCCL_LIB=$NCCLLIB \
    $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovd_off > /root/bench/ovd_off.r1.log 2>&1 &
Q1=$!
wait $Q0; S0=$?
wait $Q1; S1=$?
echo "dp off rc0=$S0 rc1=$S1"
grep -E "step " /root/bench/ovd_off.r0.log | head -6 | cut -c1-200
echo "===== diag done ====="
