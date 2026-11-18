#!/usr/bin/env bash
# 双卡数据并行冒烟：world=2，1.5B/seq512，3 步。目的是量吞吐，不是收敛。
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
JOB=dp2_$$
COMMON="--data $ROOT/data512 --vocab 96000 --seq 512 --d 1024 --layers 24 \
 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --steps 3 \
 --log-every 1 --no-final-save"

echo "=== 双卡 DP：world=2 job=$JOB ==="
cd "$ROOT"

# ★ ORV 没有选卡参数（set_device("cuda") 恒用 0 号卡），所以必须用
#   CUDA_VISIBLE_DEVICES 让每个 rank 只看见自己那张卡（驱动会把它的序号重映射成 0）。
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  $PY -m tgai_go.train.v3_pretrain $COMMON > r0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  $PY -m tgai_go.train.v3_pretrain $COMMON > r1.log 2>&1 &
P1=$!

wait $P0; RC0=$?
wait $P1; RC1=$?
echo "exit: rank0=$RC0 rank1=$RC1"

for r in 0 1; do
  echo "--- rank$r ---"
  grep -E 'loss=|Error|错误|Traceback|Timeout|显存' r$r.log | tail -5
done

# 清理本轮 rendezvous 残留，避免影响下次
rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
