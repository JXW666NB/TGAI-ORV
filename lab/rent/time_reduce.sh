#!/usr/bin/env bash
# 在真实训练步里给 all_reduce 计时：12 层、2 卡、2 步
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
JOB=timing$$
cd "$ROOT"
COMMON="--data $ROOT/data512 --vocab 96000 --seq 512 --d 1024 --layers 12 \
 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --steps 3 \
 --log-every 1 --no-final-save"

CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB TG_DP_TIMING=1 \
  $PY -m tgai_go.train.v3_pretrain $COMMON > t0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB TG_DP_TIMING=1 \
  $PY -m tgai_go.train.v3_pretrain $COMMON > t1.log 2>&1 &
P1=$!
wait $P0; wait $P1
echo "--- rank0（含归约计时）---"
grep -E '\[dp\]|step ' t0.log | tr -d '\r' | tail -8
rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
