#!/usr/bin/env bash
# 双卡通信基准：分别在 1/64 … 1× 的 1.5B 梯度上测 all_reduce 的耗时与带宽
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
JOB=bw$$
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  $PY bench_reduce.py > bw0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  $PY bench_reduce.py > bw1.log 2>&1 &
P1=$!
wait $P0; wait $P1

echo "--- rank0 ---"
cat bw0.log | tr -d '\r' | tail -12
echo "--- rank1（应完全一致）---"
cat bw1.log | tr -d '\r' | tail -12
rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
