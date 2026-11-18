#!/usr/bin/env bash
# 跑差分测试（两个进程）
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
JOB=diff$$
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  $PY diff_chunks.py > d0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  $PY diff_chunks.py > d1.log 2>&1 &
P1=$!
wait $P0; wait $P1
echo "--- rank0 ---"; tail -6 d0.log | tr -d '\r'
echo "--- rank1 ---"; tail -6 d1.log | tr -d '\r'
rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
