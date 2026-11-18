#!/usr/bin/env bash
# 数学对齐测试：DP 平均 vs 单进程手算两样本平均；world=1 空操作；两 rank 一致性
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
JOB=math$$
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  $PY math_align.py > m0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  $PY math_align.py > m1.log 2>&1 &
P1=$!
wait $P0; wait $P1
echo "--- rank0 ---"; tail -5 m0.log | tr -d '\r'
echo "--- rank1 ---"; tail -5 m1.log | tr -d '\r'
rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
