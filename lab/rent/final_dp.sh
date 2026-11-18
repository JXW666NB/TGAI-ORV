#!/usr/bin/env bash
# 最终 DP 步时（12/16 层），带归约计时；同时给出单卡对照
set -uo pipefail
PY=/root/miniconda3/bin/python
ROOT=/root/orv_rent
cd "$ROOT"

run_dp() {
  L=$1
  JOB=fin${L}$$
  COMMON="--data $ROOT/data512 --vocab 96000 --seq 512 --d 1024 --layers $L \
   --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --steps 4 \
   --log-every 1 --no-final-save"
  echo "### ORV DP ${L} 层 / 2 卡"
  CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB TG_DP_TIMING=1 \
    $PY -m tgai_go.train.v3_pretrain $COMMON > f0.log 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB TG_DP_TIMING=1 \
    $PY -m tgai_go.train.v3_pretrain $COMMON > f1.log 2>&1 &
  P1=$!
  wait $P0; wait $P1
  echo -n "  rank0: "; grep -oE '"(step_s_steady|tok_s_steady|pool_peak_live_gib)": [0-9.]+' f0.log | tr '\n' ' '; echo
  echo -n "  rank1: "; grep -oE '"(step_s_steady|tok_s_steady|pool_peak_live_gib)": [0-9.]+' f1.log | tr '\n' ' '; echo
  echo -n "  归约:  "; grep -oE 'reduce [0-9.]+ s' f0.log | tail -2 | tr '\n' ' '; echo
  echo -n "  gnorm: "; grep -oE 'gnorm= *[0-9.]+' f0.log | tail -3 | tr '\n' ' '; echo
  rm -rf "${TMPDIR:-/tmp}"/tgai_dp_$JOB 2>/dev/null || true
}

run_dp 12
run_dp 16
