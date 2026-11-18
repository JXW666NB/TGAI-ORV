#!/usr/bin/env bash
# 双卡对双卡：PyTorch DDP vs ORV DP（同形状、同 seq、各 rank 不同数据）
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"

for L in 12 16; do
  echo "##### ${L} 层 / seq512 #####"

  echo "--- PyTorch DDP（2 卡）---"
  $PY pyt_ddp.py --layers "$L" --seq 512 --world 2 --steps 5 2>&1 \
    | grep -E 'RESULT|OutOfMemory|Error' | tail -2

  echo "--- ORV DP（2 卡，m8v16）---"
  JOB=cmp$L
  COMMON="--data $ROOT/data512 --vocab 96000 --seq 512 --d 1024 --layers $L \
   --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --steps 5 \
   --log-every 1 --no-final-save"
  CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
    $PY -m tgai_go.train.v3_pretrain $COMMON > o0.log 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
    $PY -m tgai_go.train.v3_pretrain $COMMON > o1.log 2>&1 &
  P1=$!
  wait $P0; wait $P1
  grep -oE '"(step_s_steady|tok_s_steady|pool_peak_live_gib|params)": [0-9.]+' o0.log | tr '\n' ' '
  echo
done
