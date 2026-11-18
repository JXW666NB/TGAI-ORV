#!/usr/bin/env bash
# 同配置对比：PyTorch 与 ORV 在 8 层 / 12 层（24 层 PyTorch 会 OOM）
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"

run_pyt() {
  L=$1
  echo "### PyTorch ${L} 层 / seq512"
  $PY v3_pyt_train.py --d 1024 --layers "$L" --E 4 --vocab 96000 --d_ff 3328 \
      --n_heads 16 --top_k 2 --seq 512 --steps 5 2>&1 \
    | grep -E 'RESULT|OutOfMemory|Error' | tail -2
}

run_orv() {
  L=$1
  DFF=$2
  echo "### ORV ${L} 层 / seq512 / m8v16"
  $PY -m tgai_go.train.v3_pretrain --data "$ROOT/data512" --vocab 96000 --seq 512 \
      --d 1024 --layers "$L" --E 4 --n_q 16 --d_ff "$DFF" --top_k 2 --csa 4 \
      --opt int8gpu --steps 5 --log-every 1 --no-final-save 2>&1 \
    | grep -E '"step_s_steady"|显存不足' | tail -2
}

run_pyt 8
run_orv 8 3328
run_pyt 12
run_orv 12 3328
