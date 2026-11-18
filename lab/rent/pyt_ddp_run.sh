#!/usr/bin/env bash
# PyTorch DDP 双卡基线（修好 find_unused_parameters 之后）
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"
for L in 12 16; do
  echo "### PyTorch DDP ${L} 层 / seq512 / 2 卡"
  $PY pyt_ddp.py --layers "$L" --seq 512 --world 2 --steps 5 2>&1 \
    | grep -E 'RESULT|OutOfMemory|Error' | tail -2
done
