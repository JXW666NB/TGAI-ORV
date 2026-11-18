#!/usr/bin/env bash
# 参数对齐后再比：ORV 关掉共享专家（--no-shared）与 PyTorch 参考模型同参数量级
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"

for L in 8 12 16; do
  echo "### ORV ${L} 层 --no-shared / seq512 / m8v16"
  $PY -m tgai_go.train.v3_pretrain --data "$ROOT/data512" --vocab 96000 --seq 512 \
      --d 1024 --layers "$L" --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 \
      --opt int8gpu --no-shared --steps 5 --log-every 1 --no-final-save 2>&1 \
    | grep -oE '"(params|step_s_steady|tok_s_steady|pool_peak_live_gib)": [0-9.]+' | tr '\n' ' '
  echo
done

echo "### PyTorch 16 层 / seq512（看装不装得下）"
$PY v3_pyt_train.py --d 1024 --layers 16 --E 4 --vocab 96000 --d_ff 3328 \
    --n_heads 16 --top_k 2 --seq 512 --steps 5 2>&1 \
  | grep -E 'RESULT|OutOfMemory' | tail -1
