#!/bin/bash
# 5090: PyTorch baseline only (ORV numbers already collected in run5090.log)
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/pyt5090.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }
P="--d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 --n_heads 16 --top_k 2 \
--seq 512 --steps 20 --ckpt 0"

echo "=== PyT noamp ===" >> $LOG
for r in 1 2; do
  out=$($PY v3_pyt_train.py $P 2>&1)
  v=$(echo "$out" | get '"step_s": [0-9.]*')
  echo "  noamp r=$r ${v:-FAIL}" >> $LOG
  [ -z "$v" ] && echo "$out" | tail -5 >> $LOG
done

echo "=== PyT amp ===" >> $LOG
for r in 1 2; do
  out=$($PY v3_pyt_train.py $P --amp 2>&1)
  v=$(echo "$out" | get '"step_s": [0-9.]*')
  echo "  amp r=$r ${v:-FAIL}" >> $LOG
  [ -z "$v" ] && echo "$out" | tail -5 >> $LOG
done
echo DONE >> $LOG
