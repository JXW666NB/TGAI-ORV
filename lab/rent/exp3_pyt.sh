#!/bin/bash
# Experiment 3: PyTorch on the SAME real V3 config (1.82B) -- fp32 (expect OOM) vs bf16
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/exp3_pyt.log
: > $LOG
P="--d 1024 --layers 20 --E 8 --top_k 2 --d_ff 2816 --vocab 96000 --n_heads 16 \
--seq 512 --steps 3 --ckpt 0"

echo "=== PyT fp32 @1.82B (expect OOM: AdamW needs ~58 GiB) ===" >> $LOG
out=$(CUDA_VISIBLE_DEVICES=0 $PY v3_pyt_train.py $P --dtype fp32 2>&1)
echo "$out" | grep -iE '"step_s"|params|out of memory|CUDA error' | head -3 >> $LOG

echo "=== PyT bf16 @1.82B ===" >> $LOG
out=$(CUDA_VISIBLE_DEVICES=0 $PY v3_pyt_train.py $P --dtype bf16 2>&1)
echo "$out" | grep -iE '"step_s"|params|peak|out of memory|CUDA error' | head -5 >> $LOG
echo "$out" | tail -3 >> $LOG
echo DONE >> $LOG
