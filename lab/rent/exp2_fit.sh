#!/bin/bash
# Experiment 2: real V3-shaped model capacity + step time on ONE 31.4 GiB card
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/exp2_fit.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

BASE="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--n_q 16 --d_ff 2816 --E 8 --top_k 2 --csa 4 --opt int8gpu --amp off \
--batch 1 --steps 3 --log-every 99 --no-final-save"

for L in 20 28 36; do
  out=$(CUDA_VISIBLE_DEVICES=0 $PY -m tgai_go.train.v3_pretrain $BASE --layers $L 2>&1)
  v=$(echo "$out" | get '"step_s_steady": [0-9.]*')
  p=$(echo "$out" | get '"params": [0-9]*')
  pk=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')
  if [ -n "$v" ]; then
    echo "  layers=$L params=${p} step=${v} peak=${pk} GiB" >> $LOG
  else
    echo "  layers=$L FAILED (OOM?)" >> $LOG
    echo "$out" | grep -iE "out of memory|OOM|CUDA error|Error" | head -2 >> $LOG
  fi
done
echo DONE >> $LOG
