#!/bin/bash
# 5090: 4 fresh rounds (steady step each) to separate JIT/warmup from steady state
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/p5090
cd /root/bench || exit 1
LOG=/root/b5090c.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--amp off --batch 1 --steps 20 --log-every 99 --no-final-save"

echo "=== 5090: 4 fresh rounds ===" >> $LOG
for r in 1 2 3 4; do
  out=$($PY -m tgai_go.train.v3_pretrain $COMMON 2>&1)
  v=$(echo "$out" | get '"step_s_steady": [0-9.]*')
  f=$(echo "$out" | get '"step_s_mean": [0-9.]*')
  echo "  r=$r steady=${v:-FAIL} mean=${f:-?}" >> $LOG
done

echo "=== 5090: launch count per step (probe_phases) ===" >> $LOG
$PY /root/probe_phases.py off 2>&1 | grep -E "稳态|平均每次|^   [0-9]/4" | tail -4 >> $LOG
echo DONE >> $LOG
