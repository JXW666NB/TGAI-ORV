#!/bin/bash
# 5 rounds, quiet GPU: does `--amp tf32` really beat `--amp off` on this box?
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/tf32b.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

B="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"

for tree in /root/orv_src/python /root/miniconda3/lib/python3.12/site-packages; do
  echo "=== tree: $tree ===" >> $LOG
  for r in 1 2 3 4 5; do
    a=$(env PYTHONPATH=$tree $PY -m tgai_go.train.v3_pretrain $B --amp off  2>&1 | get '"step_s_steady": [0-9.]*')
    b=$(env PYTHONPATH=$tree $PY -m tgai_go.train.v3_pretrain $B --amp tf32 2>&1 | get '"step_s_steady": [0-9.]*')
    echo "  r=$r off=${a:-FAIL} tf32=${b:-FAIL}" >> $LOG
  done
done
echo DONE >> $LOG
