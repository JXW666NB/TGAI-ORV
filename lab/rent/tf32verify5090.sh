#!/bin/bash
# Independent check of `--amp tf32` on the 5090 + the tf32 x muon interaction
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/tf32verify.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

B="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"

run() { env PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $B --amp $1 2>&1; }

echo "=== seq512: alternating off / tf32 (3 rounds) ===" >> $LOG
for r in 1 2 3; do
  a=$(run off | get '"step_s_steady": [0-9.]*')
  b=$(run tf32 | get '"step_s_steady": [0-9.]*')
  echo "  r=$r off=${a:-FAIL} tf32=${b:-FAIL}" >> $LOG
done

echo "=== peak memory + loss trajectory (1 run each) ===" >> $LOG
o=$(run off); echo "  off  peak=$(echo "$o" | get '"pool_peak_live_gib": [0-9.]*') loss_last=$(echo "$o" | get '"loss_last": [0-9.]*')" >> $LOG
t=$(run tf32); echo "  tf32 peak=$(echo "$t" | get '"pool_peak_live_gib": [0-9.]*') loss_last=$(echo "$t" | get '"loss_last": [0-9.]*')" >> $LOG

echo "=== interaction: --opt muon --amp tf32 (4 steps, should not crash) ===" >> $LOG
out=$(env PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain \
      --data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
      --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt muon \
      --amp tf32 --batch 1 --steps 4 --log-every 99 --no-final-save 2>&1)
echo "  muon+tf32 step=$(echo "$out" | get '"step_s_steady": [0-9.]*')  \
peak=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')" >> $LOG
echo "$out" | grep -iE 'error|traceback|assert' | head -3 >> $LOG
echo DONE >> $LOG
