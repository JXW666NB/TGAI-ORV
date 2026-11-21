#!/bin/bash
# Step time vs number of issued f32->f16 casts (staircase), 2 rounds.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_a/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 14 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

for r in 1 2; do
  echo "--- round $r ---"
  for k in 0 2 4 6 8; do
    export PROBE_SKIP=$k
    export PROBE_OUT=/root/stair_${k}_$r.json
    $PY /root/probe_stair.py $C --amp gemm >/root/stair_${k}_$r.log 2>&1
    s=$(get step_s_steady /root/stair_${k}_$r.log)
    c=$(grep -o '"casts": [0-9]*' /root/stair_${k}_$r.json | grep -o '[0-9]*$')
    echo "  skip=$k step=${s:-FAIL} casts_total=${c:-?}"
  done
done
echo DONE
