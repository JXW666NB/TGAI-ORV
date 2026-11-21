#!/bin/bash
# Data for the "should the fp16 weight-shadow cache default to on?" question.
# 4 alternating rounds: off | gemm(cache on) | gemm(--amp-no-cache)
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_c/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag amp extra
  $PY -m tgai_go.train.v3_pretrain $C --amp $2 $3 >/root/w_$1.log 2>&1
  echo "  $1 amp=$2 $3 step=$(get step_s_steady /root/w_$1.log) peak=$(get pool_peak_live_gib /root/w_$1.log)"
}

for r in 1 2 3 4; do
  echo "--- round $r ---"
  run off_r$r   off  ""
  run on_r$r    gemm ""
  run noc_r$r   gemm "--amp-no-cache"
done
echo DONE
