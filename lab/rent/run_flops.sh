#!/bin/bash
# Where is the fp16 break-even?  off  vs  gemm gated at several FLOPs thresholds.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_a/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 14 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag  amp  flopsmin
  PROBE_FLOPSMIN=$3 PROBE_OUT=/root/fl_$1.json \
    $PY /root/probe_flops.py $C --amp $2 >/root/fl_$1.log 2>&1
  n=$(grep -o '"f16": [0-9]*' /root/fl_$1.json | grep -o '[0-9]*$')
  echo "  $1 amp=$2 fmin=$3 step=$(get step_s_steady /root/fl_$1.log) f16_gemms=${n:-?}"
}

for r in 1 2; do
  echo "--- round $r ---"
  run off_r$r   off  0
  run f0_r$r    gemm 0
  run f1_r$r    gemm 1000000000
  run f2_r$r    gemm 2000000000
  run f4_r$r    gemm 4000000000
  run f8_r$r    gemm 8000000000
done
echo DONE
