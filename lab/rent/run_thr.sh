#!/bin/bash
# Sweep the fp16 GEMM output-size threshold (TG_FP16_MIN_OUTPUT) at seq512.
# Rationale: the step is host-bound, so a cast that buys GPU time we cannot use is
# pure loss -- the optimum threshold may be far above the current 262144.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_a/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 14 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag, amp, threshold
  TG_FP16_MIN_OUTPUT=$3 $PY -m tgai_go.train.v3_pretrain $C --amp $2 >/root/th_$1.log 2>&1
  echo "  $1 amp=$2 thr=$3 step=$(get step_s_steady /root/th_$1.log) peak=$(get pool_peak_live_gib /root/th_$1.log)"
}

for r in 1 2; do
  echo "--- round $r ---"
  run off_r$r   off  262144
  run t26_r$r   gemm 262144
  run t20_r$r   gemm 1048576
  run t22_r$r   gemm 4194304
  run t23_r$r   gemm 8388608
  run t24_r$r   gemm 16777216
  run t26b_r$r  gemm 67108864
done
echo DONE
