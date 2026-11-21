#!/bin/bash
# TF32 (0 casts) vs fp16-cast vs the hybrid, 3 alternating rounds.
#   off            : baseline, PEDANTIC fp32
#   tf32           : amp off + cuBLAS TF32 math mode  -> tensor cores, 0 launches
#   f16            : amp gemm (fp16 casts), threshold 262144 (default)
#   tf32+f16big    : TF32 + fp16 only for GEMMs with m*n >= 4M  (cast bill amortized)
#   tf32+f16huge   : TF32 + fp16 only for m*n >= 16M
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag  amp  tf32(0/1)  minout
  if [ "$3" = "1" ]; then
    PROBE_OUT=/root/x_$1.json PYTHONPATH=/root/orv_b/python TG_FP16_MIN_OUTPUT=$4 \
      $PY /root/probe_tf32.py $C --amp $2 >/root/x_$1.log 2>&1
  else
    PYTHONPATH=/root/orv_b/python TG_FP16_MIN_OUTPUT=$4 \
      $PY -m tgai_go.train.v3_pretrain $C --amp $2 >/root/x_$1.log 2>&1
  fi
  echo "  $1 amp=$2 tf32=$3 minout=$4 step=$(get step_s_steady /root/x_$1.log) peak=$(get pool_peak_live_gib /root/x_$1.log) loss=$(get loss_last /root/x_$1.log)"
}

for r in 1 2 3; do
  echo "--- round $r ---"
  run off_r$r        off  0 262144
  run tf32_r$r       off  1 262144
  run f16_r$r        gemm 0 262144
  run tf32f16_r$r    gemm 1 4194304
  run tf32f16b_r$r   gemm 1 16777216
done
echo DONE
