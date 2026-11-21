#!/bin/bash
# (1) Task-2 A/B: A=/root/orv_a (before fused cast) vs B=/root/orv_b (after), 4 rounds
# (2) TF32 vs off (2 rounds) -- the zero-cast route to tensor cores
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag pyroot amp
  PYTHONPATH=$2 $PY -m tgai_go.train.v3_pretrain $C --amp $3 >/root/t2_$1.log 2>&1
  echo "  $1 step=$(get step_s_steady /root/t2_$1.log) peak=$(get pool_peak_live_gib /root/t2_$1.log) loss=$(get loss_last /root/t2_$1.log)"
}

echo "=== (1) Task-2 A/B, 4 rounds ==="
for r in 1 2 3 4; do
  echo "--- round $r ---"
  run off_r$r  /root/orv_a/python off
  run A_r$r    /root/orv_a/python gemm
  run B_r$r    /root/orv_b/python gemm
done

echo "=== (2) TF32 (amp off + cuBLAS TF32) vs off, 2 rounds ==="
for r in 1 2; do
  PROBE_OUT=/root/tf32_$r.json PYTHONPATH=/root/orv_a/python \
    $PY /root/probe_tf32.py $C --amp off >/root/tf32_$r.log 2>&1
  echo "  tf32_r$r step=$(get step_s_steady /root/tf32_$r.log) peak=$(get pool_peak_live_gib /root/tf32_$r.log) loss=$(get loss_last /root/tf32_$r.log) begun=$(cat /root/tf32_$r.json 2>/dev/null)"
done
echo DONE
