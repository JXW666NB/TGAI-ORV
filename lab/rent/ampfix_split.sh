#!/bin/bash
# Host/GPU split + cast-cost decomposition, 5090, 8 steps, 2 alternating rounds.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 8 --log-every 99 --no-final-save"

one() {  # $1=tag  $2=amp  $3=no_cast
  export PROBE_MODE=time
  export PROBE_NO_CAST=$3
  export PROBE_OUT=/root/p_$1.json
  $PY /root/probe_amp.py $COMMON --amp $2 >/root/p_$1.log 2>&1
  echo "  $1 -> $(grep -o '\"step_s_steady\": [0-9.]*' /root/p_$1.log)"
}

for r in 1 2; do
  echo "--- round $r ---"
  one off_r$r      off  0
  one gemm_r$r     gemm 0
  one gemmnc_r$r   gemm 1
done
echo DONE
