#!/bin/bash
# A/B: A=base (1.0.13), B=+O(1) f16 shadow invalidation. amp=gemm, 3 rounds interleaved.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"

one() {  # $1=tag $2=pyroot $3=amp
  export PYTHONPATH=$2
  $PY -m tgai_go.train.v3_pretrain $COMMON --amp $3 >/root/ab_$1.log 2>&1
  s=$(grep -o '"step_s_steady": [0-9.]*' /root/ab_$1.log | grep -o '[0-9.]*$')
  p=$(grep -o '"pool_peak_live_gib": [0-9.]*' /root/ab_$1.log | grep -o '[0-9.]*$')
  l=$(grep -o '"loss_last": [-0-9.e]*' /root/ab_$1.log | grep -o '[-0-9.e]*$')
  echo "  $1 amp=$3 step=${s:-FAIL} peak=${p:-?} loss_last=${l:-?}"
}

for r in 1 2 3; do
  echo "--- round $r ---"
  one A_gemm_r$r /root/orv_base/python gemm
  one B_gemm_r$r /root/orv_src/python  gemm
done
echo "--- off reference (both trees, 1 each) ---"
one A_off /root/orv_base/python off
one B_off /root/orv_src/python  off
echo DONE
