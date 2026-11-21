#!/bin/bash
# seq2048 + grad-ckpt (GEMM-heavy regime): off vs gemm on the fixed tree.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/orv_src/python
export TG_GRAD_CKPT=1
cd /root/bench || exit 1
C2048="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 12 --log-every 99 --no-final-save"

one() {
  $PY -m tgai_go.train.v3_pretrain $C2048 --amp $1 >/root/s2k_$2.log 2>&1
  s=$(grep -o '"step_s_steady": [0-9.]*' /root/s2k_$2.log | grep -o '[0-9.]*$')
  p=$(grep -o '"pool_peak_live_gib": [0-9.]*' /root/s2k_$2.log | grep -o '[0-9.]*$')
  echo "  $2 step=${s:-FAIL} peak=${p:-?}"
}

for r in 1 2; do
  echo "--- round $r (seq2048+ckpt) ---"
  one off  off_r$r
  one gemm gemm_r$r
done
echo DONE
