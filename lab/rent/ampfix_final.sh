#!/bin/bash
# Final measurement set.
#  (1) seq512 careful A/B: off vs gemm on the fixed tree, 6 alternating rounds (clean, no probe)
#  (2) --amp-no-cache x2 (shows what the weight shadow cache is worth)
#  (3) determinism: base vs fixed, TG_DETERMINISTIC=1, 3 steps, per-step loss/gnorm
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C512="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
C512D="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 4 --log-every 1 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

echo "=== (1) seq512 off vs gemm, 6 rounds ==="
export PYTHONPATH=/root/orv_src/python
for r in 1 2 3 4 5 6; do
  $PY -m tgai_go.train.v3_pretrain $C512 --amp off  >/root/f_off_$r.log  2>&1
  $PY -m tgai_go.train.v3_pretrain $C512 --amp gemm >/root/f_gemm_$r.log 2>&1
  echo "  r$r off=$(get step_s_steady /root/f_off_$r.log) peak=$(get pool_peak_live_gib /root/f_off_$r.log) | gemm=$(get step_s_steady /root/f_gemm_$r.log) peak=$(get pool_peak_live_gib /root/f_gemm_$r.log)"
done

echo "=== (2) --amp-no-cache (2 rounds) ==="
for r in 1 2; do
  $PY -m tgai_go.train.v3_pretrain $C512 --amp gemm --amp-no-cache >/root/f_nc_$r.log 2>&1
  echo "  r$r nocache=$(get step_s_steady /root/f_nc_$r.log) peak=$(get pool_peak_live_gib /root/f_nc_$r.log)"
done

echo "=== (3) determinism base vs fixed (TG_DETERMINISTIC=1) ==="
export TG_DETERMINISTIC=1
export PYTHONPATH=/root/orv_base/python
$PY -m tgai_go.train.v3_pretrain $C512D --amp gemm >/root/f_det_base.log 2>&1
export PYTHONPATH=/root/orv_src/python
$PY -m tgai_go.train.v3_pretrain $C512D --amp gemm >/root/f_det_fix.log 2>&1
grep -E '^step' /root/f_det_base.log > /root/f_det_base.txt
grep -E '^step' /root/f_det_fix.log  > /root/f_det_fix.txt
echo "--- base ---"; cat /root/f_det_base.txt
echo "--- fixed ---"; cat /root/f_det_fix.txt
if diff -q /root/f_det_base.txt /root/f_det_fix.txt >/dev/null; then
  echo "DET: IDENTICAL"
else
  echo "DET: DIFFERS"; diff /root/f_det_base.txt /root/f_det_fix.txt
fi
echo DONE
