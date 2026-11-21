#!/bin/bash
# Final A/B (5 alternating rounds): off | gemm (fp16 casts) | tf32 (0 casts)
# plus seq2048+ckpt non-regression and a determinism check for the fused-cast change.
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C512="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
C2048="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 12 --log-every 99 --no-final-save"
C4="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 4 --log-every 1 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag pyroot amp seqblocks
  PYTHONPATH=$2 $PY -m tgai_go.train.v3_pretrain $3 --amp $4 >/root/g_$1.log 2>&1
  echo "  $1 amp=$4 step=$(get step_s_steady /root/g_$1.log) peak=$(get pool_peak_live_gib /root/g_$1.log) loss=$(get loss_last /root/g_$1.log)"
}

echo "=== seq512, 5 rounds ==="
for r in 1 2 3 4 5; do
  echo "--- round $r ---"
  run off_r$r  /root/orv_c/python "$C512" off
  run f16_r$r  /root/orv_c/python "$C512" gemm
  run tf32_r$r /root/orv_c/python "$C512" tf32
done

echo "=== seq2048 + ckpt, 2 rounds ==="
export TG_GRAD_CKPT=1
for r in 1 2; do
  echo "--- round $r ---"
  run o2048_r$r  /root/orv_c/python "$C2048" off
  run f2048_r$r  /root/orv_c/python "$C2048" gemm
  run t2048_r$r  /root/orv_c/python "$C2048" tf32
done
unset TG_GRAD_CKPT

echo "=== determinism: fused cast must be bit-identical ==="
export TG_DETERMINISTIC=1
PYTHONPATH=/root/orv_a/python $PY -m tgai_go.train.v3_pretrain $C4 --amp gemm >/root/g_det_a.log 2>&1
PYTHONPATH=/root/orv_c/python $PY -m tgai_go.train.v3_pretrain $C4 --amp gemm >/root/g_det_c.log 2>&1
unset TG_DETERMINISTIC
grep -E '^step' /root/g_det_a.log | sed 's/ [0-9.]*s tok.*//' > /root/g_det_a.txt
grep -E '^step' /root/g_det_c.log | sed 's/ [0-9.]*s tok.*//' > /root/g_det_c.txt
echo "--- A(before fused cast) ---"; cat /root/g_det_a.txt
echo "--- C(after) ---"; cat /root/g_det_c.txt
if diff -q /root/g_det_a.txt /root/g_det_c.txt >/dev/null; then
  echo "DET loss+gnorm: IDENTICAL"
else
  echo "DET: DIFFERS"; diff /root/g_det_a.txt /root/g_det_c.txt
fi
echo DONE
