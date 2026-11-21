#!/bin/bash
# (1) determinism base vs final (TG_DETERMINISTIC=1)
# (2) final 3-round A/B: off | gemm(base) | gemm(final)
# (3) GPU unit tests for the shadow cache (TG_RUN_CUDA_TESTS=1)
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C20="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
C4="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 4 --log-every 1 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

echo "=== (3) GPU unit tests ==="
export TG_RUN_CUDA_TESTS=1
export PYTHONPATH=/root/orv_src/python
if $PY -c "import pytest" 2>/dev/null; then
  cp /root/test_fp16_shadow_cache.py /root/test_fp16_shadow_cache.py.bak 2>/dev/null
  cd /root && $PY -m pytest /root/test_fp16_shadow_cache.py -q 2>&1 | tail -6
  cd /root/bench || exit 1
else
  echo "  pytest not installed on remote -> run the two checks directly"
  $PY /root/probe_shadow_gpu.py 2>&1 | tail -8
fi

echo "=== (1) determinism (TG_DETERMINISTIC=1) ==="
export TG_DETERMINISTIC=1
PYTHONPATH=/root/orv_base/python $PY -m tgai_go.train.v3_pretrain $C4 --amp gemm >/root/d_base.log 2>&1
PYTHONPATH=/root/orv_src/python  $PY -m tgai_go.train.v3_pretrain $C4 --amp gemm >/root/d_final.log 2>&1
unset TG_DETERMINISTIC
grep -E '^step' /root/d_base.log | sed 's/ [0-9.]*s tok.*//' > /root/d_base.txt
grep -E '^step' /root/d_final.log | sed 's/ [0-9.]*s tok.*//' > /root/d_final.txt
echo "--- base(first2)/final(first2) ---"; head -2 /root/d_base.txt; head -2 /root/d_final.txt
if diff -q /root/d_base.txt /root/d_final.txt >/dev/null; then
  echo "DET loss+gnorm: IDENTICAL"
else
  echo "DET: DIFFERS"; diff /root/d_base.txt /root/d_final.txt
fi

echo "=== (2) final A/B, 3 rounds ==="
for r in 1 2 3; do
  PYTHONPATH=/root/orv_src/python  $PY -m tgai_go.train.v3_pretrain $C20 --amp off  >/root/z_off_$r.log 2>&1
  PYTHONPATH=/root/orv_base/python $PY -m tgai_go.train.v3_pretrain $C20 --amp gemm >/root/z_base_$r.log 2>&1
  PYTHONPATH=/root/orv_src/python  $PY -m tgai_go.train.v3_pretrain $C20 --amp gemm >/root/z_fix_$r.log 2>&1
  echo "  r$r off=$(get step_s_steady /root/z_off_$r.log) base_gemm=$(get step_s_steady /root/z_base_$r.log) fix_gemm=$(get step_s_steady /root/z_fix_$r.log)"
done
echo DONE
