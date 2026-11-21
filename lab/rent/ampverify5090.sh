#!/bin/bash
# Independent verification of the O(1) f16-shadow invalidation fix on the 5090
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/ampverify.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

echo "=== package under test ===" >> $LOG
PYTHONPATH=/root/orv_src/python $PY -c "import tgai_go;print('  tgai_go',tgai_go.__version__)" >> $LOG 2>&1

B512="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"
B2048="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 10 --log-every 99 --no-final-save"

run() {  # $1=args $2=amp $3=extra env
  local out
  out=$(env $3 PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $1 --amp $2 2>&1)
  echo "$out" | get '"step_s_steady": [0-9.]*'
}

echo "=== seq512: alternating off / gemm (3 rounds) ===" >> $LOG
for r in 1 2 3; do
  a=$(run "$B512" off "")
  b=$(run "$B512" gemm "")
  echo "  r=$r off=${a:-FAIL} gemm=${b:-FAIL}" >> $LOG
done

echo "=== seq2048+ckpt: alternating off / gemm (2 rounds) ===" >> $LOG
for r in 1 2; do
  a=$(run "$B2048" off "TG_GRAD_CKPT=1")
  b=$(run "$B2048" gemm "TG_GRAD_CKPT=1")
  echo "  r=$r off=${a:-FAIL} gemm=${b:-FAIL}" >> $LOG
done

echo "=== determinism: two runs, TG_DETERMINISTIC=1, amp gemm ===" >> $LOG
for r in 1 2; do
  out=$(env TG_DETERMINISTIC=1 PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain \
        $B512 --amp gemm --steps 4 2>&1 | grep -E '^step' | awk '{print $3,$5}' | tr '\n' ' ')
  echo "  run$r: $out" >> $LOG
done
echo DONE >> $LOG
