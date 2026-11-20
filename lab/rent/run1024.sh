#!/bin/bash
# seq1024锛歄RV/PyT 脳 鏈夋棤 ckpt锛屽悇 3 杞紝杩藉姞鍐?/root/m1024.log
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 1024 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--amp off --batch 1 --steps 20 --log-every 99 --no-final-save"

get() { grep -o "$1" | grep -o '[0-9.]*$'; }

for r in 1 2 3; do
  v=$(PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain $COMMON 2>&1 \
      | get '"step_s_steady": [0-9.]*')
  echo "orv_nockpt r=$r ${v:-OOM}" >> /root/m1024.log

  v=$(TG_GRAD_CKPT=1 PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain \
      $COMMON 2>&1 | get '"step_s_steady": [0-9.]*')
  echo "orv_ckpt r=$r ${v:-OOM}" >> /root/m1024.log

  v=$($PY v3_pyt_train.py --d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 \
      --n_heads 16 --top_k 2 --seq 1024 --steps 20 --ckpt 0 2>&1 \
      | get '"step_s": [0-9.]*')
  echo "pyt_nockpt r=$r ${v:-OOM}" >> /root/m1024.log

  v=$($PY v3_pyt_train.py --d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 \
      --n_heads 16 --top_k 2 --seq 1024 --steps 20 --ckpt 1 2>&1 \
      | get '"step_s": [0-9.]*')
  echo "pyt_ckpt r=$r ${v:-OOM}" >> /root/m1024.log
done
echo DONE >> /root/m1024.log
