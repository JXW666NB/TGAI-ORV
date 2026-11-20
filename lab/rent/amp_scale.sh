#!/bin/bash
# 4090锛欰MP 浠ｄ环鐨勬爣搴﹀疄楠?鈥斺€?鍙樺眰鏁帮紙鍙傛暟閲忥級鐪嬩唬浠锋槸鍚︽姣?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python

for L in 3 6 12; do
  for mode in gemm off; do
    OUT=$($PY -m tgai_go.train.v3_pretrain \
      --data /root/bench/data2048_v96000 --vocab 96000 \
      --seq 512 --d 1024 --layers $L --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 \
      --opt int8gpu --amp "$mode" --batch 1 --steps 6 --log-every 99 --no-final-save 2>&1)
    S=$(echo "$OUT" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')
    P=$(echo "$OUT" | grep -o '"params": [0-9]*' | grep -o '[0-9]*$')
    echo "layers=$L amp=$mode params=${P:-?} steady=${S:-FAIL}s"
  done
done
