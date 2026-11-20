#!/bin/bash
# 4090 涓婄殑 AMP A/B锛氬紑 fp16 vs 鍏筹紝鍚?5 姝ュ彇绋虫€?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 5 --log-every 1 --no-final-save"

for mode in gemm off; do
  echo "===== amp=$mode ====="
  $PY -m tgai_go.train.v3_pretrain $COMMON --amp "$mode" 2>&1 \
    | grep -E '^step|RESULT' \
    | sed -E 's/.*(step +[0-9]+\/[0-9]+).* ([0-9.]+)s.*/\1 \2s/; s/^RESULT.*"step_s_steady": ([0-9.]+).*/steady=\1 s/'
done
