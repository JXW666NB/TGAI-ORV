#!/bin/bash
# PyT 基线：单卡 vs 2-rank DDP，各 3 轮，seq512 + seq2048
PY=/root/miniconda3/bin/python
cd /root/bench || exit 1
BASE="--d 1024 --layers 12 --E 4 --d_ff 3328 --vocab 96000 --n_heads 16 --top_k 2 --steps 20 --dtype bf16 --ckpt 0"

for SEQ in 512 2048; do
  for R in 1 2 3; do
    echo "### pyt single seq=$SEQ r=$R"
    $PY /root/bench/v3_pyt_train.py $BASE --seq $SEQ > /root/bench/p_s_${SEQ}_$R.log 2>&1
    grep '^RESULT' /root/bench/p_s_${SEQ}_$R.log | sed 's/^RESULT //'
    echo "### pyt ddp seq=$SEQ r=$R"
    $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$((29800 + R)) \
        /root/bench/_v3_pyt_ddp.py $BASE --seq $SEQ > /root/bench/p_d_${SEQ}_$R.log 2>&1
    grep '^RESULT' /root/bench/p_d_${SEQ}_$R.log | sed 's/^RESULT //'
  done
done
echo "PYT BENCH DONE"
