#!/bin/bash
# PyT DDP 冒烟：先在**单进程**跑（WORLD_SIZE=1）暴露代码错误，再上 2 卡
PY=/root/miniconda3/bin/python
cd /root/bench || exit 1
echo "== single-process (world=1) smoke, 3 steps =="
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 $PY /root/bench/_v3_pyt_ddp.py \
    --d 1024 --layers 12 --E 4 --d_ff 3328 --vocab 96000 --n_heads 16 --top_k 2 \
    --seq 512 --steps 3 --dtype bf16 --ckpt 0 2>&1 | tail -8
echo "== 2-rank DDP smoke, 3 steps =="
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29778 \
    /root/bench/_v3_pyt_ddp.py \
    --d 1024 --layers 12 --E 4 --d_ff 3328 --vocab 96000 --n_heads 16 --top_k 2 \
    --seq 512 --steps 3 --dtype bf16 --ckpt 0 2>&1 | tail -12
