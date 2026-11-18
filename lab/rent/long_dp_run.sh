#!/usr/bin/env bash
# 1 小时双卡分布式稳定性实测（1.5B / seq512 / m8v16）。
# 目的：验证 DP 在长跑下不漂、不卡死、显存不涨；顺便拿到持续吞吐。
nohup bash -c '
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
JOB=longdp1h
cd "$ROOT"
COMMON="--data $ROOT/data512 --vocab 96000 --seq 512 --d 1024 --layers 24 \
 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --steps 7200 \
 --log-every 200 --no-final-save"
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  $PY -m tgai_go.train.v3_pretrain $COMMON > long_r0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  $PY -m tgai_go.train.v3_pretrain $COMMON > long_r1.log 2>&1 &
P1=$!
wait $P0 $P1
echo DONE > long_done.flag
' > /root/orv_rent/long_dp.out 2>&1 &
echo "1 小时 DP 长跑已在后台启动"
sleep 3
tail -3 /root/orv_rent/long_r0.log 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
