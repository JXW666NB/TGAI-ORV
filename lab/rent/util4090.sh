#!/bin/bash
# 鐢?nvidia-smi 閲囨牱 GPU 鍒╃敤鐜囷細鍒ゆ柇 ORV 鐨勬鏄?GPU-bound 杩樻槸 host-bound
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 30 --log-every 99 --no-final-save"

for mode in off gemm; do
  echo "===== amp=$mode ====="
  $PY -m tgai_go.train.v3_pretrain $COMMON --amp "$mode" > /tmp/run_$mode.log 2>&1 &
  PID=$!
  sleep 6            # 璺宠繃 build 闃舵
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1 > /tmp/util_$mode.csv &
  SPID=$!
  sleep 8
  kill $SPID 2>/dev/null
  wait $PID 2>/dev/null
  awk -F',' '{gsub(/[ %]/,"",$1); s+=$1; n++} END {if(n>0) printf "  GPU 鍒╃敤鐜囧潎鍊?%.1f%% (閲囨牱 %d 娆?\n", s/n, n}' /tmp/util_$mode.csv
  grep -o '"step_s_steady": [0-9.]*' /tmp/run_$mode.log | tail -1
done
