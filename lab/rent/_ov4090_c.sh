#!/bin/bash
# 4090 阶段 C：seq2048（先 2 步试内存，能跑才跑三轮；内存不够就如实记"跑不了"）
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
STAMP=$(date +%s)
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 2 --log-every 1 --no-final-save --amp off"
rm -rf /tmp/tgai_dp_* /root/bench/b2_* 2>/dev/null || true

echo "===== 内存试探：单卡 seq2048 2 步 ====="
$PY -m tgai_go.train.v3_pretrain $C --out /root/bench/b2_probe > /root/bench/b2_probe.log 2>&1
PRC=$?
echo "probe rc=$PRC"
tail -3 /root/bench/b2_probe.log
if [ "$PRC" != "0" ]; then
  echo "SEQ2048_UNRUNNABLE=1"
  exit 0
fi

C20="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
rs() {
  local out=/root/bench/b2_s$1
  rm -rf "$out"
  $PY -m tgai_go.train.v3_pretrain $C20 --out "$out" > /root/bench/b2_s$1.log 2>&1
  echo "single r$1 rc=$?"
}
rd() {
  local tag=o$2r$1
  local out=/root/bench/b2_$tag
  local job=b2_${tag}_$STAMP
  rm -rf "$out"
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=$2 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C20 --dp 2 --out "$out" \
      > /root/bench/b2_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=$2 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C20 --dp 2 --out "$out" \
      > /root/bench/b2_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  echo "dp ov=$2 r$1 rc=$r0/$r1"
}
for R in 1 2 3; do
  echo "--- round $R ---"
  rs $R
  rd $R 1
done
echo "SEQ2048_UNRUNNABLE=0"
echo "===== phase C done ====="
