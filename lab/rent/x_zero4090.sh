#!/bin/bash
# 2×RTX 4090 (24 GiB/卡) 上的 ZeRO 验收：
#   A) 逐位：848M/12L/seq512，zero 0/1/2 各 20 步，loss & gnorm 必须逐位相同
#   B) 装得下：同一个 1.82B（20L/E=8/d_ff=2816），DP(--zero 0) 应 OOM、ZeRO-2 应跑通
#
# 前置（本地已上传）：/root/zero_verify.py（HEAD v3_pretrain + ZeRO snippet）、
#   site-packages/tgai_go/train/zero.py、/root/zero_cmp.py、数据 /root/bench/data2048_v96000
# 用法：bash /root/x_zero4090.sh bitwise|mem <seq> <zero> [steps]|all
set -u
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
# ★ 必须用干净的 site-packages(1.0.15) + 我传进去的 zero.py，不能用别人正在改的源码 overlay
unset PYTHONPATH
cd /root/bench || exit 1
DATA=/root/bench/data2048_v96000
STAMP=$(date +%s)
COMMON="--data $DATA --vocab 96000 --d 1024 --n_q 16 --csa 4 --opt int8gpu --batch 1 --amp off --log-every 1 --no-final-save"

echo "[preflight] $PY"
$PY -c 'import tgai_go.train.zero as Z; print("[preflight] zero.py:", Z.__file__)'
$PY -c 'import tgai_go as G; print("[preflight] tgai_go:", G.__file__)'

run2() {  # tag layers E dff seq zero steps
  local tag=$1 L=$2 E=$3 DFF=$4 SEQ=$5 Z=$6 ST=$7
  local job=z${tag}_$STAMP
  rm -rf /root/bench/$tag
  for R in 0 1; do
    env TGAI_DP_WORLD=2 TGAI_DP_RANK=$R TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=$R \
      TGAI_DP_TIMEOUT=${DPTIMEOUT:-180} \
      $PY /root/zero_verify.py $COMMON --layers $L --E $E --d_ff $DFF --seq $SEQ \
        --top_k 2 --zero $Z --steps $ST --out /root/bench/$tag \
        > /root/bench/$tag.r$R.log 2>&1 &
  done
  wait
  echo "===== $tag layers=$L E=$E dff=$DFF seq=$SEQ zero=$Z steps=$ST ====="
  echo "--- [错误/OOM] r0 ---"
  grep -aE "显存不足|OutOfMemory|out of memory|CudaError|RuntimeError|Traceback" \
    /root/bench/$tag.r0.log | head -4
  echo "--- RESULT r0 ---"
  grep -aE '^RESULT ' /root/bench/$tag.r0.log | tail -1
  echo "--- RESULT r1 ---"
  grep -aE '^RESULT ' /root/bench/$tag.r1.log | tail -1
  echo "--- tail r0 ---"
  tail -3 /root/bench/$tag.r0.log
}

bitwise() {
  echo "############ A) 逐位：848M / 12L / E=4 / d_ff=3328 / seq512 / 20 步 ############"
  run2 zA0_$STAMP 12 4 3328 512 0 20
  run2 zA1_$STAMP 12 4 3328 512 1 20
  run2 zA2_$STAMP 12 4 3328 512 2 20
  echo "############ COMPARE (zero 0/1/2 逐位) ############"
  $PY /root/zero_cmp.py \
    /root/bench/zA0_$STAMP/train_log.jsonl \
    /root/bench/zA1_$STAMP/train_log.jsonl \
    /root/bench/zA2_$STAMP/train_log.jsonl \
    --r1 /root/bench/zA0_$STAMP/train_log.r1.jsonl \
         /root/bench/zA1_$STAMP/train_log.r1.jsonl \
         /root/bench/zA2_$STAMP/train_log.r1.jsonl
  echo "BITWISE_TAG=zA0_$STAMP"
}

mem() {  # mem <seq> <zero> [steps]
  local SEQ=$1 Z=$2 ST=${3:-2}
  echo "############ B) 1.82B / 20L / E=8 / d_ff=2816 / seq=$SEQ / zero=$Z / $ST 步 ############"
  DPTIMEOUT=150 run2 zM${SEQ}_${Z}_$STAMP 20 8 2816 $SEQ $Z $ST
}

case "${1:-all}" in
  bitwise) bitwise ;;
  mem) mem "$2" "$3" "${4:-2}" ;;
  all) bitwise; mem 1024 2 3; mem 1024 0 2 ;;
  *) echo "用法: bash /root/x_zero4090.sh bitwise|mem <seq> <zero> [steps]|all"; exit 2 ;;
esac
echo ZERO4090_DONE stamp=$STAMP
