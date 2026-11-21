#!/bin/bash
# ZeRO-1/2 GPU 验收（远端执行；2 个进程 = 2 张卡）
#
# 前置（本地已上传）：
#   /root/zero_src/python/tgai_go/...   ← 干净的 HEAD(1.0.15) 源码 overlay
#   /root/zero_src/python/tgai_go/train/zero.py
#   /root/zero_verify.py                ← HEAD v3_pretrain + ZeRO snippet
#   /root/zero_cmp.py                   ← 逐位比对器
#
# 用法：bash /root/x_zero.sh smoke|bitwise|big|all
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/zero_src/python
export PYTHONIOENCODING=utf-8
export TGAI_DP_TIMEOUT=900
cd /root/bench || exit 1
DATA=/root/bench/data2048_v96000
STAMP=$(date +%s)
WHAT=${1:-all}

run2() {  # $1=tag $2=layers $3=steps $4=zero $5=extra
  local tag=$1 L=$2 S=$3 Z=$4 EXTRA=${5:-}
  local job=z${tag}_$STAMP
  rm -rf /root/bench/$tag
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
    $PY /root/zero_verify.py \
      --data $DATA --vocab 96000 --seq 512 --d 1024 --n_q 16 --d_ff 2816 \
      --E 8 --top_k 2 --csa 4 --opt int8gpu --batch 1 --amp off \
      --layers $L --steps $S --log-every 1 --no-final-save --zero $Z \
      --out /root/bench/$tag $EXTRA > /root/bench/$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
    $PY /root/zero_verify.py \
      --data $DATA --vocab 96000 --seq 512 --d 1024 --n_q 16 --d_ff 2816 \
      --E 8 --top_k 2 --csa 4 --opt int8gpu --batch 1 --amp off \
      --layers $L --steps $S --log-every 1 --no-final-save --zero $Z \
      --out /root/bench/$tag $EXTRA > /root/bench/$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "===== RUN $tag (layers=$L steps=$S zero=$Z) rc=$r0/$r1 ====="
  grep -E '^\[zero\]|^\[model\]|^\[int8gpu\]' /root/bench/$tag.r0.log | head -5
  grep -E '^RESULT ' /root/bench/$tag.r0.log | tail -1
  grep -E '^RESULT ' /root/bench/$tag.r1.log | tail -1
  return 0
}

smoke() {
  echo "########## SMOKE (2 rank, 2 层, 2 步, zero=2) ##########"
  local tag=zsmoke_$STAMP job=zsmoke_$STAMP
  rm -rf /root/bench/$tag
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
    $PY /root/zero_verify.py --data $DATA --vocab 96000 --seq 128 --d 256 \
      --layers 2 --E 2 --n_q 8 --d_ff 512 --top_k 1 --opt int8gpu --batch 1 \
      --steps 2 --log-every 1 --no-final-save --zero 2 --amp off \
      --out /root/bench/$tag > /root/bench/$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
    $PY /root/zero_verify.py --data $DATA --vocab 96000 --seq 128 --d 256 \
      --layers 2 --E 2 --n_q 8 --d_ff 512 --top_k 1 --opt int8gpu --batch 1 \
      --steps 2 --log-every 1 --no-final-save --zero 2 --amp off \
      --out /root/bench/$tag > /root/bench/$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "SMOKE rc=$r0/$r1"
  grep -E '^\[zero\]|step ' /root/bench/$tag.r0.log | head -6
  grep -E '^RESULT ' /root/bench/$tag.r0.log | tail -1
  grep -E 'Traceback|Error|error' /root/bench/$tag.r0.log | head -5
  tail -3 /root/bench/$tag.r1.log
}

bitwise() {
  # 1.82B：两边都装得下 ⇒ 用同一份数据/种子跑 20 步，三种模式的 loss & gnorm 必须逐位相同
  echo "########## BITWISE (1.82B / 20L / seq512 / int8gpu / 20 steps) ##########"
  run2 zA0_$STAMP 20 20 0
  run2 zA1_$STAMP 20 20 1
  run2 zA2_$STAMP 20 20 2
  echo "########## COMPARE ##########"
  $PY /root/zero_cmp.py \
    /root/bench/zA0_$STAMP/train_log.jsonl \
    /root/bench/zA1_$STAMP/train_log.jsonl \
    /root/bench/zA2_$STAMP/train_log.jsonl \
    --r1 /root/bench/zA0_$STAMP/train_log.r1.jsonl \
         /root/bench/zA1_$STAMP/train_log.r1.jsonl \
         /root/bench/zA2_$STAMP/train_log.r1.jsonl
}

big() {
  # 2.55B（28 层）：单卡 OOM，这里看 2 rank ZeRO 的峰值与步时
  echo "########## BIG (2.55B / 28L / seq512 / int8gpu / 4 steps) ##########"
  run2 zB1_$STAMP 28 4 1
  run2 zB2_$STAMP 28 4 2
}

case "$WHAT" in
  smoke) smoke ;;
  bitwise) bitwise ;;
  big) big ;;
  all) smoke; bitwise; big ;;
  *) echo "用法: bash /root/x_zero.sh smoke|bitwise|big|all"; exit 2 ;;
esac
echo "ZERO_RUN_DONE stamp=$STAMP"
