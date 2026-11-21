#!/bin/bash
# 4090 阶段 B：逐位正确性（快）+ seq512 三档基准（单卡 / 不重叠 / 重叠，3 轮）
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
STAMP=$(date +%s)
rm -rf /tmp/tgai_dp_* 2>/dev/null || true

C2="--data /root/bench/data2048_v96000 --vocab 96000 --seq 128 --d 256 --layers 2 --E 2 --n_q 4 --d_ff 512 --top_k 2 --csa 4 --opt adamw --batch 1 --steps 2 --log-every 1 --no-final-save --amp off --seed 42"

single2() {  # $1=tag $2=ov $3=extra
  local tag=$1 ov=$2 extra=${3:-}
  TG_DP_OVERLAP=$ov $PY -m tgai_go.train.v3_pretrain $C2 $extra \
      --out /root/bench/c4_$tag > /root/bench/c4_$tag.log 2>&1
  echo "single $tag rc=$?"
}
dp2() {  # $1=tag $2=backend $3=ov $4=extra
  local tag=$1 be=$2 ov=$3 extra=${4:-}
  local job=c4_${tag}_$STAMP
  local envx=""
  if [ "$be" = "nccl" ]; then envx="TG_NCCL_LIB=$NCCLLIB"; fi
  rm -rf /root/bench/c4_$tag
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C2 --dp 2 --out /root/bench/c4_$tag $extra \
      > /root/bench/c4_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C2 --dp 2 --out /root/bench/c4_$tag $extra \
      > /root/bench/c4_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  echo "dp $tag $be ov=$ov rc=$r0/$r1"
}

echo "########## 1. 逐位正确性（2 步） ##########"
single2 w1_off 0
single2 w1_on 1
dp2 nccl_off nccl 0
dp2 nccl_on nccl 1
dp2 host_off host 0
dp2 host_on host 1
dp2 tied_nccl_off nccl 0 "--tie-embeddings"
dp2 tied_nccl_on nccl 1 "--tie-embeddings"
echo "--- CMP ---"
ok=1
for pair in "w1_off w1_on" "nccl_off nccl_on" "host_off host_on" "tied_nccl_off tied_nccl_on"; do
  set -- $pair
  echo "--- $1 vs $2 ---"
  $PY /root/bench/_ov_cmp.py /root/bench/c4_$1 /root/bench/c4_$2 || ok=0
done
echo "ALL_CORRECTNESS_OK=$ok"

echo "########## 2. seq512 基准（20 步；single/off/on 交替） ##########"
C5="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
rs() {  # $1=round
  local out=/root/bench/b5_s$1
  rm -rf "$out"
  $PY -m tgai_go.train.v3_pretrain $C5 --out "$out" > /root/bench/b5_s$1.log 2>&1
  echo "single r$1 rc=$?"
}
rd() {  # $1=round $2=ov
  local tag=o$2r$1
  local out=/root/bench/b5_$tag
  local job=b5_${tag}_$STAMP
  rm -rf "$out"
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=$2 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C5 --dp 2 --out "$out" \
      > /root/bench/b5_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=$2 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C5 --dp 2 --out "$out" \
      > /root/bench/b5_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  echo "dp ov=$2 r$1 rc=$r0/$r1"
}
for R in 1 2 3; do
  echo "--- round $R ---"
  rs $R
  rd $R 0
  rd $R 1
done
echo "===== phase B done ====="
