#!/bin/bash
# 裸 NCCL allreduce 吞吐：不同 NCCL_* 环境下的 2-rank 实测
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
STAMP=$(date +%s)

run_env() {  # $1=tag $2...=env
  local tag=$1; shift
  local job=arl_${tag}_$STAMP
  rm -rf /tmp/tgai_dp_* 2>/dev/null || true
  echo "===== CFG $tag : $* ====="
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB ITERS=3 "$@" \
      $PY /root/bench/_ov_arlat.py > /root/bench/arl_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=nccl TG_NCCL_LIB=$NCCLLIB ITERS=3 "$@" \
      $PY /root/bench/_ov_arlat.py > /root/bench/arl_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "rc=$r0/$r1"
  grep -hE "\[ar\].*(iter|median)" /root/bench/arl_$tag.r0.log
  grep -hE "\[ar\].*median" /root/bench/arl_$tag.r1.log
}

echo "===== arlat start ====="
run_env base
run_env ll NCCL_PROTO=LL
run_env simple NCCL_PROTO=Simple
run_env ring NCCL_ALGO=Ring
run_env tree NCCL_ALGO=Tree
run_env ch8 NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8
run_env ch4 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
run_env shm NCCL_SHM_DISABLE=1
echo "===== arlat done ====="
