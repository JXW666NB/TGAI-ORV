#!/bin/bash
# 4090 阶段 A：overlay 自检 + 2-rank comm 基线 + 逐位正确性
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=180
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
STAMP=$(date +%s)
rm -rf /tmp/tgai_dp_* 2>/dev/null || true

echo "===== 0. overlay 自检 ====="
$PY -c "import tgai_go, tgai_go.train as T; from tgai_go.train import dp_overlap; print('pkg', tgai_go.__version__ if hasattr(tgai_go,'__version__') else '?'); print('hook', hasattr(T,'_set_bwd_overlap'), 'dp_overlap', dp_overlap.__name__)" 2>&1 | tail -3

echo "===== 1. 裸 allreduce（848.6M fp32 = 3.16 GiB），host 与 nccl 两条 ====="
arl() {  # $1=tag $2=backend $3...=env
  local tag=$1 be=$2; shift 2
  local job=ar4090_${tag}_$STAMP
  rm -rf /tmp/tgai_dp_* 2>/dev/null || true
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be TG_NCCL_LIB=$NCCLLIB ITERS=3 "$@" \
      $PY /root/bench/_ov_arlat.py > /root/bench/ar4090_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be TG_NCCL_LIB=$NCCLLIB ITERS=3 "$@" \
      $PY /root/bench/_ov_arlat.py > /root/bench/ar4090_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  echo "--- $tag ($be) rc=$r0/$r1"
  grep -hE "\[ar\].*(iter|median)" /root/bench/ar4090_$tag.r0.log | tail -4
}
arl host host
arl nccl nccl
arl nccl_ch8 nccl NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8

echo "===== 2. 逐位正确性（2 步，adamw） ====="
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 128 --d 256 --layers 2 --E 2 --n_q 4 --d_ff 512 --top_k 2 --csa 4 --opt adamw --batch 1 --steps 2 --log-every 1 --no-final-save --amp off --seed 42"
single() {  # $1=tag $2=ov $3=extra
  local tag=$1 ov=$2 extra=${3:-}
  TG_DP_OVERLAP=$ov $PY -m tgai_go.train.v3_pretrain $C $extra \
      --out /root/bench/c4090_$tag > /root/bench/c4090_$tag.log 2>&1
  echo "single $tag rc=$?"
}
dp() {  # $1=tag $2=backend $3=ov $4=extra
  local tag=$1 be=$2 ov=$3 extra=${4:-}
  local job=c4090_${tag}_$STAMP
  local envx=""
  if [ "$be" = "nccl" ]; then envx="TG_NCCL_LIB=$NCCLLIB"; fi
  rm -rf /root/bench/c4090_$tag
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/c4090_$tag $extra \
      > /root/bench/c4090_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/c4090_$tag $extra \
      > /root/bench/c4090_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  echo "dp $tag $be ov=$ov rc=$r0/$r1"
}
single w1_off 0
single w1_on 1
dp nccl_off nccl 0
dp nccl_on nccl 1
dp host_off host 0
dp host_on host 1
dp tied_nccl_off nccl 0 "--tie-embeddings"
dp tied_nccl_on nccl 1 "--tie-embeddings"

echo "===== 3. 对拍 ====="
ok=1
for pair in "w1_off w1_on" "nccl_off nccl_on" "host_off host_on" "tied_nccl_off tied_nccl_on"; do
  set -- $pair
  echo "--- $1 vs $2 ---"
  $PY /root/bench/_ov_cmp.py /root/bench/c4090_$1 /root/bench/c4090_$2 || ok=0
done
echo "ALL_CORRECTNESS_OK=$ok"
echo "===== phase A done ====="
