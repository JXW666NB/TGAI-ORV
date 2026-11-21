#!/bin/bash
# 最终正确性对拍（2 步，adamw）：world=1 / world=2 + host/nccl + tied，全部逐位
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=180
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 128 --d 256 --layers 2 --E 2 --n_q 4 --d_ff 512 --top_k 2 --csa 4 --opt adamw --batch 1 --steps 2 --log-every 1 --no-final-save --amp off --seed 42"
STAMP=$(date +%s)
rm -rf /root/bench/ovf_* /tmp/tgai_dp_* 2>/dev/null || true

single() {  # $1=tag $2=ov $3=extra
  local tag=$1 ov=$2 extra=${3:-}
  TG_DP_OVERLAP=$ov $PY -m tgai_go.train.v3_pretrain $C $extra \
      --out /root/bench/ovf_$tag > /root/bench/ovf_$tag.log 2>&1
  echo "single $tag rc=$?"
}

dp() {  # $1=tag $2=backend $3=ov $4=extra
  local tag=$1 be=$2 ov=$3 extra=${4:-}
  local job=ovf_${tag}_$STAMP
  local envx=""
  if [ "$be" = "nccl" ]; then envx="TG_NCCL_LIB=$NCCLLIB"; fi
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovf_$tag $extra \
      > /root/bench/ovf_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out /root/bench/ovf_$tag $extra \
      > /root/bench/ovf_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "dp $tag $be ov=$ov rc=$r0/$r1"
}

echo "===== singles ====="
single w1_off 0
single w1_on 1
single w1tie_off 0 "--tie-embeddings"
single w1tie_on 1 "--tie-embeddings"
echo "===== dp ====="
dp nccl_off nccl 0
dp nccl_on nccl 1
dp host_off host 0
dp host_on host 1
dp tied_nccl_off nccl 0 "--tie-embeddings"
dp tied_nccl_on nccl 1 "--tie-embeddings"
dp tied_host_off host 0 "--tie-embeddings"
dp tied_host_on host 1 "--tie-embeddings"

echo "===== CMP ====="
ok=1
for pair in "w1_off w1_on" "w1tie_off w1tie_on" "nccl_off nccl_on" "host_off host_on" "tied_nccl_off tied_nccl_on" "tied_host_off tied_host_on"; do
  set -- $pair
  echo "--- $1 vs $2 ---"
  $PY /root/bench/_ov_cmp.py /root/bench/ovf_$1 /root/bench/ovf_$2 || ok=0
done
echo "ALL_CORRECTNESS_OK=$ok"
