#!/bin/bash
# 第四轮：干净目录 + NCCL/host 全部对拍（含 tied）
set -u
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=180
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 128 --d 256 --layers 2 --E 2 --n_q 4 --d_ff 512 --top_k 2 --csa 4 --opt adamw --batch 1 --steps 2 --log-every 1 --no-final-save --amp off --seed 42"
STAMP=$(date +%s)
rm -rf /root/bench/ovsm_* 2>/dev/null || true
rm -rf /tmp/tgai_dp_* 2>/dev/null || true

single() {  # $1=name $2=overlap $3=extra
  local name=$1 ov=$2 extra=${3:-}
  TG_DP_OVERLAP=$ov $PY -m tgai_go.train.v3_pretrain $COMMON $extra \
      --out /root/bench/ovsm_$name > /root/bench/ovsm_$name.log 2>&1
  echo "single $name rc=$?"
}

dp() {  # $1=name $2=backend $3=overlap(0/1) $4=extra
  local name=$1 be=$2 ov=$3 extra=${4:-}
  local job=ovsm_${name}_$STAMP
  local envx=""
  if [ "$be" = "nccl" ]; then envx="TG_NCCL_LIB=$NCCLLIB"; fi
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $COMMON --dp 2 \
      --out /root/bench/ovsm_$name $extra > /root/bench/ovsm_$name.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be TG_DP_OVERLAP=$ov $envx \
      $PY -m tgai_go.train.v3_pretrain $COMMON --dp 2 \
      --out /root/bench/ovsm_$name $extra > /root/bench/ovsm_$name.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "dp $name rc0=$r0 rc1=$r1"
}

single w1_off 0
single w1_on 1
single w1tie_off 0 "--tie-embeddings"
single w1tie_on 1 "--tie-embeddings"
dp nccl_off nccl 0
dp nccl_on nccl 1
dp host_off host 0
dp host_on host 1
dp tied_nccl_off nccl 0 "--tie-embeddings"
dp tied_nccl_on nccl 1 "--tie-embeddings"
dp tied_host_off host 0 "--tie-embeddings"
dp tied_host_on host 1 "--tie-embeddings"

echo "===== CMP ===== "
for pair in "w1_off w1_on" "w1tie_off w1tie_on" "nccl_off nccl_on" "host_off host_on" "tied_nccl_off tied_nccl_on" "tied_host_off tied_host_on"; do
  set -- $pair
  echo "--- $1 vs $2 ---"
  $PY /root/bench/_ov_cmp.py /root/bench/ovsm_$1 /root/bench/ovsm_$2
done
echo "===== round4 done ====="
