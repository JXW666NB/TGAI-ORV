#!/bin/bash
# 最终基准：交替 3 轮 × 20 步，single vs 2-rank NCCL（重叠，默认通道 8、桶 256MB、2 槽）
# 用法：bash _ov_final.sh <seq>
set -u
SEQ=${1:?usage: _ov_final.sh <seq>}
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq $SEQ --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
STAMP=$(date +%s)
rm -rf /tmp/tgai_dp_* 2>/dev/null || true

rs() {  # $1 = round
  local out=/root/bench/of_s${SEQ}r$1
  rm -rf "$out"
  $PY -m tgai_go.train.v3_pretrain $C --out "$out" > /root/bench/of_s${SEQ}r$1.log 2>&1
  echo "single r$1 rc=$?"
}

rd() {  # $1 = round
  local tag=d${SEQ}r$1
  local out=/root/bench/of_$tag
  local job=of_${tag}_$STAMP
  rm -rf "$out"
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out "$out" \
      > /root/bench/of_$tag.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 TG_NCCL_LIB=$NCCLLIB TG_DP_TIMING=1 \
      $PY -m tgai_go.train.v3_pretrain $C --dp 2 --out "$out" \
      > /root/bench/of_$tag.r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "dp r$1 rc0=$r0 rc1=$r1"
}

echo "===== final seq=$SEQ stamp=$STAMP ====="
echo "--- round 1: single, dp ---"
rs 1
rd 1
echo "--- round 2: dp, single ---"
rd 2
rs 2
echo "--- round 3: single, dp ---"
rs 3
rd 3
echo "===== final seq=$SEQ done ====="
