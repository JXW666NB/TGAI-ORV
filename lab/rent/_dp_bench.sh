#!/bin/bash
# 主表 A/B：单卡 vs 2-rank host vs 2-rank nccl，交替 3 轮
# 用法：bash _dp_bench.sh <seq>
SEQ=${1:-512}
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/orv_src/python
export TGAI_DP_TIMEOUT=300
cd /root/bench || exit 1
NCCLLIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq $SEQ --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
STAMP=$(date +%s)

run_single() {
  local r=$1
  $PY -m tgai_go.train.v3_pretrain $C > /root/bench/b_s_${SEQ}_$r.log 2>&1
  $PY /root/bench/_dp_parse.py "s${SEQ}r$r" /root/bench/b_s_${SEQ}_$r.log
}

run_dp() {
  local r=$1 be=$2 lib=$3
  local job=${be}_${SEQ}_${STAMP}_$r
  local envx=""
  if [ "$be" = "nccl" ]; then envx="TG_NCCL_LIB=$lib"; fi
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=0 \
      TG_DP_BACKEND=$be $envx $PY -m tgai_go.train.v3_pretrain $C --dp 2 \
      > /root/bench/b_${be}_${SEQ}_$r.r0.log 2>&1 &
  local p0=$!
  env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job CUDA_VISIBLE_DEVICES=1 \
      TG_DP_BACKEND=$be $envx $PY -m tgai_go.train.v3_pretrain $C --dp 2 \
      > /root/bench/b_${be}_${SEQ}_$r.r1.log 2>&1 &
  local p1=$!
  wait $p0; local rc0=$?
  wait $p1; local rc1=$?
  if [ "$rc0" != "0" ] || [ "$rc1" != "0" ]; then
    echo "BENCH tag=${be}${SEQ}r$r ERROR rc0=$rc0 rc1=$rc1"
    tail -3 /root/bench/b_${be}_${SEQ}_$r.r0.log
    tail -3 /root/bench/b_${be}_${SEQ}_$r.r1.log
    return 1
  fi
  $PY /root/bench/_dp_parse.py "${be}${SEQ}r$r" /root/bench/b_${be}_${SEQ}_$r.r0.log
}

echo "===== seq=$SEQ stamp=$STAMP ====="
echo "--- round 1: single, host, nccl ---"
run_single 1
run_dp 1 host ""
run_dp 1 nccl "$NCCLLIB"
echo "--- round 2: nccl, host, single ---"
run_dp 2 nccl "$NCCLLIB"
run_dp 2 host ""
run_single 2
echo "--- round 3: single, host, nccl ---"
run_single 3
run_dp 3 host ""
run_dp 3 nccl "$NCCLLIB"
echo "===== done seq=$SEQ ====="
