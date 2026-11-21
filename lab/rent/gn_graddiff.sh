#!/bin/bash
# 逐参数对拍：普通归约 vs NCCL 重叠归约（各 1 步），找出哪些参数的梯度不同。
PY=/root/miniconda3/bin/python
LOG=/root/gn_graddiff.log
: > $LOG
cd /root/bench || exit 1
rm -f /root/d_plain* /root/d_ov*

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 1 --log-every 1 --no-final-save --dp 2"

run_pair () {
  tag=$1
  shift
  job=graddiff_$tag
  echo "=== $tag (env: $*) ===" >> $LOG
  env "$@" CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/gd_${tag}_r0.log 2>&1 &
  p0=$!
  env "$@" CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/gd_${tag}_r1.log 2>&1 &
  p1=$!
  wait $p0; r0=$?
  wait $p1; r1=$?
  echo "  rc: $r0/$r1" >> $LOG
}

run_pair plain TG_DP_DUMP_GRADS=/root/d_plain
run_pair ov TG_DP_OVERLAP=1 TG_DP_BACKEND=nccl TG_DP_DUMP_GRADS=/root/d_ov

$PY - >> $LOG 2>&1 <<'PY'
import glob
import json
import math


def load(prefix):
    files = sorted(glob.glob(prefix + ".rank0.step*.json"))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


a = load("/root/d_plain")
b = load("/root/d_ov")
if a is None or b is None:
    print("  !! 缺 dump 文件：plain=%s ov=%s" % (a is not None, b is not None))
    raise SystemExit(0)

ra, rb = a["rows"], b["rows"]
print("  参数个数: plain=%d ov=%d" % (len(ra), len(rb)))
sum_a = sum(r[2] for r in ra if isinstance(r[2], float))
sum_b = sum(r[2] for r in rb if isinstance(r[2], float))
print("  梯度平方和: plain=%.6f ov=%.6f  比值=%.6f"
      % (sum_a, sum_b, (sum_b / sum_a) if sum_a else float("nan")))
print("  gnorm: plain=%.6f ov=%.6f 比值=%.6f"
      % (math.sqrt(sum_a), math.sqrt(sum_b),
         (math.sqrt(sum_b) / math.sqrt(sum_a)) if sum_a else float("nan")))

bad = []
for x, y in zip(ra, rb):
    i, n, va = x[0], x[1], x[2]
    vy = y[2]
    if not isinstance(va, float) or not isinstance(vy, float):
        bad.append((i, n, va, vy, None))
        continue
    if va <= 0:
        continue
    r = vy / va
    if abs(r - 1.0) > 0.02:
        bad.append((i, n, va, vy, r))
print("  偏差 >2%% 的参数个数: %d / %d" % (len(bad), len(ra)))
for row in bad[:15]:
    i, n, va, vy, r = row
    print("    #%d numel=%-8d plain=%.6f ov=%.6f ratio=%s"
          % (i, n, va, vy, ("%.4f" % r) if r else "?"))
PY
echo DONE >> $LOG
