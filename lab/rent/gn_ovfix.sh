#!/bin/bash
# 验证：host+重叠 必须 WARN 退回；nccl+重叠 必须真的通信（comm>0）
PY=/root/miniconda3/bin/python
LOG=/root/gn_ovfix.log
: > $LOG
cd /root/bench || exit 1

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 4 --log-every 1 --no-final-save --dp 2"

run_pair () {
  tag=$1
  shift
  job=ovfix_$tag
  echo "=== $tag (env: $*) ===" >> $LOG
  env "$@" CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/ovfix_${tag}_r0.log 2>&1 &
  p0=$!
  env "$@" CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/ovfix_${tag}_r1.log 2>&1 &
  p1=$!
  wait $p0; r0=$?
  wait $p1; r1=$?
  echo "  rc: $r0/$r1" >> $LOG
}

run_pair HOST TG_DP_OVERLAP=1 TG_DP_BACKEND=host
run_pair NCCL TG_DP_OVERLAP=1 TG_DP_BACKEND=nccl

$PY - >> $LOG 2>&1 <<'PY'
import re


def scan(path):
    gnorm, ov, warn = [], None, []
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.search(r"gnorm=\s*([0-9.]+)", line)
            if m:
                gnorm.append(m.group(1))
            if "[dp-ov]" in line and ov is None:
                ov = line.split("[dp-ov]")[1].strip()[:70]
            if "WARN" in line:
                warn.append(line.strip()[:110])
    except OSError:
        pass
    return gnorm, ov, warn


for tag in ("HOST", "NCCL"):
    a = scan("/root/ovfix_%s_r0.log" % tag)
    b = scan("/root/ovfix_%s_r1.log" % tag)
    print("== %s ==" % tag)
    print("  rank0 gnorm:", a[0])
    print("  rank1 gnorm:", b[0])
    print("  逐位相同:", a[0] == b[0] if (a[0] and b[0]) else "（没取到）")
    print("  [dp-ov]:", a[1])
    if a[2]:
        print("  WARN:", a[2][0])
PY
echo DONE >> $LOG
