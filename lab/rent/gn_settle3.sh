#!/bin/bash
# 定案：关掉重叠后，两个 rank 的 gnorm 是否逐位相同？
# 对照组：同配置、同数据、同 4 步，唯一变量 TG_DP_OVERLAP=0 vs 默认。
PY=/root/miniconda3/bin/python
LOG=/root/gn_settle3.log
: > $LOG

cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 4 --log-every 1 --no-final-save --dp 2"

run_pair () {
  local tag=$1; shift
  local envs="$@"
  local job=gn3_$tag
  echo "=== $tag (env: $envs) ===" >> $LOG
  env $envs CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/gn3_${tag}_r0.log 2>&1 &
  local p0=$!
  env $envs CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$job \
    PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
    > /root/gn3_${tag}_r1.log 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "  rc: $r0/$r1" >> $LOG
}

#run_pair ovOFF "TG_DP_OVERLAP=0"
run_pair ovON "TG_DP_OVERLAP=1"

$PY - >> $LOG 2>&1 <<'PY'
import re


def scan(path):
    dbg, gn, ov = [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if "[dpdbg]" in line:
            dbg.append(line.strip())
        m = re.search(r"step\s+(\d+)/\d+.*?gnorm=([0-9.]+)", line)
        if m:
            gn.append(m.group(2))
        if "[dp-ov]" in line:
            ov.append(line.split("[dp-ov]")[1].strip()[:60])
    return dbg, gn, ov


for tag in ("ovOFF", "ovDEF"):
    a = scan("/root/gn3_%s_r0.log" % tag)
    b = scan("/root/gn3_%s_r1.log" % tag)
    print("== %s ==" % tag)
    print("  rank0 dbg=%d ov=%d gnorm=%s" % (len(a[0]), len(a[2]), a[1]))
    print("  rank1 dbg=%d ov=%d gnorm=%s" % (len(b[0]), len(b[2]), b[1]))
    if a[1] and b[1]:
        print("  gnorm 逐位相同:", a[1] == b[1])
        for x, y in zip(a[1], b[1]):
            if x != y:
                d = abs(float(x) - float(y))
                print("    %s vs %s |Δ|=%.3e 相对 %.2e" % (x, y, d, d / float(x)))
    if a[2]:
        print("  rank0 [dp-ov] 首行:", a[2][0])
PY
echo DONE >> $LOG
