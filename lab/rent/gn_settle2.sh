#!/bin/bash
# 定案 A/B：--dp 2 下 DP 是否真的启用？启用后两 rank 的 gnorm 是否逐位相同？
PY=/root/miniconda3/bin/python
OV=/root/orv_src/python/tgai_go/train/v3_pretrain.py
LOG=/root/gn_settle2.log
: > $LOG

echo "=== patch overlay: 高精度 gnorm + [dpdbg] ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
import re

p = "/root/orv_src/python/tgai_go/train/v3_pretrain.py"
s = open(p, encoding="utf-8").read()
n = 0
if "gnorm=%9.3f" in s:
    s = s.replace("gnorm=%9.3f", "gnorm=%.9f", 1)
    n += 1
if "[dpdbg]" not in s:
    def add(m):
        indent, call = m.group(1), m.group(2)
        return (indent + call + "\n" + indent +
                'print("[dpdbg] world=%d rank=%d reduced=1 job=%s" % '
                '(_dp.world, _dp.rank, getattr(_dp, "job", "?")), flush=True)')
    s2, k = re.subn(r"(?m)^([ \t]+)(_dp\.all_reduce_grads\(params\))", add, s)
    if k:
        s = s2
        n += k
open(p, "w", encoding="utf-8").write(s)
print("  改为高精度/dpdbg 的处数:", n)
print("  含 [dpdbg]:", "[dpdbg]" in s, "| 含 gnorm=%.9f:", "gnorm=%.9f" in s)
PY

cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 4 --log-every 1 --no-final-save --dp 2"
JOB=gnsettle2

echo "=== 2 ranks, --dp 2, 4 steps ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
  > /root/gn2_r0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
  > /root/gn2_r1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "  rc: rank0=$R0 rank1=$R1" >> $LOG
if [ "$R0" != "0" ] || [ "$R1" != "0" ]; then
  echo "  --- rank0 tail ---" >> $LOG; tail -6 /root/gn2_r0.log >> $LOG 2>&1
  echo "  --- rank1 tail ---" >> $LOG; tail -6 /root/gn2_r1.log >> $LOG 2>&1
fi

$PY - >> $LOG 2>&1 <<'PY'
import re

def scan(path):
    dbg, gn = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"\[dpdbg\] (.*)", line)
        if m:
            dbg.append(m.group(1).strip())
        m = re.search(r"step\s+(\d+)/\d+.*?gnorm=([0-9.]+)", line)
        if m:
            gn.append((int(m.group(1)), m.group(2)))
    return dbg, gn

for tag, path in (("rank0", "/root/gn2_r0.log"), ("rank1", "/root/gn2_r1.log")):
    dbg, gn = scan(path)
    print("  %s dpdbg(首行): %s" % (tag, dbg[0] if dbg else "（没有打印！归约未调用）"))
    print("  %s dpdbg 条数=%d  gnorm=%s" % (tag, len(dbg), gn))

a = scan("/root/gn2_r0.log")[1]
b = scan("/root/gn2_r1.log")[1]
if a and b:
    print("  逐位相同:", all(x == y for x, y in zip(a, b)))
PY
echo DONE >> $LOG
