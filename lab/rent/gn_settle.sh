#!/bin/bash
# 定案：纯 DP（--zero 0）下 rank0 与 rank1 的 gnorm 到底是否逐位相同。
# 做法：只在**远端 overlay** 里把 gnorm 的打印精度从 %.3f 提到 %.9f（不改仓库文件），
#       跑 4 步、两个 rank，然后把两边的 gnorm 逐行对比。
PY=/root/miniconda3/bin/python
OV=/root/orv_src/python/tgai_go/train/v3_pretrain.py
LOG=/root/gn_settle.log
: > $LOG

echo "=== patch overlay precision ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
p = "/root/orv_src/python/tgai_go/train/v3_pretrain.py"
s = open(p, encoding="utf-8").read()
old = "gnorm=%9.3f"
new = "gnorm=%.9f"
if old in s:
    open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print("  patched %.3f -> %.9f")
else:
    print("  已打过或找不到，现状包含 gnorm=%.9f:", "gnorm=%.9f" in s)
PY

cd /root/bench || exit 1
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 4 --log-every 1 --no-final-save --dp 2"
JOB=gnsettle

echo "=== 2 ranks, --zero 0, 4 steps ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=$JOB \
  PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
  > /root/gn_r0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=$JOB \
  PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $COMMON \
  > /root/gn_r1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "  rc: rank0=$R0 rank1=$R1" >> $LOG
if [ "$R0" != "0" ] || [ "$R1" != "0" ]; then
  echo "  --- rank0 tail ---" >> $LOG; tail -4 /root/gn_r0.log >> $LOG 2>&1
  echo "  --- rank1 tail ---" >> $LOG; tail -4 /root/gn_r1.log >> $LOG 2>&1
fi

$PY - >> $LOG 2>&1 <<'PY'
import re
def g(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"step\s+(\d+)/\d+.*?gnorm=([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out
a = g("/root/gn_r0.log")
b = g("/root/gn_r1.log")
print("  rank0:", [(s, v) for s, v in a])
print("  rank1:", [(s, v) for s, v in b])
same = [x == y for x, y in zip(a, b)]
print("  逐位相同:", all(same) if same else "无法比较（步数不一致）")
for (s0, v0), (s1, v1) in zip(a, b):
    if v0 != v1:
        d = abs(float(v0) - float(v1))
        print("    step%-3d %s vs %s  |Δ|=%.3e 相对 %.2e" % (s0, v0, v1, d, d / float(v0)))
PY
echo DONE >> $LOG
