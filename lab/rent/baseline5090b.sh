#!/bin/bash
# 5090: ORV with the arch-fallback patch (PYTHONPATH=/root/p5090), 2 rounds + util sampling
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/p5090
cd /root/bench || exit 1
LOG=/root/b5090b.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--amp off --batch 1 --steps 20 --log-every 99 --no-final-save"

echo "=== ORV seq512 with patched package ===" >> $LOG
( for i in $(seq 1 200); do nvidia-smi --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null; sleep 0.4; done > /root/u5090b.csv ) &
SMP=$!
for r in 1 2; do
  out=$($PY -m tgai_go.train.v3_pretrain $COMMON 2>&1)
  v=$(echo "$out" | get '"step_s_steady": [0-9.]*')
  pk=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')
  echo "  orv r=$r step=${v:-FAIL} peak=${pk:-?}" >> $LOG
  if [ -z "$v" ]; then echo "$out" | tail -6 >> $LOG; fi
done
kill $SMP 2>/dev/null

echo "=== GPU util ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PYEOF'
rows = []
for line in open("/root/u5090b.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 2:
        continue
    try:
        rows.append((int(p[0]), int(p[1])))
    except ValueError:
        pass
busy = sorted(u for u, mm in rows if mm > 5000)
if busy:
    print("  采样 %d 次，训练窗口 %d 次 | 均值 %.1f%% 中位 %d%% 最大 %d%%"
          % (len(rows), len(busy), sum(busy)/len(busy), busy[len(busy)//2], max(busy)))
else:
    print("  采样 %d 次，没采到训练窗口（可能秒退）" % len(rows))
PYEOF
echo DONE >> $LOG
