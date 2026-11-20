#!/bin/bash
# 5090 baseline: gen data, ORV vs PyT at seq512, plus GPU utilization sampling during ORV
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
mkdir -p /root/bench
cd /root/bench || exit 1
LOG=/root/b5090.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

echo "=== torch ===" >> $LOG
$PY -c 'import torch;print("  torch",torch.__version__,"cuda",torch.version.cuda,"avail",torch.cuda.is_available(),"cap",torch.cuda.get_device_capability() if torch.cuda.is_available() else None)' >> $LOG 2>&1
HAS_TORCH=$($PY -c 'import torch;print(1)' 2>/dev/null || echo 0)
echo "  HAS_TORCH=$HAS_TORCH" >> $LOG

echo "=== gen data ===" >> $LOG
if [ ! -d /root/bench/data2048_v96000 ]; then
  $PY /root/gen_data.py >> $LOG 2>&1
fi
ls /root/bench/data2048_v96000 2>/dev/null | head -2 >> $LOG

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--amp off --batch 1 --steps 20 --log-every 99 --no-final-save"

# GPU util sampling for the first ORV run
( for i in $(seq 1 120); do nvidia-smi --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null; sleep 0.4; done > /root/u5090.csv ) &
SMP=$!

echo "=== ORV seq512 (2 rounds) ===" >> $LOG
for r in 1 2; do
  v=$($PY -m tgai_go.train.v3_pretrain $COMMON 2>&1 | get '"step_s_steady": [0-9.]*')
  echo "  orv r=$r $v" >> $LOG
done
kill $SMP 2>/dev/null

if [ "$HAS_TORCH" = "1" ]; then
  echo "=== PyT seq512 (2 rounds, no amp) ===" >> $LOG
  for r in 1 2; do
    v=$($PY v3_pyt_train.py --d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 \
        --n_heads 16 --top_k 2 --seq 512 --steps 20 --ckpt 0 2>&1 | get '"step_s": [0-9.]*')
    echo "  pyt r=$r $v" >> $LOG
  done
fi

echo "=== GPU util (samples with mem>5GiB) ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PYEOF'
rows = []
try:
    for line in open("/root/u5090.csv"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 2:
            continue
        try:
            u, mem = int(p[0]), int(p[1])
        except ValueError:
            continue
        rows.append((u, mem))
except FileNotFoundError:
    pass
busy = [u for u, mm in rows if mm > 5000]
if busy:
    busy.sort()
    print("  采样 %d 次，训练窗口 %d 次 | 均值 %.1f%% 中位 %d%% 最大 %d%%"
          % (len(rows), len(busy), sum(busy)/len(busy), busy[len(busy)//2], max(busy)))
else:
    print("  采样 %d 次，没采到训练窗口" % len(rows))
PYEOF
echo DONE >> $LOG
