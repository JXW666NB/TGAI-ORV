#!/bin/bash
# 閸忋劎鈻奸柌鍥ㄧ壉 GPU 閸掆晝鏁ら悳鍥风礉閸欘亜婀弰鎯х摠瀹歌弓绗傞柌蹇ョ礄>5 GiB閿涘娈戠粣妤€褰涢崘鍛埠鐠?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 40 --log-every 99 --no-final-save"

for mode in off gemm; do
  echo "===== amp=$mode ====="
  $PY -m tgai_go.train.v3_pretrain $COMMON --amp "$mode" > /tmp/run_$mode.log 2>&1 &
  PID=$!
  : > /tmp/util_$mode.csv
  while kill -0 $PID 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> /tmp/util_$mode.csv
    sleep 1
  done
  wait $PID 2>/dev/null
  /root/miniconda3/bin/python - <<PYEOF
rows = []
for line in open("/tmp/util_$mode.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 2:
        continue
    try:
        u = int(p[0].replace("%", "").strip())
        m = int(p[1].replace("MiB", "").strip())
    except ValueError:
        continue
    rows.append((u, m))
busy = [u for u, m in rows if m > 5000]
allu = [u for u, _ in rows]
print("  閹鍣伴弽?%d 濞嗏槄绱遍弰鎯х摠>5GiB 閻ㄥ嫮鐛ラ崣?%d 濞? % (len(rows), len(busy)))
if busy:
    print("  >>> 鐠侇厾绮岄張?GPU 閸掆晝鏁ら悳鍥风窗閸у洤鈧?%.1f%%  娑擃厺缍?%.0f%%  閺堚偓婢?%d%%"
          % (sum(busy) / len(busy), sorted(busy)[len(busy) // 2], max(busy)))
if allu:
    print("  閸忋劎鈻奸崸鍥р偓?%.1f%%閿涘牆鎯堥弸鍕紦閺堢喓鈹栭梻璇х礆" % (sum(allu) / len(allu)))
PYEOF
  grep -o '"step_s_steady": [0-9.]*' /tmp/run_$mode.log | tail -1
done
