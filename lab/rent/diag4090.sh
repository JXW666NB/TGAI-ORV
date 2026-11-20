#!/bin/bash
# 鎶?ORV 鍦?4090 涓婄殑涓€娆″畬鏁磋繍琛屾棩蹇楋紝閲嶇偣鐪嬫湁娌℃湁闈欓粯鍥為€€/鍛婅
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
export TG_DEBUG=1
LOG=/root/bench/_diag.log
/root/miniconda3/bin/python -m tgai_go.train.v3_pretrain \
  --data /root/bench/data2048_v96000 --vocab 96000 \
  --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 \
  --opt int8gpu --amp gemm --batch 1 --steps 2 --log-every 1 --no-final-save \
  > "$LOG" 2>&1
echo "=== 閫€鍑虹爜 $? ==="
echo "=== 鍚?fp16 / 鍥為€€ / 璀﹀憡 鐨勮 ==="
grep -inE 'fp16|鍥為€€|璀﹀憡|warn|fallback|sm_|arch' "$LOG" | head -20
echo "=== 鍓?12 琛?==="
head -12 "$LOG"
echo "=== 鏈熬 6 琛?==="
tail -6 "$LOG"
