#!/bin/bash
SOAK_PAT='[s]oak_test\.py'
echo "[waiter] $(date -Is) 等待长稳进程退出..."
while pgrep -f "$SOAK_PAT" > /dev/null 2>&1; do
  sleep 60
done
echo "[waiter] $(date -Is) 长稳已结束，开始扫参（seq_len=256，截断率 2.1%）"
cd /tmp/tgo || exit 1
export PYTHONPATH=/tmp/tgo/python
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader > /mnt/workspace/sweep_gpu_before.txt 2>&1
python3 -u tools/sft_sweep.py \
  --safetensors /mnt/workspace/qwen25 \
  --data /mnt/workspace/sft_big.jsonl \
  --tokenizer /mnt/workspace/qwen25/tokenizer.json \
  --device cuda --steps 600 --holdout 8 --limit 2000 \
  --seq-len 256 --warmup 40 --clip 1.0 \
  --grid lr=5e-6,1e-5,2e-5 --batch 4,8 \
  --out /mnt/workspace/sweep_q25.csv \
  > /mnt/workspace/sweep_q25.log 2>&1
echo "[waiter] $(date -Is) 扫参结束 rc=$?"
tail -n 20 /mnt/workspace/sweep_q25.log
