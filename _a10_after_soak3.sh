#!/bin/bash
# 长稳（L3 判据）结束后自动跑两段扫参：Qwen2.5 先、Qwen3 后。
# 为什么等长稳：长稳期间 GPU 被占（0.3 -> 8 s/step），抢卡会污染"学习率是否合适"的判断。
SOAK_PAT='[s]oak_test\.py'
echo "[waiter] $(date -Is) 等待长稳进程退出..."
while pgrep -f "$SOAK_PAT" > /dev/null 2>&1; do
  sleep 60
done
echo "[waiter] $(date -Is) 长稳已结束，开始扫参（seq_len=256，截断率 2.1%）"
cd /tmp/tgo || exit 1
export PYTHONPATH=/tmp/tgo/python
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader > /mnt/workspace/sweep_gpu_free.txt 2>&1

echo "[waiter] $(date -Is) 第一段：Qwen2.5-0.5B（lr x batch 共 6 组）"
python3 -u tools/sft_sweep.py \
  --safetensors /mnt/workspace/qwen25 \
  --data /mnt/workspace/sft_big.jsonl \
  --tokenizer /mnt/workspace/qwen25/tokenizer.json \
  --device cuda --steps 600 --holdout 8 --limit 2000 \
  --seq-len 256 --warmup 40 --clip 1.0 \
  --grid lr=5e-6,1e-5,2e-5 --batch 4,8 \
  --out /mnt/workspace/sweep_q25.csv \
  > /mnt/workspace/sweep_q25.log 2>&1
echo "[waiter] $(date -Is) 第一段结束 rc=$?"
tail -n 12 /mnt/workspace/sweep_q25.log

echo "[waiter] $(date -Is) 第二段：Qwen3-0.6B（用第一段较好的 lr 范围）"
python3 -u tools/sft_sweep.py \
  --safetensors /mnt/workspace/qwen3 \
  --data /mnt/workspace/sft_big.jsonl \
  --tokenizer /mnt/workspace/qwen3/tokenizer.json \
  --device cuda --steps 400 --holdout 8 --limit 2000 \
  --seq-len 256 --warmup 30 --clip 1.0 \
  --grid lr=1e-5,2e-5 --batch 4 \
  --out /mnt/workspace/sweep_q3.csv \
  > /mnt/workspace/sweep_q3.log 2>&1
echo "[waiter] $(date -Is) 第二段结束 rc=$?"
tail -n 12 /mnt/workspace/sweep_q3.log
echo "[waiter] $(date -Is) 全部完成"
