#!/usr/bin/env bash
# 租机一键测量（Linux / AutoDL 类机器）。用法：
#     bash setup_measure.sh
# 全程只写 /root/orv_rent；任何一步失败都会停下并打印原因（不带着错误烧钱）。
set -euo pipefail

PY=/root/miniconda3/bin/python
[ -x "$PY" ] || PY=$(command -v python || command -v python3)
ROOT=/root/orv_rent
mkdir -p "$ROOT"
cd "$ROOT"

echo "===== 1/5 环境 ====="
"$PY" -V
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

echo "===== 2/5 安装 tgai-orv[cuda]==0.9.97 ====="
"$PY" -m pip install -q --upgrade pip 2>&1 | tail -2 || true
"$PY" -m pip install -q "tgai-orv[cuda]==0.9.97" 2>&1 | tail -8
"$PY" - <<'EOF'
import tgai_go
print("ORV", tgai_go.__version__)
EOF

echo "===== 3/5 生成 packed 分片（随机 token，仅测性能）====="
if [ ! -d "$ROOT/data512" ]; then
    "$PY" "$ROOT/make_packed.py" --out "$ROOT/data512" --seq 512 --vocab 96000 \
        --shards 16 --rows 64
else
    echo "已存在 $ROOT/data512"
fi

echo "===== 4/5 冒烟：2 层 / seq512 / 3 步（验证内核能在 sm_86 上编译）====="
"$PY" -m tgai_go.train.v3_pretrain --data "$ROOT/data512" --vocab 96000 --seq 512 \
    --d 1024 --layers 2 --E 4 --n_q 16 --d_ff 2816 --top_k 2 --csa 4 \
    --steps 3 --log-every 1 --no-final-save

echo "===== 5/5 1.5B / seq512 / 6 步（bf16 动量，约 21.5 GiB）====="
"$PY" -m tgai_go.train.v3_pretrain --data "$ROOT/data512" --vocab 96000 --seq 512 \
    --d 1024 --layers 24 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 \
    --opt fp16mom --opt-state bf16 --steps 6 --log-every 1 --no-final-save

echo "===== 完成 ====="
echo "对照基准（V100 32GB）：1.5B/seq512 = 0.446 s/步"
