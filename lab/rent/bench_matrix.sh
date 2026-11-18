#!/usr/bin/env bash
# 基准矩阵：ORV vs PyTorch，同一形状与 seq，真实训练步（4 步取稳态）。
# 每一项都只打印一行，便于汇总成 README 表格。
# 失败（多为 OOM）不会中断整体，会打印 FAIL。
set -uo pipefail
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"
STEPS=4

# ---------- ORV ----------
orv() {   # 名字 层数 d_ff E top_k seq 额外参数
  local name=$1 L=$2 DFF=$3 E=$4 TK=$5 SEQ=$6; shift 6
  local out
  out=$($PY -m tgai_go.train.v3_pretrain --data "$ROOT/data512" --vocab 96000 \
        --seq "$SEQ" --d 1024 --layers "$L" --E "$E" --n_q 16 --d_ff "$DFF" \
        --top_k "$TK" --csa 4 --opt int8gpu --steps $STEPS --log-every 1 \
        --no-final-save "$@" 2>&1)
  if echo "$out" | grep -q '"status": "ok"'; then
    local s t m
    s=$(echo "$out" | grep -oE '"step_s_steady": [0-9.]+' | grep -oE '[0-9.]+')
    t=$(echo "$out" | grep -oE '"tok_s_steady": [0-9.]+' | grep -oE '[0-9.]+')
    m=$(echo "$out" | grep -oE '"pool_peak_live_gib": [0-9.]+' | grep -oE '[0-9.]+')
    printf "ORV   %-22s %7.3f s  %8.1f tok/s  %6.2f GiB\n" "$name" "$s" "$t" "$m"
  else
    printf "ORV   %-22s FAIL (%s)\n" "$name" "$(echo "$out" | grep -oE '显存不足|OutOfMemory|CudaError' | head -1)"
  fi
}

# ---------- PyTorch ----------
pyt() {   # 名字 层数 d_ff E top_k seq
  local name=$1 L=$2 DFF=$3 E=$4 TK=$5 SEQ=$6
  local out
  out=$($PY v3_pyt_train.py --d 1024 --layers "$L" --E "$E" --vocab 96000 \
        --d_ff "$DFF" --n_heads 16 --top_k "$TK" --seq "$SEQ" --steps $STEPS 2>&1)
  if echo "$out" | grep -q '"impl": "pytorch"'; then
    local s m
    s=$(echo "$out" | grep -oE '"step_s": [0-9.]+' | grep -oE '[0-9.]+')
    m=$(echo "$out" | grep -oE '"peak_mib": [0-9]+' | grep -oE '[0-9]+')
    awk -v n="$name" -v s="$s" -v m="$m" 'BEGIN{printf "PyT   %-22s %7.3f s  %8.1f tok/s  %6.2f GiB\n", n, s, 512/s, m/1024}'
  else
    printf "PyT   %-22s FAIL (%s)\n" "$name" "$(echo "$out" | grep -oE 'out of memory|OutOfMemory' | head -1)"
  fi
}

echo "===== A. MoE（V3 形状：E=4+1 共享, top_k=2, d_ff=3328）seq512 ====="
for L in 6 8 12 16 24; do
  orv "moe-${L}L" "$L" 3328 4 2 512
  pyt "moe-${L}L" "$L" 3328 4 2 512
done

echo
echo "===== B. 稠密（无专家：E=1, top_k=1, d_ff=13312 ≈ 同参数量级）seq512 ====="
for L in 8 12 16 24; do
  orv "dense-${L}L" "$L" 13312 1 1 512 --no-shared
  pyt "dense-${L}L" "$L" 13312 1 1 512
done

echo
echo "===== C. 序列长度（MoE 12 层）====="
for S in 256 512 1024 2048; do
  orv "moe-12L-seq${S}" 12 3328 4 2 "$S"
  pyt "moe-12L-seq${S}" 12 3328 4 2 "$S"
done
