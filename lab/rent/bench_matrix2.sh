#!/usr/bin/env bash
# 补跑：A 段（MoE 各层数）+ 长 seq（先按正确行宽生成数据）
set -uo pipefail
ROOT=/root/orv_rent
PY=/root/miniconda3/bin/python
cd "$ROOT"
STEPS=4

# 长 seq 需要更宽的分片行（data512 的每行只有 512 个 token，切不出 1024 的窗口）
if [ ! -d data2048 ]; then
  $PY make_packed.py --out "$ROOT/data2048" --seq 2048 --vocab 96000 --shards 8 --rows 16
fi

orv() {   # 名字 层数 d_ff E top_k seq 数据目录 额外参数
  local name=$1 L=$2 DFF=$3 E=$4 TK=$5 SEQ=$6 DATA=$7; shift 7
  local out
  out=$($PY -m tgai_go.train.v3_pretrain --data "$DATA" --vocab 96000 \
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
    printf "ORV   %-22s FAIL: %s\n" "$name" "$(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-120)"
  fi
}

pyt() {
  local name=$1 L=$2 DFF=$3 E=$4 TK=$5 SEQ=$6
  local out
  out=$($PY v3_pyt_train.py --d 1024 --layers "$L" --E "$E" --vocab 96000 \
        --d_ff "$DFF" --n_heads 16 --top_k "$TK" --seq "$SEQ" --steps $STEPS 2>&1)
  if echo "$out" | grep -q '"impl": "pytorch"'; then
    local s m
    s=$(echo "$out" | grep -oE '"step_s": [0-9.]+' | grep -oE '[0-9.]+')
    m=$(echo "$out" | grep -oE '"peak_mib": [0-9]+' | grep -oE '[0-9]+')
    awk -v n="$name" -v s="$s" -v m="$m" -v q="$SEQ" \
      'BEGIN{printf "PyT   %-22s %7.3f s  %8.1f tok/s  %6.2f GiB\n", n, s, q/s, m/1024}'
  else
    printf "PyT   %-22s FAIL: OOM\n" "$name"
  fi
}

echo "===== A. MoE（E=4+1, top_k=2, d_ff=3328）seq512 ====="
for L in 6 8 12 16 24; do
  orv "moe-${L}L" "$L" 3328 4 2 512 "$ROOT/data512"
  pyt "moe-${L}L" "$L" 3328 4 2 512
done

echo
echo "===== C. 长序列（MoE 12 层）====="
for S in 1024 2048; do
  orv "moe-12L-seq${S}" 12 3328 4 2 "$S" "$ROOT/data2048"
  pyt "moe-12L-seq${S}" 12 3328 4 2 "$S"
done
