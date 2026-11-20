#!/bin/bash
# 4090锛歠p16 / bf16 / off 涓夋。瀵规瘮
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 6 --log-every 99 --no-final-save"

run() {
  local tag="$1"; shift
  local out
  out=$(env "$@" $PY -m tgai_go.train.v3_pretrain $COMMON 2>&1)
  local s
  s=$(echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')
  local f
  f=$(echo "$out" | grep -o "'fp16': [A-Za-z]*" | head -1)
  echo "  $tag -> steady=${s:-FAIL}s  $f"
}

run "fp16(amp gemm)"        AMPMODE=1 >/dev/null 2>&1  # 鍗犱綅锛屼笅闈㈡墠鏄湡璺?echo "=== fp16 (--amp gemm) ==="
out=$(env $PY -m tgai_go.train.v3_pretrain $COMMON --amp gemm 2>&1); echo "  steady=$(echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')s"
echo "=== bf16 (--amp gemm + TG_BF16=1) ==="
out=$(env TG_BF16=1 $PY -m tgai_go.train.v3_pretrain $COMMON --amp gemm 2>&1)
echo "  steady=$(echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')s"
echo "$out" | grep -iE 'bf16|cannot|涓嶅彲鐢▅鍥為€€' | head -3
echo "=== off (fp32) ==="
out=$(env $PY -m tgai_go.train.v3_pretrain $COMMON --amp off 2>&1); echo "  steady=$(echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')s"
