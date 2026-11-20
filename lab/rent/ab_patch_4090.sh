#!/bin/bash
# A/B锛氬熀绾匡紙宸插畨瑁?1.0.8锛塿s 琛ヤ竵鍖咃紙/root/patched_pkg锛夛紝浜ゆ浛璺?3 杞?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 12 --log-every 99 --no-final-save"

one() {
  local tag="$1" pp="$2"
  local out
  if [ -n "$pp" ]; then
    out=$(env PYTHONPATH="$pp" $PY -m tgai_go.train.v3_pretrain $COMMON 2>&1)
  else
    out=$(env $PY -m tgai_go.train.v3_pretrain $COMMON 2>&1)
  fi
  local s l g
  s=$(echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$')
  l=$(echo "$out" | grep -o '"loss_last": [0-9.]*' | grep -o '[0-9.]*$')
  g=$(echo "$out" | grep -o '"gnorm[^,]*' | tail -1)
  echo "  $tag steady=${s:-FAIL} loss_last=${l:-?} $g"
}

echo "=== 鍩虹嚎锛?.0.8锛変笌琛ヤ竵浜ゆ浛 ==="
for i in 1 2 3; do
  one "base[${i}]" ""
  one "patch[${i}]" "/root/patched_pkg"
done
echo "=== 鏁板€煎鎷嶏紙琛ヤ竵 vs 鍩虹嚎锛岄€愭 loss锛?=="
env $PY -m tgai_go.train.v3_pretrain $COMMON --log-every 1 2>&1 | grep -E '^step' | sed 's/^/  base /' | head -6
env PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain $COMMON --log-every 1 2>&1 | grep -E '^step' | sed 's/^/  patc /' | head -6
