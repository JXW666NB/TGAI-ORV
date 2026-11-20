#!/bin/bash
# 4090 鍏钩瀵规瘮鍩虹嚎锛氫袱渚у悇鍙栨渶浼樿缃紝浜ゆ浛 3 杞?#   ORV : --amp off锛堝疄娴嬫渶浼橈級/ --amp gemm
#   PyT : --amp / 鏃?--amp
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
ORV="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"

orv() {  # $1=mode  $2=py_path(绌?宸插畨瑁?
  local out
  if [ -n "$2" ]; then
    out=$(env PYTHONPATH="$2" $PY -m tgai_go.train.v3_pretrain $ORV --amp "$1" 2>&1)
  else
    out=$(env $PY -m tgai_go.train.v3_pretrain $ORV --amp "$1" 2>&1)
  fi
  echo "$out" | grep -o '"step_s_steady": [0-9.]*' | grep -o '[0-9.]*$'
}
pyt() {  # $1=ampflag(1/0)
  local extra=""
  [ "$1" = "1" ] && extra="--amp"
  local out
  out=$($PY v3_pyt_train.py --d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 \
        --n_heads 16 --top_k 2 --seq 512 --steps 20 --ckpt 0 $extra 2>&1)
  echo "$out" | grep -o '"step_s": [0-9.]*' | grep -o '[0-9.]*$'
}

echo "=== 浜ゆ浛 3 杞紙鍗曚綅 s锛?=="
printf "  %-6s %-10s %-10s %-10s %-10s\n" "杞? "ORV_fp32" "ORV_fp16" "PyT_fp16" "PyT_fp32"
for i in 1 2 3; do
  a=$(orv off /root/patched_pkg); b=$(orv gemm /root/patched_pkg)
  c=$(pyt 1); d=$(pyt 0)
  printf "  %-6s %-10s %-10s %-10s %-10s\n" "$i" "${a:-FAIL}" "${b:-FAIL}" "${c:-FAIL}" "${d:-FAIL}"
done
