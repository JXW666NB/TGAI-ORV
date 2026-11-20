#!/bin/bash
# 鈶?鍏。浼樺寲鍣ㄦ暣姝ュ姣?鈶?1.72B 鍦?24GB 涓婄殑鍐呭瓨杈圭晫銆傚叏閮ㄨ拷鍔犲啓 /root/m2.log
cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
LOG=/root/m2.log
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

P512="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --amp off \
--batch 1 --steps 20 --log-every 99 --no-final-save"

# 鈶?浼樺寲鍣紙鍚?2 杞級
for opt in int8gpu adamw fp16mom int8 muon adafactor; do
  for r in 1 2; do
    out=$(PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain $P512 --opt "$opt" 2>&1)
    s=$(echo "$out" | get '"step_s_steady": [0-9.]*')
    pk=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')
    echo "opt=$opt r=$r step=${s:-FAIL} peak=${pk:-?}" >> $LOG
  done
done

# 鈶?1.72B锛?0 灞?/ E=8 / d_ff=2816 / tie锛塻eq512 杈圭晫
B="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 20 --E 8 --d_ff 2816 --top_k 2 --csa 4 --n_q 16 --amp off \
--batch 1 --steps 10 --log-every 99 --no-final-save"

out=$(TG_GRAD_CKPT=0 PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain $B --opt int8gpu 2>&1)
s=$(echo "$out" | get '"step_s_steady": [0-9.]*')
pk=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')
pa=$(echo "$out" | get '"params": [0-9]*')
echo "big: orv_int8gpu params=${pa:-?} step=${s:-OOM} peak=${pk:-?}" >> $LOG

out=$(TG_GRAD_CKPT=0 PYTHONPATH=/root/patched_pkg $PY -m tgai_go.train.v3_pretrain $B --opt adamw 2>&1)
s=$(echo "$out" | get '"step_s_steady": [0-9.]*')
echo "big: orv_adamw_fp32 step=${s:-OOM}" >> $LOG

# PyT 渚х敤 v3_pyt_train.py锛堝悓鏍风殑 20 灞?/ E=8 / d_ff=2816锛?out=$($PY v3_pyt_train.py --d 1024 --layers 20 --E 8 --vocab 96000 --d_ff 2816 \
      --n_heads 16 --top_k 2 --seq 512 --steps 10 --ckpt 0 2>&1)
s=$(echo "$out" | get '"step_s": [0-9.]*')
echo "big: pyt_fp32 step=${s:-OOM}" >> $LOG

echo DONE >> $LOG
