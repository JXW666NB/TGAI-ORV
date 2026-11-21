#!/bin/bash
# (1) seq512 ablation on FIXED tree: off | gemm | gemm-no-cast  (x2 rounds)
# (2) seq2048+ckpt on BASE tree (pre-fix): off | gemm          (x2 rounds)
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1

C512="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 12 --log-every 99 --no-final-save"
C2048="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 12 --log-every 99 --no-final-save"

get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run512() {  # $1=tag $2=amp $3=nocast
  export PROBE_MODE=time PROBE_NO_CAST=$3 PROBE_OUT=/root/abl_$1.json
  export PYTHONPATH=/root/orv_src/python
  $PY /root/probe_amp.py $C512 --amp $2 >/root/abl_$1.log 2>&1
  echo "  $1 amp=$2 nocast=$3 step=$(get step_s_steady /root/abl_$1.log)"
}

run2048base() {  # $1=tag $2=amp
  export PYTHONPATH=/root/orv_base/python
  TG_GRAD_CKPT=1 $PY -m tgai_go.train.v3_pretrain $C2048 --amp $2 >/root/b2k_$1.log 2>&1
  echo "  base2048 $1 amp=$2 step=$(get step_s_steady /root/b2k_$1.log)"
}

for r in 1 2; do
  echo "--- seq512 ablation round $r ---"
  run512 off_r$r  off  0
  run512 gemm_r$r gemm 0
  run512 nc_r$r   gemm 1
done
for r in 1 2; do
  echo "--- base-tree seq2048 round $r ---"
  run2048base off_r$r  off
  run2048base gemm_r$r gemm
done
echo DONE
