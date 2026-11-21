#!/bin/bash
# Compare 4 variants at seq512, 3 alternating rounds:
#   off      : /root/orv_fix2  --amp off
#   A        : /root/orv_src   --amp gemm        (O(1) invalidation only)
#   AB       : /root/orv_fix2  --amp gemm        (+ reuse shadow buffers across steps)
#   NC       : /root/orv_fix2  --amp gemm --amp-no-cache
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save"
get() { grep -o "\"$1\": [0-9.]*" "$2" | grep -o '[0-9.]*$'; }

run() {  # tag pyroot amp extra
  PYTHONPATH=$2 $PY -m tgai_go.train.v3_pretrain $C --amp $3 $4 >/root/c_$1.log 2>&1
  echo "  $1 step=$(get step_s_steady /root/c_$1.log) peak=$(get pool_peak_live_gib /root/c_$1.log)"
}

for r in 1 2 3; do
  echo "--- round $r ---"
  run off_r$r  /root/orv_fix2/python off  ""
  run A_r$r    /root/orv_src/python  gemm ""
  run AB_r$r   /root/orv_fix2/python gemm ""
  run NC_r$r   /root/orv_fix2/python gemm "--amp-no-cache"
done
echo DONE
