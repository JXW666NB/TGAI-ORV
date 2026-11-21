#!/bin/bash
# 最终对照：seq512 —— 同一 1.82B，DP 是否 OOM / ZeRO-2 是否跑通
unset PYTHONPATH
set -u
bash /root/x_zero4090.sh mem 512 0 2
bash /root/x_zero4090.sh mem 512 2 3
echo WRAPPER4_DONE
