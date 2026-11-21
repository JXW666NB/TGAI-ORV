#!/bin/bash
# 1.82B 的"装得下"对照：seq2048 DP(--zero 0) 应 OOM；同一个模型 --zero 2 看能否跑通
unset PYTHONPATH
set -u
bash /root/x_zero4090.sh mem 2048 0 2
bash /root/x_zero4090.sh mem 2048 2 3
echo WRAPPER_DONE
