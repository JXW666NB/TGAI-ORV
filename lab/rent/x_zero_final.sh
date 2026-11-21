#!/bin/bash
# 优化 reduce_scatter 之后复测：1.82B / seq512 / zero=2 峰值与步时
unset PYTHONPATH
set -u
bash /root/x_zero4090.sh mem 512 2 3
echo WRAPPER5_DONE
