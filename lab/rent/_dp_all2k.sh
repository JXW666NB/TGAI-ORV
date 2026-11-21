#!/bin/bash
# 1) 复核 arbench 的数值（应为 1.5 = (1+2)/2）
# 2) seq=2048 的 3 轮 A/B
echo "##### ARBENCH #####"
bash /root/bench/_dp_arbench.sh
echo "##### BENCH 2048 #####"
bash /root/bench/_dp_bench.sh 2048
