#!/usr/bin/env bash
# TGAI GO 训练引擎 · 一条命令自检（Linux / macOS / DSW / 容器）
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "仓库：$HERE"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then echo "找不到 python3，请先安装 Python 3.8+"; exit 1; fi
echo "Python：$($PY -V 2>&1)"
$PY -c "import numpy" 2>/dev/null || { echo "-> pip install numpy"; $PY -m pip install -q numpy; }
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed "s/^/GPU：/"
else
  echo "警告：没检测到 nvidia-smi（可能没有 NVIDIA 驱动）"
fi
export PYTHONPATH="$HERE/python${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"
echo "===== 自检（12 项）====="
$PY -m tgai_go.train.doctor
rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "自检通过。下一步："
  echo "  训练：PYTHONPATH=$HERE/python python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out chat10m"
  echo "  验收：PYTHONPATH=$HERE/python python3 tools/accept.py --quick"
  echo "  文档：docs/QUICKSTART_TRAIN.md 与 docs/PRODUCTION_READINESS.md"
else
  echo "自检未通过：每项都写了「哪个算子 / 什么形状 / 怎么修」，按提示处理。"
fi
exit $rc
