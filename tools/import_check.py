# -*- coding: utf-8 -*-
"""最小导入自检：确认远端包能 import，并打印版本。"""
import sys

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import tgai_go.train as T

print("导入 OK")
print("版本:", getattr(T, "__version__", "?"))
print("amp:", hasattr(T, "amp"), "| preset:", hasattr(T, "preset"),
      "| fit:", hasattr(T, "fit"), "| DPOTrainer:", hasattr(T, "DPOTrainer"))
