import sys, importlib.util
sys.path.insert(0, "python"); sys.path.insert(0, ".")
import numpy as np
import tgai_go.train as T
spec = importlib.util.spec_from_file_location("q", "examples/qwen25_sft.py")
q = importlib.util.module_from_spec(spec); sys.modules["q"] = q; spec.loader.exec_module(q)

T.set_default_device("cpu")
cfg = q.TINY_CFG
for lr in (1e-3, 3e-3):
    model = q.Qwen25(cfg, q.make_tiny_weights(cfg))
    seq = q.tiny_sequences(cfg)[0]          # 固定同一条序列
    losses = q.sft(model, [seq], 12, lr, 32, log_every=99)
    print("lr=%-6g  首 %.6f  末 %.6f  最低 %.6f  %s"
          % (lr, losses[0], losses[-1], min(losses),
             "下降" if losses[-1] < losses[0] else "未降"))
