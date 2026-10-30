import sys, importlib.util
sys.path.insert(0, "python"); sys.path.insert(0, ".")
import numpy as np
import tgai_go.train as T
from tgai_go.train import functional as F
spec = importlib.util.spec_from_file_location("q", "examples/qwen25_sft.py")
q = importlib.util.module_from_spec(spec); sys.modules["q"] = q; spec.loader.exec_module(q)

T.set_default_device("cpu")
cfg = q.TINY_CFG
model = q.Qwen25(cfg, q.make_tiny_weights(cfg))
ids = list(q.tiny_sequences(cfg)[0])[:32]
xb = np.asarray([ids[:-1]], np.int64); yb = np.asarray([ids[1:]], np.int64)
for p in model.params():
    p.zero_grad()
logits = model.forward(xb)
loss = F.cross_entropy(logits.reshape(xb.size, cfg["V"]), yb.reshape(-1))
loss.backward()
print("loss=%.6f" % float(loss.to_numpy()))
rows = []
for k, p in model.p.items():
    g = getattr(p, "grad", None)
    rows.append((k, None if g is None else float(np.abs(np.asarray(g, np.float32)).max())))
none_cnt = sum(1 for _, v in rows if v is None)
print("参数 %d 个，其中 grad 为 None 的 %d 个" % (len(rows), none_cnt))
for k, v in rows[:6]:
    print("  %-38s %s" % (k, "None" if v is None else "%.3e" % v))
for k, v in rows[-6:]:
    print("  %-38s %s" % (k, "None" if v is None else "%.3e" % v))
