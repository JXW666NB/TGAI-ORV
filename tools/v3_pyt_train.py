# -*- coding: utf-8 -*-
"""PyTorch 渚э細鏋勯€犱笌 ORV 鍚岃妯＄殑 V3 閰嶇疆锛岃窇鐪熷疄璁粌姝ワ紙AdamW锛夈€?
## 鍙瘮涓庝笉鍙瘮锛堝厛鍐欐竻锛?
* **鍙瘮**锛氬嘲鍊兼樉瀛樸€佹瘡姝ヨ€楁椂銆乼ok/s锛堥兘鏄?鍚岃妯°€佸悓 batch銆佸悓 seq銆佸悓 step 鏁?锛夈€?* **涓嶅彲姣?*锛歭oss 鈥斺€?涓や晶鐙珛闅忔満鍒濆鍖栵紝璧风偣涓嶅悓銆?* **浠嶄笉涓€鑷?*锛歅yTorch 鐨?V3 榛樿鎶婇儴鍒嗗眰鍋氭垚 **CSA**锛堝帇缂╃█鐤忥級锛孫RV 渚х洰鍓嶅叏鏄?  GlobalAttention锛涗笓瀹舵暟/婵€娲绘暟/鍏变韩涓撳鍙榻愶紝浣嗘敞鎰忓姏绉嶇被鏆備笉鍙€傝繖涓€鐐瑰繀椤婚殢鏁板瓧涓€璧锋姤銆?
鐢ㄦ硶锛歚python v3_pyt_train.py --d 512 --layers 8 --E 8 --seq 512 --steps 5`
"""
import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np

V3DIR = r"D:/TGAI-V3-1.5B/03_src"
sys.path.insert(0, V3DIR)

import torch  # noqa: E402


def vram_peak(stop, base, prof):
    peak = base
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    prof["peak"] = peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import model_v3 as M

    cfg = M.TGAIV3Config(vocab_size=4096, d_model=a.d, n_layers=a.layers,
                         n_heads=4, n_kv_heads=2, d_ff=4 * a.d,
                         n_experts=a.E, n_activated=1, n_shared_experts=1,
                         max_seq_len=a.seq, moe_balance="sign")
    model = M.TGAIV3Model(cfg).cuda()
    # ★ 与 ORV 侧同口径：PyTorch 也开梯度检查点
    #   参考实现里检查点只在 `gradient_checkpointing and training and targets is not None`
    #   时才生效 ⇒ 必须把 targets 传进 forward，否则等于没开（我上一版就是漏了这点）
    model.enable_gradient_checkpointing(True)
    model.train()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    stop = threading.Event(); prof = {}
    th = threading.Thread(target=vram_peak, args=(stop, base, prof), daemon=True); th.start()

    rng = np.random.default_rng(0)
    ids_all = rng.integers(0, 4096, (a.steps, a.seq)).astype(np.int64)
    times, losses = [], []
    for step in range(a.steps):
        ids = torch.from_numpy(ids_all[step]).cuda()[None, :]
        tgt = ids[:, 1:].contiguous()
        t0 = time.time()
        # 模型内部自己平移 targets（传等长 ids），与 train_v3.py 的用法一致
        out = model(ids, targets=ids)
        if isinstance(out, (tuple, list)):
            logits, loss = out[0], out[1]
        else:
            logits = out
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
                tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0); losses.append(float(loss.item()))
        print("step %d loss=%.4f %.2fs" % (step, losses[-1], times[-1]), flush=True)

    stop.set(); th.join(timeout=3)
    res = dict(impl="pytorch", cfg=dict(d=a.d, layers=a.layers, E=a.E, seq=a.seq,
                                        vocab=4096, steps=a.steps),
               n_params=int(n_par), losses=losses,
               step_s=float(np.mean(times)), tok_s=a.seq / float(np.mean(times)),
               peak_mib=prof.get("peak"), base_mib=base,
               torch_peak_mib=int(torch.cuda.max_memory_allocated() // 1024 // 1024))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
    print("losses", losses)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
