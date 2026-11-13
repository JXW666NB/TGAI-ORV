# -*- coding: utf-8 -*-
"""单档训练步测试：--impl orv|torch --seq N --steps N。真语料、同一份 id。"""
import argparse, importlib.util, json, os, subprocess, sys, threading, time
import numpy as np

def vram_peak(stop, base):
    peak = [base]
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak[0] = max(peak[0], int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    return peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("--impl", required=True)
    ap.add_argument("--seq", type=int, required=True); ap.add_argument("--steps", type=int, default=10)
    # ★ 只测前反向：不建优化器状态（省掉 m/v 的 11–22 GiB）。
    #   这是目标允许的口径（"若放不下：…/ 只测前反向 /…"），两侧**同时**使用才可比。
    ap.add_argument("--no-opt", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    ids_all = np.load(os.path.join(cfg["outdir"], "ids_%d.npy" % a.seq))
    need = a.steps + 1
    if len(ids_all) < need:
        raise SystemExit("窗口不够：需要 %d，只有 %d" % (need, len(ids_all)))
    lr = float(cfg["lr"])
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0])
    res = {"impl": a.impl, "seq": a.seq, "steps": a.steps, "base_mib": base}
    # ★ 峰值显存必须**在跑之前**就开始采：放在循环之后只能采到收尾时的值
    _pstop = threading.Event()
    _phold = []
    _th = threading.Thread(target=lambda: _phold.append(vram_peak(_pstop, base)), daemon=True)
    _th.start()

    if a.impl == "orv":
        import tgai_go.train as T
        from tgai_go.train.v27_train import V27Trainer, TGState
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        T.set_default_device("cuda")
        st = TGState(cfg["model_tg"])
        h = None
        from tgai_go.read_tg import read_tg, close_tg
        info = read_tg(cfg["model_tg"]); h = info["header"]; close_tg(info)
        c = dict(n_layers=int(h["n_layers"]), d_model=int(h["d_model"]), n_heads=int(h["n_heads"]),
                 d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
                 n_activated=int(h["n_activated"]), rope_theta=float(h["rope_theta"]),
                 vocab=int(h["vocab"]), max_seq=int(h["max_seq"]),
                 pad=int(h["pad"]), bos=int(h["bos"]), eos=int(h["eos"]), unk=int(h["unk"]),
                 n_kv_heads=int(h.get("num_kv_heads") or 0), arch=int(h.get("arch") or 0))
        # ★ 同口径关键：ORV 侧参数也走 fp16（与 torch 侧 model.to(bfloat16) 对齐到"半精度存储"）。
        #   0.9.63 起支持；不给这个参数就是 fp32 参数 ⇒ 44.7 GiB，必 OOM。
        tr = V27Trainer(c, n_layers=c["n_layers"])
        tr.load_all(st)
        # ★ 必须先挂 head（token_embedding + final_norm），否则 head_params() 里
        #   访问 self.final_norm 会 AttributeError（实测踩过）
        tr.attach_head(st)
        # ★ RoPE 表必须**现算**成 (seq, dh)，不能直接取检查点里的 `blocks.0.attn.rope.cos`：
        #   后者形状是 (1,1,8192,64)（预置 batch/head 维），灌进 `_rope_composed` 会广播失败
        #   （实测报错 `(16,512,64) vs (1,1,8192,64)`）。算法与 v27_train_run.py 一致。
        _dh = c["d_model"] // c["n_heads"]
        _j = np.arange(0, _dh, 2, dtype=np.float32) / _dh
        _ang = np.repeat(np.arange(a.seq, dtype=np.float32)[:, None]
                         * (float(c["rope_theta"]) ** (-_j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(_ang).astype(np.float32))
        sin = T.tensor(np.sin(_ang).astype(np.float32))
        # ★ 手动指定 fp16 动量的优化器：`tr.make_optimizers()` 默认给的是 **fp32 AdamW**
        #   （2.999B ⇒ m/v 各 11.18 GiB，合计 44.7 GiB，32 GB 卡必 OOM）。
        #   这里换成 Fp16MomAdamW（4 B/参数），状态从 22.36 降到 11.18 GiB。
        # --no-opt：不建优化器（省 m/v）；只做前反向
        opts = None if a.no_opt else [Fp16MomAdamW(tr.params_of(i), lr=lr) for i in range(tr.NL)]
        head_opt = None if a.no_opt else Fp16MomAdamW(tr.head_params(), lr=lr)
        losses, times = [], []
        for step in range(a.steps):
            ids = ids_all[step].astype(np.int32)
            tg = np.roll(ids, -1); tg[-1] = int(h["eos"])
            t0 = time.time()
            x0_buf, _id = tr.x0_from_ids(ids)
            x0 = np.asarray(x0_buf.to_numpy(), np.float32)
            xs = tr.forward_all(x0, cos, sin)
            loss, g = tr.loss_and_head_backward(xs[-1], tg, cos, sin)
            _l = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
            lv = float(_l) if isinstance(_l, (int, float)) else float(np.asarray(_l).reshape(-1)[0])
            losses.append(lv)
            if head_opt is not None:
                head_opt.step(); head_opt.zero_grad()
            for p in tr.head_params(): p.grad = None
            tr.backward_and_step(xs, g, cos, sin, opts)
            T._cuda().sync() if hasattr(T._cuda(), "sync") else None
            times.append(time.time() - t0)
            print("  step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)
        res.update(losses=losses, times=times)

    else:
        import torch
        spec = importlib.util.spec_from_file_location("ref", cfg["model_py"])
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        ck = torch.load(cfg["model_pt"], map_location="cpu", weights_only=False)
        m = mod.TGAILanguageModel(mod.TGAIConfig(**cfg["config"]))
        m.load_state_dict(ck["model_state_dict"])
        m = m.cuda().to(torch.bfloat16)
        # ★ 手写 bf16 AdamW（2026-11-13）：**不用** torch.optim.AdamW。
        #   原生 AdamW 是 fp32 状态（m/v 各 11.18 GiB）⇒ 2.999B 在 32 GB 卡上必 OOM，
        #   那等于故意让 PyTorch 跑不通。这里与 ORV 的 Fp16MomAdamW 对齐：
        #   权重/梯度/动量全部 bf16，同一套 AdamW 更新式、同超参（betas=(0.9,0.95), eps=1e-8, wd=0.1）。
        class Bf16AdamW:
            def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, wd=0.1):
                self.ps = [p for p in params if p.requires_grad]
                self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
                self.t = 0
                self.m = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.ps]
                self.v = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.ps]
                self.bytes_per_param = 4          # m + v，各 2 字节

            @torch.no_grad()
            def step(self):
                self.t += 1
                bc1 = 1.0 - self.b1 ** self.t
                bc2 = 1.0 - self.b2 ** self.t
                for p, mm, vv in zip(self.ps, self.m, self.v):
                    if p.grad is None:
                        continue
                    g = p.grad.to(torch.bfloat16)
                    mm.mul_(self.b1).add_(g * (1 - self.b1))
                    vv.mul_(self.b2).add_(g * g * (1 - self.b2))
                    denom = (vv.float() / bc2).sqrt_().add_(self.eps)
                    upd = (mm.float() / bc1) / denom
                    p.data.add_(upd.to(torch.bfloat16), alpha=-self.lr)
                    if self.wd:
                        p.data.mul_(1.0 - self.lr * self.wd)

            def zero_grad(self):
                for p in self.ps:
                    p.grad = None

        opt = None if a.no_opt else Bf16AdamW(m.parameters(), lr=lr)
        losses, times = [], []
        for step in range(a.steps):
            ids = torch.from_numpy(ids_all[step].astype(np.int64)).cuda()[None, :]
            t0 = time.time()
            out = m(ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            lo = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            tg = ids[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(lo, tg)
            if opt is not None: opt.zero_grad()
            loss.backward()
            if opt is not None: opt.step()
            torch.cuda.synchronize()
            losses.append(float(loss.item())); times.append(time.time() - t0)
            print("  step %d loss=%.4f %.2fs" % (step, losses[-1], times[-1]), flush=True)
        res["peak_mib_torch"] = int(torch.cuda.max_memory_allocated() // 1024 // 1024)
        res.update(losses=losses, times=times)

    res["step_s"] = float(np.mean(res["times"]))
    res["tok_s"] = a.seq / res["step_s"]
    _pstop.set(); _th.join(timeout=3)
    if _phold:
        res["peak_mib_smi"] = _phold[0][0]
        res["peak_delta_mib"] = _phold[0][0] - base
    out = os.path.join(cfg["outdir"], "res_%s_%d.json" % (a.impl, a.seq))
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({k: res[k] for k in res if k not in ("losses", "times")},
                               ensure_ascii=False))
    print("losses", res["losses"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
