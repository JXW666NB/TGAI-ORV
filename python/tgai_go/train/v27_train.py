# -*- coding: utf-8 -*-
"""V2.7 全参训练循环（ORV）：**逐层反向**，把梯度显存压到"单层"。

为什么这样设计（实测依据，见 docs/TRAIN_V27_ORV.md）：
* 3.033B 全参 fp32 权重 11.3 GB、fp16 动量 11.3 GB；
* 若梯度也全量常驻（fp32 11.3 GB）合计 33.9 GB ⇒ 超 32 GB；
* 逐层反向把梯度变成"算完这层立刻 step 再释放" ⇒ 峰值约 23 GB。

用法：
    tr = V27Trainer(cfg).load_all(sd)          # sd: {name: torch/np 张量}
    opts = tr.make_optimizers(lr=1e-5)
    xs = tr.forward_all(x0, cos, sin)          # 不建图，只缓存每层输入
    tr.backward_and_step(xs, g_out, cos, sin, opts)
"""
from __future__ import annotations

import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import _cuda
from tgai_go.train import Linear, MoE, RMSNorm


def _np(z):
    """Tensor/ndarray → ndarray（CPU 上 grad 是 ndarray，GPU 上是 Tensor）。"""
    return z.to_numpy() if hasattr(z, "to_numpy") else np.asarray(z)


def _as_np(a):
    """torch tensor / ndarray → fp32 连续 ndarray（流式装载要求逐张量）。"""
    if hasattr(a, "float") and hasattr(a, "numpy"):
        a = a.float().numpy()
    return np.ascontiguousarray(a, np.float32)


class V27Trainer(object):
    """按层组织，支持逐层反向；权重经 `load_state_dict_stream` 流式上卡。"""

    def __init__(self, cfg, n_layers=None, param_dtype="fp32"):
        self.cfg = dict(cfg)
        # ★ 参数存储精度（2026-11-13 加）：默认 fp32 保持历史行为。
        #   2.999B 全参训练在 32 GB 卡上必须用 fp16：fp32 权重+梯度+优化器状态 = 44.7 GiB 放不下，
        #   fp16 全套 = 22.35 GiB 才进得去。**只作用于注意力和 MoE 权重**，
        #   RMSNorm / final_norm / RoPE 表由调用方保持 fp32（数值敏感）。
        _pd = str(param_dtype).lower()
        self._pdt = np.float16 if _pd in ("fp16", "float16", "half") else np.float32
        self.D = int(cfg["d_model"])
        self.NH = int(cfg["n_heads"])
        self.DH = self.D // self.NH
        self.FF = int(cfg["d_ff"])
        self.E = int(cfg["n_experts"])
        self.K = int(cfg["n_activated"])
        self.NL = int(n_layers if n_layers is not None else cfg["n_layers"])
        self.layers = []

    # ---------- 建层 ----------
    def _build_layer(self):
        D, NH, DH, FF, E, K = self.D, self.NH, self.DH, self.FF, self.E, self.K
        L = dict(ln1=RMSNorm(D), ln2=RMSNorm(D), q=[], k=[], v=[], o=[])
        if getattr(self, "fused_heads", False):
            # 全宽投影：列表长度 1，modules_of/params_of 不用改（见 docs/PLAN_FUSED_ATTN.md）
            L["q"].append(Linear(D, D, bias=False))
            L["k"].append(Linear(D, D, bias=False))
            L["v"].append(Linear(D, D, bias=False))
            L["o"].append(Linear(D, D, bias=False))
        else:
            for _ in range(NH):
                L["q"].append(Linear(D, DH, bias=False))
                L["k"].append(Linear(D, DH, bias=False))
                L["v"].append(Linear(D, DH, bias=False))
                L["o"].append(Linear(DH, D, bias=False))
        L["moe"] = MoE(D, FF, E, top_k=K, gate="softmax_topk",
                       qb_bias=np.zeros(E, np.float32))
        # 构造期分配的随机权重在 GPU 上会占满一份（MoE 每层 384 MB）；
        # 这里**连设备缓冲一起释放**，之后由流式装载只分配一次 ⇒ 避免 2~3 倍开销。
        for m in [L["ln1"], L["ln2"], L["moe"]] + L["q"] + L["k"] + L["v"] + L["o"]:
            for prm in m.parameters():
                if getattr(prm, "device", "cpu") == "cuda":
                    buf = getattr(prm, "_gpu", None)
                    if buf is not None and int(getattr(buf, "n", 0)) > 0:
                        try:
                            _cuda().free(int(buf.ptr))
                        except Exception:
                            pass
                    prm._lazy_shape = tuple(int(x) for x in buf.shape) if buf is not None else prm._lazy_shape
                    prm._gpu = None
                    prm.data = None
        if getattr(L["moe"].qb_bias, "device", "cpu") == "cuda":
            L["moe"].qb_bias.data = None
        return L

    # ---------- 装载 ----------
    def load_layer(self, i, sd):
        """把第 i 层权重流式装进已在卡上的模块（宿主峰值 = 单张量）。"""
        D, NH, DH = self.D, self.NH, self.DH
        L = self._build_layer()
        p = "blocks.%d." % i

        def items():
            yield "ln1.weight", _as_np(sd[p + "ln1.weight"])
            yield "ln2.weight", _as_np(sd[p + "ln2.weight"])
        for mod, key in ((L["ln1"], "ln1.weight"), (L["ln2"], "ln2.weight")):
            mod.load_state_dict_stream(iter([(key.split(".")[-1], _as_np(sd[p + key]))]))

        qw = _as_np(sd[p + "attn.q_proj.weight"])
        kw = _as_np(sd[p + "attn.k_proj.weight"])
        vw = _as_np(sd[p + "attn.v_proj.weight"])
        ow = _as_np(sd[p + "attn.out_proj.weight"])
        if getattr(self, "fused_heads", False):
            # 检查点本来就是全宽 [D, D]（见 load_layer 上方说明），整张装载即可
            L["q"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw))]), dtype=self._pdt)
            L["k"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw))]), dtype=self._pdt)
            L["v"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw))]), dtype=self._pdt)
            L["o"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(ow))]), dtype=self._pdt)
        else:
            for h in range(NH):
                sl = slice(h * DH, (h + 1) * DH)
                L["q"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw[sl]))]), dtype=self._pdt)
                L["k"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw[sl]))]), dtype=self._pdt)
                L["v"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw[sl]))]), dtype=self._pdt)
                L["o"][h].load_state_dict_stream(
                    iter([("weight", np.ascontiguousarray(ow[:, sl]))]), dtype=self._pdt)
        del qw, kw, vw, ow

        def moe_items():
            yield "router.weight", _as_np(sd[p + "moe.router.weight"])
            for e in range(self.E):
                yield ("experts.%d.gate.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w1.weight" % e]))
                yield ("experts.%d.up.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w2.weight" % e]))
                yield ("experts.%d.down.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w3.weight" % e]))
        L["moe"].load_state_dict_stream(moe_items(), dtype=self._pdt)
        qb = _as_np(sd[p + "moe._qb_bias"]) if (p + "moe._qb_bias") in sd else None
        if qb is not None:
            L["moe"].qb_bias = T.tensor(qb)
            if getattr(L["moe"].qb_bias, "device", "cpu") == "cuda":
                L["moe"].qb_bias.data = None
        self.layers.append(L)
        return L

    def load_all(self, sd, on_layer=None):
        for i in range(self.NL):
            self.load_layer(i, sd)
            # 每层装完把池里的旧块真正归还驱动：流式装载会替换每个参数的 _gpu，
            # 旧块进池但不会被复用（尺寸不同），不回收就会把 3B 权重放大 3 倍。
            if getattr(self.layers[0]["ln1"].weight, "device", "cpu") == "cuda":
                _cuda().empty_cache()
            if on_layer:
                on_layer(i)
        return self

    # ---------- 参数与优化器 ----------
    def modules_of(self, i):
        L = self.layers[i]
        return [L["ln1"], L["ln2"], L["moe"]] + L["q"] + L["k"] + L["v"] + L["o"]

    def params_of(self, i):
        out = []
        for m in self.modules_of(i):
            out += [p for p in m.parameters() if getattr(p, "requires_grad", False)]
        return out

    def make_optimizers(self, lr=1e-5, **kw):
        return [T.optim.AdamW(self.params_of(i), lr=lr, **kw) for i in range(self.NL)]

    # ---------- 前向 ----------
    def _causal_mask(self, n):
        """因果掩码（上三角 -1e9）。

        **这是一个真实 bug 的修复**（2026-11-09 实测定位）：
        原实现 `softmax((qh @ kh.T) * inv)` 没有任何掩码 ⇒ ORV 是**双向全注意力**，
        而产线引擎与 PyTorch 参考都是**因果注意力**。
        后果：单层就能让 loss 与 PyTorch 差 9.6e-3，15 步后差 5.8e-2；
        更严重的是"能看到未来 token"，训练/评测口径都不对。
        """
        cache = getattr(self, "_mask_cache", None)
        if cache is not None and cache[0] == n:
            return cache[1]
        a = np.triu(np.full((n, n), -1e9, np.float32), k=1)
        t = T.tensor(a)
        self._mask_cache = (n, t)
        return t

    def fwd_attn(self, x, i, cos, sin):
        """层的前半段：`x + attention(ln1(x))`，**不含 MoE**。

        默认走堆叠实现（`attn_impl="stacked"`，实测单层前向快 3.66×）；
        `attn_impl="loop"` 可切回原来的逐头循环，用于对照与回退。
        """
        if getattr(self, "attn_impl", "stacked") == "loop":
            return self.fwd_attn_loop(x, i, cos, sin)
        return self.fwd_attn_stacked(x, i, cos, sin)

    def fwd_attn_loop(self, x, i, cos, sin):
        """**逐头循环**版（原始实现，保留做对照与回退）。

        为什么要拆出 attention 段（2026-11-10 实测）：MoE 的 `_combine_cuda` 有一次主机往返
        （D2H），捕获窗口内禁止 D2H，所以"整层一张图"不可能；而 attention 段没有 D2H，可以
        逐层捕获成图。拆分只是把同一批算子分成两个方法，算术不变。
        """
        L = self.layers[i]
        h = L["ln1"](x)
        acc = None
        inv = 1.0 / math.sqrt(self.DH)
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)
            sc = (qh @ kh.T) * inv
            pr = T.functional.softmax(sc + self._causal_mask(sc.shape[0]), axis=-1)
            part = L["o"][hh](pr @ vh)
            acc = part if acc is None else acc + part
        return x + acc

    def fwd_attn_stacked(self, x, i, cos, sin):
        """`fwd_attn` 的**堆叠实现**：把 16 个 head 合并成少数几次大 GEMM + 一次 SDPA。

        为什么快（根因）：逐头循环每层要 16 次 `Linear(D→DH)` + 16 次 `qh@kh.T` + 16 次 softmax
        + 16 次 `pr@vh` + 16 次 `Linear(DH→D)` ≈ 80+ 次**小** kernel，每个都填不满 V100。
        堆叠后每层变成 3 次大投影 GEMM + 1 次批量 SDPA + 1 次输出 GEMM。

        权重合并方式：每层每次 forward 用 `cat` 把 16 个 `[D,DH]` 拼成 `[D,D]`（约 4 MB，可忽略），
        这样反向会自动流回 16 个独立的 `Parameter`，**不需要改权重布局、不影响 .tg 加载**。

        数学与 `fwd_attn` 等价：rope 逐头相同（广播），causal 掩码与 `1/sqrt(dh)` 缩放由 SDPA 负责。
        """
        L = self.layers[i]
        h = L["ln1"](x)
        S = int(h.shape[0])
        NH, DH = self.NH, self.DH
        # 拼权重（反向经 cat 回到原来的 16 个 Parameter）
        # Linear(D→DH) 的权重是 [DH, D]：把 16 个按输出维拼起来得到 [NH*DH, D]，
        # 再转置成 [D, NH*DH] 供 `h @ W` 使用（等价于一次全宽投影）。
        Wq = T.cat([m.weight for m in L["q"]], dim=0).transpose(0, 1)
        Wk = T.cat([m.weight for m in L["k"]], dim=0).transpose(0, 1)
        Wv = T.cat([m.weight for m in L["v"]], dim=0).transpose(0, 1)
        # Linear(DH→D) 的权重是 [D, DH]，转置成 [DH, D] 后按 head 拼成 [NH*DH, D]；
        # `o_all @ Wo` 正好等于 16 个 head 各自 Linear 的**求和**。
        Wo = T.cat([m.weight.transpose(0, 1) for m in L["o"]], dim=0)
        q = (h @ Wq).reshape(S, NH, DH).permute(1, 0, 2)
        k = (h @ Wk).reshape(S, NH, DH).permute(1, 0, 2)
        v = (h @ Wv).reshape(S, NH, DH).permute(1, 0, 2)
        q = T._rope_composed(q, cos, sin)
        k = T._rope_composed(k, cos, sin)
        o = T.functional.scaled_dot_product_attention(q, k, v, causal=True)
        o = o.permute(1, 0, 2).reshape(S, self.D)
        return x + (o @ Wo)

    def moe_part(self, y, i):
        """层的后半段：`y + MoE(ln2(y))`（保持逐算子，不参与图捕获）。"""
        L = self.layers[i]
        return y + L["moe"](L["ln2"](y))

    def fwd(self, x, i, cos, sin):
        return self.moe_part(self.fwd_attn(x, i, cos, sin), i)

    def add_embed_grad(self, ids, g_x0=None):
        """把 `d(loss)/d(x0)` 散射回 tied 的 `head.weight.grad`（输入 embedding 那条路）。

        为什么需要（2026-11-09 实测）：训练循环里 `x0` 是从 numpy 拿的常量，
        tied embedding 只拿到"输出投影"一侧的梯度，**输入查表那一侧被整体丢掉**，
        实测 `head.weight.grad` 与 PyTorch 差 **34%**，也让 loss 曲线每步都偏。

        实现：`rt.emb_bwd` 做 scatter-add 得到 `dw[V, D]`，与现有 head 梯度相加。
        代价：每步一次 `[V,D]` 的 D2H/H2D（3.0B 模型约 134 MB）。先保证正确，再谈优化。
        """
        rt = T._cuda()
        g = self.last_input_grad if g_x0 is None else np.asarray(g_x0, np.float32)
        if g is None:
            return False
        ids = np.asarray(ids, np.int64)
        w = self.head.weight
        V, D = int(w.shape[0]), int(w.shape[1])
        # rt.emb_bwd 需要**设备缓冲**（numpy 传进去会 AttributeError: no attribute 'ptr'）
        g_buf = rt.h2d(np.ascontiguousarray(g, np.float32))
        # 内核按 int32 读 id（实测：传 int64 会把索引读错，梯度落到序列里没出现的 row 0）
        id_buf = rt.h2d(ids.astype(np.int32), dtype=np.int32)
        dw = np.asarray(rt.d2h(rt.emb_bwd(g_buf, id_buf, int(ids.size), D, (V, D))), np.float32)
        cur = getattr(w, "grad", None)
        if cur is None:
            w.grad = T.tensor(dw)
        else:
            prev = np.asarray(cur.to_numpy() if hasattr(cur, "to_numpy") else cur, np.float32)
            w.grad = T.tensor(prev + dw)
        return True

    def x0_from_ids(self, ids):
        """设备侧构造 `x0 = head.weight[ids]`（gather_rows），**不做 D2H**。

        返回 (x0_buf, id_buf)：id_buf 可复用给 `add_embed_grad_device`。
        原先训练循环用 `emb[ids]`（numpy 索引）再 h2d，每步一次 134 MB 级主机往返。
        """
        rt = T._cuda()
        w = self.head.weight
        V, D = int(w.shape[0]), int(w.shape[1])
        id_buf = rt.h2d(np.ascontiguousarray(np.asarray(ids, np.int32)), dtype=np.int32)
        x0 = rt.gather_rows(w._gpu, id_buf, V, D, int(len(ids)))
        return x0, id_buf

    def add_embed_grad_device(self, id_buf, g_buf):
        """设备版输入 embedding 梯度：`emb_bwd` 后与 `head.weight.grad` 在设备上相加。

        与 `add_embed_grad` 数学等价（同一个 scatter-add 内核），但**不 D2H/H2D**：
        原来每步要把 [V,D]（约 134 MB）拉回主机相加再传回去。
        """
        rt = T._cuda()
        w = self.head.weight
        V, D = int(w.shape[0]), int(w.shape[1])
        dw_buf = rt.emb_bwd(g_buf, id_buf, int(id_buf.n), D, (V, D))
        cur = getattr(w, "grad", None)
        if cur is None:
            w.grad = T.Tensor._from_gpu(dw_buf, requires_grad=False)
        else:
            w.grad = T.Tensor._from_gpu(rt.add(cur._gpu, dw_buf), requires_grad=False)
        return True

    def forward_all_dev(self, x0_buf, cos, sin, keep=None):
        """与 `forward_all` 相同，但第一层直接吃**设备缓冲**（省掉 x0 的 H2D）。

        `forward_all` 会把 x0 先 `np.asarray`（对设备缓冲无效），所以单独开这条路径。
        """
        rt = T._cuda()
        xs = [x0_buf]
        for i in range(self.NL):
            if i == 0:
                x = T.Tensor._from_gpu(xs[0], requires_grad=False)
            else:
                x = T.tensor(xs[-1])
            y = self.fwd(x, i, cos, sin)
            if keep is not None:
                keep(i, y)
            xs.append(rt.d2h(y._gpu) if getattr(y, "device", "cpu") == "cuda" else _np(y))
        return xs

    def forward_all(self, x0, cos, sin, keep=None):
        """不建图地跑完所有层，缓存每层输入。`keep=lambda i,x: None` 可用于逐步落盘。"""
        xs = [np.asarray(x0, np.float32) if not isinstance(x0, np.ndarray) else x0]
        for i in range(self.NL):
            y = self.fwd(T.tensor(xs[-1]), i, cos, sin)
            arr = np.asarray(_np(y), np.float32)
            xs.append(arr)
            if keep:
                keep(i, arr)
        return xs

    # ---------- 逐层反向 + step ----------
    def backward_and_step(self, xs, g_out, cos, sin, opts=None):
        """从最后一层往回：重算该层前向 → 对 (y·g) 求和反向 → 拿到该层梯度与上游梯度 → step。"""
        g = np.asarray(g_out, np.float32)
        stats = []
        for i in range(self.NL - 1, -1, -1):
            x_in = T.tensor(xs[i], requires_grad=True)
            y = self.fwd(x_in, i, cos, sin)
            gt = T.tensor(g)
            (y * gt).sum().backward()
            n_grad = sum(1 for p in self.params_of(i) if getattr(p, "grad", None) is not None)
            if x_in.grad is not None:
                g = np.asarray(_np(x_in.grad), np.float32)
                # i 走到 0 时，g 就是 d(loss)/d(x0)（对输入 embedding 的梯度）。
                # 存下来供 `add_embed_grad` 散射回 tied 的 head.weight.grad —— 之前训练
                # 因为 x0 是 numpy 常量，这条梯度被整体丢掉（实测 embed 梯度差 34%）。
                self.last_input_grad = g
            if opts is not None:
                opts[i].step()
                opts[i].zero_grad()
            for p in self.params_of(i):
                p.grad = None
            stats.append(dict(layer=i, with_grad=n_grad))
        return g, stats

    # ---------- LM head（tied embed + CE）----------
    def attach_head(self, sd, embed_key="token_embedding.weight", norm_key="final_norm.weight"):
        """挂上 final_norm 与 tied 输出权重（V2.7 没有独立 lm_head）。"""
        self.final_norm = RMSNorm(self.D)
        self.final_norm.load_state_dict_stream(iter([("weight", _as_np(sd[norm_key]))]))
        self.head = Linear(self.D, int(sd[embed_key].shape[0]), bias=False)
        self.head.load_state_dict_stream(iter([("weight", _as_np(sd[embed_key]))]))
        return self

    def head_params(self):
        return [p for p in (self.final_norm.weight, self.head.weight)
                if getattr(p, "requires_grad", False)]

    def loss_and_head_backward(self, x_last, targets, cos, sin):
        """算 CE loss，反传到第 0..N-1 层的上游梯度，并返回 (loss, 上游梯度 g)。

        注意：这里顺便把 head（final_norm/embed）的梯度算出来，调用方 step 完要清。
        """
        x_t = T.tensor(np.asarray(x_last, np.float32), requires_grad=True)
        logits = self.head(self.final_norm(x_t))
        loss = T.functional.cross_entropy(logits, np.asarray(targets, np.int64))
        loss.backward()
        g = np.asarray(_np(x_t.grad), np.float32) if x_t.grad is not None else None
        return loss, g

    def head_grads(self):
        return [getattr(p, "grad", None) for p in self.head_params()]


class TGState(object):
    """把 .TG（v3/v4 同容器）包装成 load_all 需要的 dict 接口。

    为什么需要：**推训一体**要求训练直接读产线那份 .tg，而不是中间 .pt。
    实现要点：
    * 用 
ead_tg 读一次头部/表（不进内存整块），张量按名字**惰性**取；
    * __getitem__ 返回 fp32 ndarray（bf16 源解码），主机峰值 = 单张量；
    * 实测：从 v4 .tg 取出的权重与 .pt **逐位相同**（11 个抽样张量 max|Δ|=0）。
    """

    def __init__(self, path):
        from tgai_go.read_tg import read_tg
        self.path = path
        self._info = read_tg(path)
        self._names = set()
        for e in self._info.get("table", []):
            nm = e.get("name")
            if nm:
                self._names.add(nm)

    def __contains__(self, name):
        return True if not self._names else (name in self._names)

    def __getitem__(self, name):
        from tgai_go.read_tg import load_tensor
        return load_tensor(self._info, name)

    def close(self):
        from tgai_go.read_tg import close_tg
        close_tg(self._info)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
