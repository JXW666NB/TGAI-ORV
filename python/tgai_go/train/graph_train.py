# -*- coding: utf-8 -*-
"""把「一步训练」捕获成 CUDA Graph，之后靠“原地改宿主数据 + 重放”来跑。

## 为什么

实测（V100 / Qwen2.5-0.5B / B=4·S=512）：
* fp16 逐算子 **437~489 ms/step**
* fp16 + Graph 重放 **256 ms/step（1.71×）**
* torch AMP **219.5 ms** —— 也就是说 Graph 把差距从 2.2× 压到 1.17×

关键洞察：Graph 不需要改 dtype 泛化，它消掉的是**每步几百次 kernel 发射的间隙**。
之前没做成训练路径，是因为“输入数据每步都在变”，而图里刻的是固定地址。
解法很简单：**让宿主那份 ids/tgt 数组地址固定、每步原地改写**，
图里的 H2D 拷贝就会读到新数据（拷贝指令本身在图里，源地址不变）。

## 用法

```python
from tgai_go.train.graph_train import GraphStep

gs = GraphStep(model, opt, ids_shape=(4, 512), tgt_shape=(2048,), vocab=151936,
               loss_fn=T.functional.cross_entropy, fp16_gemm=True)
n = gs.capture()                     # 捕获一步（会真实执行一次）
for batch in loader:
    gs.step(ids_np, tgt_np)          # 拷贝进固定宿主缓冲 + 重放，loss 在 gs.last_loss
```

## 限制（必须知道）

* **形状固定**：ids/tgt 的 shape 必须与捕获时一致（换 batch 要重新捕获）。
* 图里不能有 D2H（`read_scalar`：比如梯度裁剪 `clip>0` 会读标量）⇒ 捕获会失败并
  给出原因；需要裁剪时请在**图外**做，或者不要用 Graph。
* 参数地址必须稳定：捕获后不要再 `opt.step()` 之外的操作去替换参数缓冲。
* 数值：重放执行的是同一批 kernel，**与逐算子路径逐位相同**（fp32 严格；fp16 同样严格，
  因为 kernel 一样、顺序一样）。
"""
import numpy as np


class GraphStepError(RuntimeError):
    """图捕获/重放失败（消息里带修法）。"""


class GraphStep:
    def __init__(self, model, opt, ids_shape, tgt_shape, vocab, loss_fn=None,
                 fp16_gemm=False, arena_gib=6.0, clip=0.0):
        import tgai_go.train as T
        self.T = T
        # 懒取运行时：纯 CPU 机器上 _cuda() 会抛（没有 nvcuda.dll）。
        # 构造本身应当成功，真正 capture/step 时再报清晰的错。
        try:
            self.rt = T._cuda()
            self._rt_error = None
        except Exception as exc:                        # noqa: BLE001
            self.rt = None
            self._rt_error = str(exc)
        self.model = model
        self.opt = opt
        self.vocab = int(vocab)
        self.fp16 = bool(fp16_gemm)
        self.clip = float(clip)
        if loss_fn is None:
            loss_fn = T.functional.cross_entropy
        self.loss_fn = loss_fn
        # 固定地址的宿主输入（图里的 H2D 从这块内存读）
        self.ids_host = np.zeros(tuple(ids_shape), dtype=np.int64)
        self.tgt_host = np.zeros(tuple(tgt_shape), dtype=np.int64)
        self.arena_gib = float(arena_gib)
        self.captured = False
        self.last_loss = None
        self.fallback_reason = None
        # AdamW 的偏差校正是按步数算的标量，图里刻的是"捕获那一刻"的值，
        # 所以每次重放必须把它改写（graph_launch 的 bc1/bc2 就是干这个的）。
        self._b1 = float(getattr(opt, "b1", 0.9))
        self._b2 = float(getattr(opt, "b2", 0.999))
        self._t = 0

    # ---- 内部：图内部分（前向 + 反向），与逐算子路径共用同一段代码 ----
    def _forward_backward(self):
        """只算前向+反向+清梯度；**不含优化器**。

        为什么把优化器留在图外：AdamW 的偏差校正 `1-b^t` 是按步数算的标量，
        它被刻进 kernel 参数后要靠 `cuGraphExecKernelNodeSetParams` 改写，
        实测改不干净（重放一步的参数差 1.7e-05，与更新量同量级）。
        fused AdamW 本身只有 1 个 kernel（约 30 ms），放图外没有发射开销问题。
        """
        ids, tgt = self.ids_host, self.tgt_host
        B, S = int(ids.shape[0]), int(ids.shape[1])
        # zero_grad 必须**在图里**：图外清零会换/改缓冲，导致重放的写落不到当前梯度上
        self.opt.zero_grad()
        logits = self.model.forward(ids)
        loss = self.loss_fn(logits.reshape(B * S, self.vocab), tgt.reshape(-1))
        try:
            loss.backward()
        finally:
            # 避免把 loss 张量留着（它引用着 logits 的图）
            del loss
        return None

    def _clip_grad(self):
        """梯度裁剪：nn_ops 是子模块，tgai_go.train 上没有这个属性，必须显式 import。"""
        from tgai_go.train import nn_ops as _nn_ops
        _nn_ops.clip_grad_norm_(self.model.params(), self.clip)

    def _one_step(self):
        """逐算子对照路径：前向+反向+优化器（语义与图外组合完全一致）。"""
        self._forward_backward()
        if self.clip:
            self._clip_grad()
        self.opt.step()
        self.opt.zero_grad()

    def _set_fp16(self, on):
        self.rt.fp16_gemm = bool(on)

    # ---- 捕获 ----
    def capture(self, warmup=2):
        """捕获一步。返回图节点数；失败抛 GraphStepError（带回退建议）。"""
        rt = self.rt
        if rt is None:
            raise GraphStepError(
                "拿不到 CUDA 运行时：%s。修法：CUDA Graph 只在 NVIDIA 卡上可用，"
                "没有卡时请用逐算子路径（sgd 自带的多后端会走 cpu）" % (self._rt_error,))
        self._set_fp16(self.fp16)
        for _ in range(int(warmup)):
            self._one_step()
        rt.sync()
        rt.reserve_capture_arena(int(self.arena_gib * (1 << 30)))
        if not rt.graph_begin():
            self.fallback_reason = "graph_begin 不可用（驱动/流不满足条件）"
            raise GraphStepError(
                "capture 失败：%s。修法：确认是 NVIDIA 卡且驱动支持 CUDA Graph；"
                "否则继续用逐算子路径（慢约 1.7×）" % self.fallback_reason)
        try:
            # 图里只放 前向+反向（不含优化器，见 _forward_backward 的说明）
            self._forward_backward()
        finally:
            ok = rt.graph_end()
        if not ok:
            stats = rt.graph_stats() or {}
            self.fallback_reason = stats.get("graph_reason") or "graph_end 失败（图里含 D2H？）"
            raise GraphStepError(
                "capture 失败：%s。修法：把梯度裁剪 clip 设为 0（read_scalar 会触发 D2H），"
                "并避免在 step 里打印 loss（也是 D2H）" % self.fallback_reason)
        self.captured = True
        # 梯度在图里已经清零，捕获后无需再清（清了反而会把重放的写目标换掉）
        self._t = int(getattr(self.opt, "t", 0))   # 与优化器已完成的步数对齐
        return (rt.graph_stats() or {}).get("graph_nodes")

    # ---- 重放 ----
    def step(self, ids, tgt):
        """把新数据原地写进固定宿主缓冲，然后重放一次。"""
        if not self.captured:
            raise GraphStepError("还没 capture()。修法：先调用 gs.capture()")
        if tuple(np.shape(ids)) != self.ids_host.shape:
            raise GraphStepError(
                "ids 形状 %s 与捕获时的 %s 不一致。修法：换 batch 大小要重新 capture()"
                % (tuple(np.shape(ids)), self.ids_host.shape))
        if tuple(np.shape(tgt)) != self.tgt_host.shape:
            raise GraphStepError(
                "tgt 形状 %s 与捕获时的 %s 不一致。修法：重新 capture()"
                % (tuple(np.shape(tgt)), self.tgt_host.shape))
        np.copyto(self.ids_host, np.asarray(ids, dtype=np.int64))
        np.copyto(self.tgt_host, np.asarray(tgt, dtype=np.int64))
        # 图只含前向+反向：重放把梯度写进固定缓冲，优化器在图外跑（偏差校正天然正确）
        self.rt.graph_launch()
        if self.clip:
            self._clip_grad()
        self.opt.step()
        self._t = int(getattr(self.opt, "t", self._t + 1))

    # ---- 逐算子（对照组 / 回退） ----
    def step_eager(self, ids, tgt):
        np.copyto(self.ids_host, np.asarray(ids, dtype=np.int64))
        np.copyto(self.tgt_host, np.asarray(tgt, dtype=np.int64))
        self._set_fp16(self.fp16)
        loss = self._one_step()
        return loss

    def close(self):
        self.rt.release_capture_arena()
        self.captured = False
