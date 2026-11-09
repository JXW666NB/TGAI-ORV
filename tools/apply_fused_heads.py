# -*- coding: utf-8 -*-
"""把 fused-heads（全宽投影）改造写进 `v27_train.py`，默认**关闭**。

设计（详见 docs/PLAN_FUSED_ATTN.md）：
  * 用 `getattr(self, "fused_heads", False)`，**不改 `__init__`**；调用方在 `load_all` 之前
    设置 `tr.fused_heads = True` 即可开启。
  * q/k/v/o 保持**列表**形态（fused 时长度为 1），因此 `modules_of` / `params_of` 不用改。
  * 默认 False ⇒ 现有行为逐位不变。

用法：
    python tools/apply_fused_heads.py --dry     # 只检查锚点是否都在
    python tools/apply_fused_heads.py           # 实际写入（已应用则跳过）
"""
import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "python", "tgai_go", "train", "v27_train.py")

OLD_BUILD = '''        L = dict(ln1=RMSNorm(D), ln2=RMSNorm(D), q=[], k=[], v=[], o=[])
        for _ in range(NH):
            L["q"].append(Linear(D, DH, bias=False))
            L["k"].append(Linear(D, DH, bias=False))
            L["v"].append(Linear(D, DH, bias=False))
            L["o"].append(Linear(DH, D, bias=False))'''

NEW_BUILD = '''        L = dict(ln1=RMSNorm(D), ln2=RMSNorm(D), q=[], k=[], v=[], o=[])
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
                L["o"].append(Linear(DH, D, bias=False))'''

OLD_LOAD = '''        for h in range(NH):
            sl = slice(h * DH, (h + 1) * DH)
            L["q"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw[sl]))]))
            L["k"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw[sl]))]))
            L["v"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw[sl]))]))
            L["o"][h].load_state_dict_stream(
                iter([("weight", np.ascontiguousarray(ow[:, sl]))]))
        del qw, kw, vw, ow'''

NEW_LOAD = '''        if getattr(self, "fused_heads", False):
            # 检查点本来就是全宽 [D, D]（见 load_layer 上方说明），整张装载即可
            L["q"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw))]))
            L["k"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw))]))
            L["v"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw))]))
            L["o"][0].load_state_dict_stream(iter([("weight", np.ascontiguousarray(ow))]))
        else:
            for h in range(NH):
                sl = slice(h * DH, (h + 1) * DH)
                L["q"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw[sl]))]))
                L["k"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw[sl]))]))
                L["v"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw[sl]))]))
                L["o"][h].load_state_dict_stream(
                    iter([("weight", np.ascontiguousarray(ow[:, sl]))]))
        del qw, kw, vw, ow'''

OLD_FWD = '''        acc = None
        inv = 1.0 / math.sqrt(self.DH)
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)'''

NEW_FWD = '''        acc = None
        inv = 1.0 / math.sqrt(self.DH)
        if getattr(self, "fused_heads", False):
            # 一次全宽 GEMM，再按头切片（切片与反向已由 test_slice_probe.py 验证）
            q_all = T._rope_composed(L["q"][0](h), T.tensor(np.tile(_np(cos), (1, self.NH))),
                                     T.tensor(np.tile(_np(sin), (1, self.NH))))
            k_all = T._rope_composed(L["k"][0](h), T.tensor(np.tile(_np(cos), (1, self.NH))),
                                     T.tensor(np.tile(_np(sin), (1, self.NH))))
            v_all = L["v"][0](h)
            for hh in range(self.NH):
                sl = slice(hh * self.DH, (hh + 1) * self.DH)
                pr = T.functional.softmax(
                    (q_all[:, sl] @ k_all[:, sl].T) * inv + self._causal_mask(q_all.shape[0]),
                    axis=-1)
                part = (pr @ v_all[:, sl]) @ L["o"][0].weight[:, sl].T
                acc = part if acc is None else acc + part
            y = x + acc
            return y + L["moe"](L["ln2"](y))
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)'''

PAIRS = [("_build_layer", OLD_BUILD, NEW_BUILD),
         ("load_layer", OLD_LOAD, NEW_LOAD),
         ("fwd", OLD_FWD, NEW_FWD)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    s = io.open(TARGET, encoding="utf-8").read()
    missing = [(n, s.count(o)) for n, o, _ in PAIRS if s.count(o) != 1]
    if missing:
        print("锚点不唯一/不存在，未改动：%s" % missing)
        return 2
    if all(new in s for _, _, new in PAIRS):
        print("看起来已应用（新代码已存在），跳过")
        return 0
    if args.dry:
        print("锚点全部唯一 ✔（--dry，未写入）")
        return 0
    for name, old, new in PAIRS:
        s = s.replace(old, new, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(s)
    print("已写入 3 处（fused_heads 默认关闭）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
