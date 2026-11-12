# -*- coding: utf-8 -*-
"""决定性测量：ORV 的反向峰值是否随层数累积？

## 为什么要这样测

`peak_live_bytes` 是**进程级高水位、从不重置**。在同一进程里连跑多组配置，
后一组会把前一组的峰值算进来 —— 我为此误判过一次（把权重累积当成激活累积）。
所以**每个配置必须单独起进程**。

## 判据

| 结果 | 含义 | 下一步 |
| --- | --- | --- |
| 峰值随 L **线性增长** | 反向没有逐层释放，多个层的中间量同时活着 | 查 autograd 的 grads 生命周期 |
| 峰值**与 L 无关**（≈恒定） | 行为正确，一次只有一层 | 只能降单层成本（lowmem / 原地复用）|

用法：python ckpt_scale.py
"""
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ckpt_scale.txt")


def main():
    rows = []
    for S in (2048,):
        for ck in (1, 0):
            for L in (2, 4, 6):
                cmd = [sys.executable, os.path.join(HERE, "ckpt_one.py"),
                       str(L), str(S), "2", str(ck)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                line = ""
                for ln in r.stdout.split("\n"):
                    if ln.startswith("@@RESULT@@"):
                        line = ln[len("@@RESULT@@"):]
                if not line:
                    rows.append(dict(L=L, S=S, ckpt=ck, error=(r.stderr or "")[-160:]))
                    continue
                rows.append(json.loads(line))

    rep = ["配置：D=1280 NH=10 dh=128 FF=5120 B=2（每配置独立进程）", ""]
    rep.append("| L | 检查点 | 权重 GiB | 前向活 GiB | **反向峰值 GiB** | 结束后 GiB |")
    rep.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if "error" in r:
            rep.append("| %s | %s | - | - | 失败 | %s |"
                       % (r["L"], "开" if r["ckpt"] else "关", r["error"][:60]))
            continue
        rep.append("| %d | %s | %.3f | %.3f | **%.3f** | %.3f |"
                   % (r["L"], "开" if r["ckpt"] else "关", r["weights_gib"],
                      r["fwd_live_gib"], r.get("bwd_peak_gib", -1),
                      r.get("after_live_gib", -1)))

    rep.append("")
    rep.append("判读：检查点开着时，**反向峰值应几乎不随 L 增长**；")
    rep.append("若随 L 线性增长 ⇒ 反向没有逐层释放。")
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
