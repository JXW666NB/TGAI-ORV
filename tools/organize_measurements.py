# -*- coding: utf-8 -*-
"""把根目录的测量输出整理到 docs/measurements/，并生成索引。

为什么是「移动」而不是「删除」：这些是 `docs/ROADMAP_3B_PERF.md`、
`docs/MOE_MODERN_ARCH.md` 里数字的**原始证据**。证据要留，但不该躺在仓库根目录。
"""
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "docs", "measurements")

#: 文件 -> （类别, 说明）。说明只写我能从本会话确证的内容，猜的一律标「未核实」。
DESC = {
    "_m60.log": ("训练日志", "60M 级 LLaMA 试训（配置行可见 D=768 L=8）"),
    "_m60e.log": ("训练日志", "空文件"),
    "ad16.txt": ("显存", "AdamW fp16 动量逐层更新：空闲 20.47 GiB / 峰值 11.39 GiB，2.970B"),
    "bf.txt": ("AMP", "fp32 基线与 amp 状态的对照（bf16=True fp16=False）"),
    "cg.txt": ("失败排查", "容量因子 GPU 校验曾因 min 被遮蔽而抛错的 traceback"),
    "d2h.txt": ("性能", "去 d2h 同步的实测：4.183 vs 4.328 ms/层（结论：收益在噪声内）"),
    "eq.txt": ("失败排查", "分组未覆盖参数（embed/final_norm/lm_head）的告警"),
    "g512.txt": ("性能", "T=512 图反向整步：min 754 / mean 780 ms"),
    "gex.txt": ("失败排查", "符号探测：cublasSgemmGroupedBatched 存在、Ex 变体缺失"),
    "gfx.txt": ("失败排查", "分组 GEMM 返回 rc=7（unsupported）"),
    "gfx2.txt": ("失败排查", "分组 GEMM rc=7 后的回退路径"),
    "m16.txt": ("显存", "fp16 配置下 2550 张量 / 2.970B 的显存占用"),
    "mab.txt": ("性能", "MoE 单层 fp32 11.823 ms vs AMP 10.214 ms"),
    "mem.txt": ("显存", "28 层权重 11.39 GiB；空闲 20.47 / 31.86 GiB"),
    "mg.txt": ("失败排查", "8 次独立 GEMM vs 分组 GEMM 不可用的对照"),
    "mlb.txt": ("性能", "单层 MoE 前向 6.42 ms、前向+反向 13.12 ms"),
    "mpb.txt": ("性能", "单层 MoE 各原语耗时（gate/up/down 等）"),
    "mu16.txt": ("显存", "Muon fp16 动量下的显存与参数统计"),
    "oom.txt": ("关键证据", "PyTorch 28 层 MoE 1.997B 在 31.40 GiB 已分配后 OOM —— README 核心卖点的原始记录"),
    "p8.txt": ("失败排查", "某次切换的 traceback"),
    "q.txt": ("性能", "PyTorch 侧 seq=512 amp 108.1 ms / 峰值 10.65 GiB"),
    "r7.txt": ("失败排查", "分组 GEMM 按标量签名修正 argtypes 后仍 rc=7 的逐项探测"),
    "sp.txt": ("性能", "前向链 28 层 234.3 ms；MoE 反向 28 层 329.8 ms"),
    "sp2.txt": ("性能", "同上复测：237.0 / 332.4 ms"),
    "status_out.txt": ("数值", "各算子 CUDA-vs-CPU 相对差（sqrt 5.92e-08 等）"),
    "stk.txt": ("数值", "逐层梯度校验：前向 1.77e-07 / dL-dx 1.17e-07 / 权重梯度最大 1.46e-06"),
    "t3a.txt": ("失败排查", "分组未覆盖参数告警（版本 A）"),
    "t3b.txt": ("失败排查", "Muon 分组未覆盖告警（版本 B）"),
    "tc.txt": ("性能", "tensor core 探针：fp32@PEDANTIC 0.410 ms（20.9 TFLOP/s）"),
    "tgf.txt": ("测试", "pytest 单条结果：PASS test_grouped_helpers_exist"),
    "tmb.txt": ("失败排查", "Muon 分组未覆盖告警"),
    "tmf.txt": ("失败排查", "Muon 分组未覆盖告警（另一版）"),
}

GROUPS = ["关键证据", "性能", "显存", "数值", "AMP", "失败排查", "测试", "训练日志", "未分类"]


def main():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8").stdout.split("\n")
    root_txt = [f for f in files
                if f and "/" not in f and f.lower().endswith((".txt", ".log"))]
    if not root_txt:
        print("根目录已无 txt/log")
        return 0
    os.makedirs(DST, exist_ok=True)
    moved = []
    for f in root_txt:
        src = os.path.join(ROOT, f)
        if not os.path.isfile(src):
            continue
        shutil.move(src, os.path.join(DST, f))
        moved.append(f)
    print("已移动:", len(moved))

    # 生成索引
    lines = ["# 测量原始证据", "",
             "> 这些是仓库根目录原先散落的测量/排查输出，已整理到这里。",
             "> **为什么保留**：`docs/ROADMAP_3B_PERF.md`、`docs/MOE_MODERN_ARCH.md`、README",
             "> 里的数字都源自这些文件，删了就无法复核。",
             "> **说明只写能从文件内容直接确认的内容**；标「未核实」的是我没能确认的。", ""]
    for g in GROUPS:
        rows = [(f, DESC[f][1]) for f in sorted(moved) if DESC.get(f, ("未分类", ""))[0] == g]
        if not rows:
            continue
        lines.append("## %s" % g)
        lines.append("")
        lines.append("| 文件 | 内容 |")
        lines.append("| --- | --- |")
        for f, d in rows:
            lines.append("| `%s` | %s |" % (f, d))
        lines.append("")
    unknown = [f for f in sorted(moved) if f not in DESC]
    if unknown:
        lines.append("## 未分类")
        lines.append("")
        for f in unknown:
            lines.append("- `%s`（未核实）" % f)
        lines.append("")
    io.open(os.path.join(DST, "README.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines))
    print("索引已生成:", os.path.join("docs", "measurements", "README.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
