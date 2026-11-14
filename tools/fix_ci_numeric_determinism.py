# -*- coding: utf-8 -*-
"""修 CI 的数值抖动：锁 numpy 版本 + 数值测试单线程。

## 为什么

`test_step_layers_matches_step_within_fp32_noise` 在 Python 3.10 上是红的、3.12 上是绿的
（同一提交）。差异 2.4%，而测试注释里的实测值是 0.02% —— 差 120 倍，不像纯浮点噪声。

最可能的原因：CI 没锁 numpy，3.10 与 3.12 装到了**不同版本**（numpy 2.2 是最后一个支持
3.10 的线，3.12 会拿到更新的），BLAS 实现与累加顺序因此不同。

**修法不是放宽容差**（那会掩盖真问题），而是让数值环境可复现：
1. 锁 `numpy==2.2.6`（支持 3.10–3.13，两个 job 拿到同一份 BLAS）
2. 跑测试时把 BLAS 线程数固定为 1（多线程归约顺序是做数值对拍的头号不确定性来源）

## 用法

    python -X utf8 tools/fix_ci_numeric_determinism.py
"""
import io
import re

CI = ".github/workflows/ci.yml"
PIN = "numpy==2.2.6"


def main():
    s = io.open(CI, encoding="utf-8").read()
    if "numpy==2.2.6" in s:
        print("已经锁过 numpy，跳过")
        return 0

    # 1) 锁 numpy
    old = "python -m pip install numpy pytest build safetensors transformers"
    new = ("# numpy 锁版本：不同 Python 版本会装到不同 numpy，BLAS 累加顺序随之不同，\n"
           "          # 数值对拍会偶发越界（实测 3.10 与 3.12 不一致）。锁定后两边环境一致。\n"
           "          python -m pip install %s pytest build safetensors transformers" % PIN)
    if old not in s:
        raise SystemExit("没找到依赖安装行，未做修改")
    s = s.replace(old, new, 1)

    # 2) 测试步骤固定 BLAS 线程数
    old_run = "      - name: 单元测试（纯 CPU，不需要 GPU）\n        run: python -m pytest tests -q"
    new_run = ("      - name: 单元测试（纯 CPU，不需要 GPU）\n"
               "        # 数值对拍类测试必须单线程：多线程 BLAS 的归约顺序会导致\n"
               "        # 同一算法两次运行结果不同，从而偶发失败。\n"
               "        env:\n"
               "          OPENBLAS_NUM_THREADS: \"1\"\n"
               "          OMP_NUM_THREADS: \"1\"\n"
               "          MKL_NUM_THREADS: \"1\"\n"
               "        run: python -m pytest tests -q")
    if old_run not in s:
        raise SystemExit("没找到测试步骤，未做修改")
    s = s.replace(old_run, new_run, 1)

    io.open(CI, "w", encoding="utf-8", newline="\n").write(s)
    print("已锁 numpy==2.2.6 并把测试步骤改成单线程 BLAS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
