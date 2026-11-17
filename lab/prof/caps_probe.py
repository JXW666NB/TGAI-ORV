# -*- coding: utf-8 -*-
"""验证设备能力探测与精度策略解析（V100 上应得到 bf16 存储 + fp16 计算）。"""
import tgai_go.train as T

T.set_default_device("cuda")


def main():
    caps = T.device_caps()
    print("caps =", caps)

    pol = T.precision_policy()
    print("auto  ->", {k: pol[k] for k in ("storage", "compute", "storage_bytes")})

    for spec in (("fp32", None), (None, "fp32"), (None, "bf16"), ("fp16", "fp16")):
        p = T.precision_policy(storage=spec[0], compute=spec[1])
        print("storage=%-5s compute=%-5s -> storage=%-4s compute=%-4s"
              % (str(spec[0]), str(spec[1]), p["storage"], p["compute"]))

    # 非法输入必须报错（且消息里带修法）
    try:
        T.precision_policy(storage="int8")
        print("** 非法输入没有报错，是 bug **")
    except ValueError as e:
        print("非法输入已拒绝:", str(e)[:60], "...")


if __name__ == "__main__":
    main()
