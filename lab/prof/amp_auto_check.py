# -*- coding: utf-8 -*-
"""验证 amp_on(dtype='auto') 在 V100 上解析成 fp16，且 amp_off 两条路径都关。"""
import tgai_go.train as T

T.set_default_device("cuda")


def show(tag, st):
    print("%-16s on=%-5s fp16=%-5s bf16=%-5s" %
          (tag, st["on"], st["fp16"], st["bf16"]))


def main():
    print("caps:", T.device_caps())
    show("auto", T.amp_on())
    show("dtype=bf16", T.amp_on(dtype="bf16"))   # V100 无 bf16 算力 -> 应落到 fp16
    show("dtype=fp32", T.amp_on(dtype="fp32"))   # 应关闭
    T.amp_on(dtype="bf16")
    show("after off()", T.amp_off())             # 两条路径都应为 False


if __name__ == "__main__":
    main()
