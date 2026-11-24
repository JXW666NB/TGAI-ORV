# 回填表（把数字填进 ⬜ 里，连同 two 个 json 一起回传即可）

## 0. 环境（先跑 `python bench_fair.py --probe`，把它打印的几行贴这里）

```
torch:
cuda:
device:
cc:
SDPA mem_efficient 可用: ⬜
SDPA math 可用:          ⬜
ORV 版本:
native_available:
```

## 1. 公平对照（`bench_fair.py --bs 2,4 --iters 5 --json fair.json`）

### B=2

| 配置 | ORV s/step | ORV tok/s | ORV live | ORV reserved | ORV nvidia-smi | torch s/step | torch tok/s | torch live | torch reserved | torch nvidia-smi | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: fp32 + 低显存注意力 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| B: fp32 + 物化注意力 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| C: fp16 + 低显存注意力 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| D: fp16 + 物化注意力 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### B=4

| 配置 | ORV s/step | ORV tok/s | ORV live | ORV reserved | ORV nvidia-smi | torch s/step | torch tok/s | torch live | torch reserved | torch nvidia-smi | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| C | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| D | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

> R = ORV 步时 / torch 步时（<1 表示 ORV 更快）。

## 2. ORV 旋钮阶梯（`bench_ladder.py --bs 4 --iters 4 --json ladder.json`）

脚本末尾会打印"按峰值显存排序"和"按步时排序"两张小表，**直接把它们贴回来即可**。

另外单独回答三个问题：

1. `lowmem=True` 相对 `False`，峰值显存降了多少 GB？ ⬜
2. `bf16_act=True` 的峰值显存是**降**还是**涨**？（文档说这个开关可能反而涨，需实测） ⬜
3. `ce_chunk=2048` 相对 0，峰值显存降了多少？速度变化？ ⬜

## 3. 最想要的一份输出：显存归因（v3exp2 脚本里加 3 行即可）

```python
import os
os.environ["TG_POOL_DEBUG"] = "1"
...
print(rt.pool_stats())
rt.pool_dump_live(top=20)
```

把 `pool_dump_live` 的 20 行原样贴回来（它形如 `块大小 池偏移 命中率 ...`）。
我们要用它判断那多出来的 1.7 GB 是注意力分数、MoE dispatch 缓冲，还是激活副本。

```
⬜ 粘贴这里
```

## 4. 其它想说的

```
⬜
```
