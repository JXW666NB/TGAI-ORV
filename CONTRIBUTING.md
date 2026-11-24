# 贡献指南 / Contributing

ORV 是一个**从零写起的训练引擎**（纯 Python + CUDA driver API + cuBLAS + NVRTC，不依赖 torch）。
我们欢迎贡献，但对**证据**的要求比对代码风格更严格。

## 三条铁律（违反的 PR 不会被合并）

1. **改前读源码，不要猜 API。** 我们的 API 名字和 PyTorch 像，但语义常不同（例如 `Tensor` 没有 `.item()`）。
2. **每个结论必须有实测数字。** 「跑通了」不等于「验证了」。性能类 PR 必须给出前后 A/B（同机、同口径、同步数）。
3. **未验证不发版。** 任何会改变训练数值的改动，必须有数值对拍（理想逐位，至少 ≤1e-6）与全套测试通过。

## 提交前自检清单

- [ ] `python -m pytest tests -q` 通过（当前基线：824 passed, 59 skipped）
- [ ] 新增/改动的算子有单测；CUDA 测试用 `skipif(no CUDA)` 守住
- [ ] 性能改动给出实测表格（配置、步数、AMP、batch 都要写清）
- [ ] 没把调试探针留在提交里（`rt.sync()` 包裹计时、临时 print）
- [ ] 改了公开 API 就同步更新 `README.md` 与 `skills/orv-engine-dev/SKILL.md`

## 性能测法的特别提醒（血换来的）

- **不要用「调用前后各 sync 一次」给单个 kernel 计时** —— 探针开销会把小 kernel 放大几十倍。
  我们曾据此把 `transpose2d` 算成 76 ms/步，实际端到端替换后步时不变。**请用 CUDA event。**
- **稳态要跑够步数**：PyTorch 前几步在内核自动调优，3 步测出的"稳态"能比 20 步慢 30%+。
- **对比要同口径**：梯度检查点、AMP、batch、数据分片都会显著改变结果。
  我们曾让 PyTorch 默认开着梯度检查点，把自己优势高估了 24~26%。

## 开发环境

```bash
git clone <repo> && cd orv
python -m pip install -e .
python -m pytest tests -q
```

CUDA 测试需要 NVIDIA 驱动 + NVRTC（`pip install tgai-orv[cuda]` 带 Linux 的 CUDA 轮子）。

## 交流

- 设计讨论：Discussion
- Bug / 性能回归：用 Issue 模板
- 插件：见 `PLUGINS.md`