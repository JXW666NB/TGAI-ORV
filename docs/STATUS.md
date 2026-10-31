# 现状与证据（给接手的人）

> 这份文件回答三个问题：**现在到底能干什么、证据在哪、还差什么**。
> 判据与实验细节在 [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md)，
> 微调全部实验（含失败）在 [FINETUNE_RESULTS.md](FINETUNE_RESULTS.md)，
> 工程规则在 [ENGINEERING_STANDARD.md](ENGINEERING_STANDARD.md)。

**对外标签：可用（自用 / 小团队）** —— 未经 L3 全部达标，不升级。

---

## 一、能证明的（每条都有可重跑的证据）

| 能力 | 判据 | 实测 | 证据 |
| --- | --- | --- | --- |
| 训别人写的结构（LLaMA） | 与 PyTorch 孪生逐步对齐 | **1.16e-07** | `examples/llama_tiny.py`、`tests/test_llama_structure.py` |
| 加载真实开源权重做前向 | ≤1e-5 | Qwen2.5 **4.816e-06** / Qwen3 **4.294e-06** | `tools/qwen25_gpu_parity.py`、`tools/qwen3_gpu_parity.py` |
| 微调 Qwen2.5-0.5B | 留出集降 ≥5% | **−50.0%**（3.1148→1.5576，60 步） | FINETUNE_RESULTS R11 |
| 微调 Qwen3-0.6B | 同上 | **−53.2%**（3.2036→1.5000，60 步） | FINETUNE_RESULTS R10 |
| **模型级反向对齐** | 每个参数梯度 vs torch autograd | 全部参数相对误差 **≤1e-4**（含 q_norm / RoPE / GQA / tie 两路合并） | `tests/test_backward_parity.py` |
| **优化器对齐** | AdamW 逐步 vs `torch.optim.AdamW` | wd ∈ {0, 0.01, 0.1} 各 6 步 **≤1e-5** | `tests/test_adamw_parity.py` |
| 长跑可续训 | 断点续训轨迹不变 | 「3 步+存档+续 3 步」== 「连续 6 步」末 3 步 | `tests/test_resume.py` |
| 数据并行同步正确 | 权重广播 / 梯度求平均 | 2 进程 CPU 验证通过 | `tests/test_dist_cpu.py` |
| 引擎正确性自检 | 能把小 batch 过拟合到 ~0 | loss **1.1433 → 0.000021** | FINETUNE_RESULTS R8 |
| 工程门禁可自动跑 | 一条命令四道门禁 | **PASS**（约 90 秒） | `python tools/verify_all.py` |
| 打包可发布 | check_package | **PASS**（wheel 内无 BOM/CRLF、干净 venv 可 import） | `tools/check_package.py` |

测试总量：**160 项**（纯 CPU 可跑；CI 装 CPU torch，所以数值对齐用例是真门禁）。

---

## 二、还不能证明的（如实列出，别对外宣称）

| 项 | 状态 | 卡在哪 |
| --- | --- | --- |
| 大规模语料能训出**可用**模型 | ❌ 未证明 | `sft_big` 2000 条在 128 截断下只 −14.8% 且回退；已定位截断（27.6%）+ 学习率偏高，扫参 22:11 后跑 |
| 2 卡加速比 ≥1.7× | ❌ 缺硬件 | 只有单卡 A10；**同步正确性**已用 2 进程 CPU 验证 |
| 真 AMP（激活 bf16 存储） | ❌ 未做，但已量化 | 实测 **81 个内核 / 0 个支持 bf16**（matmul 走 cuBLAS 已支持）；S1 需 41 个，见 [AMP_PLAN.md](AMP_PLAN.md) |
| 24h 长稳 | 🟡 进行中 | 2026-10-30 22:11 启动，预计 22:11 结束（2,000,000+ 步，loss 0.088、显存零漂移） |
| 与 PyTorch 的速度/显存对比 | ❌ 未测 | 一直有长稳抢卡；0.5B 在 batch=8/seq=256 实测约 15 GB 峰值，但**没有对照组** |
| Qwen3 导出 `.TG` 部署 | ❌ 明确拒绝 | `.TG` 头无 `head_dim` 字段，Qwen3 的 128≠64；导出现在会报错并给修法 |

---

## 三、一条命令复验

```bash
python tools/verify_all.py          # 四道门禁，约 90 秒
python tools/verify_all.py --quick  # 跳过打包
```

单独看某项：

```bash
python -m pytest tests -q                                  # 160 项
python tools/check_package.py --build                      # 打包自检
python tools/qwen3_tg_parity.py                            # Qwen3 真权重对拍（需本机 .TG）
python tools/dataset_report.py --data x.jsonl --tokenizer t.json --seq-len 256   # 数据体检
```

---

## 四、已知欠账（都写在文档里，没有偷偷忽略）

| 欠账 | 影响 | 备注 |
| --- | --- | --- |
| 仓库根目录历史脚本仍是 CRLF | 不影响包 | `python/ tools/ tests/ docs/` 已规范化并有 CI 守着 |
| 显存效率未优化 | 0.5B 微调峰值 ~15 GB（PyTorch 同配置约 10~12 GB） | 引擎分配缓存偏保守，尚未动 |
| 31 个历史报告未入库 | 不影响包与门禁 | `docs/*_2026100x.md` / `*_2026101x.md` 是 2026-10-09~11 的早期报告，`.gitignore` 未覆盖但一直未提交；**不是这一轮产生的**，需要时单独整理入库 |
| 生成质量只有人工抽检 | 没有自动评测（BLEU/困惑度/裁判模型都没有） | 只有 3 条人工看过的样例 |
| 「学习率偏高」是推断 | gnorm 中位数 35~63 高于 clip=1.0 | 扫参会给结论 |

---

## 五、接手后先做什么

1. 跑 `python tools/verify_all.py`，全 PASS 再谈别的；
2. 要训新模型：先 `dataset_report.py` 体检数据（截断率 >20% 会直接拦）；
3. 要长跑：用 `--save` + `--resume`，存档自带 AdamW 动量；
4. 要改结构：先改 `checkpoints.py` 的映射（单一来源），再动 `Qwen25`；
5. 怀疑引擎有问题：**先过拟合一个小 batch**（FINETUNE_RESULTS R8），能立刻分清"引擎坏了"和"训得不够"。
