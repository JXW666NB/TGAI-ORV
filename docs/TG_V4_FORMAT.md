# `.TG` **V4**：从内部打破（推理格式 → 训练/推理统一格式）

> 决策：v3 是**为推理长的**（表里只有 FNV 哈希、无优化器态）；继续外挂 sidecar 只是"从外面打破"。
> V4 的目标是**自描述、可枚举、可 mmap、可承载训练态**，把"检查点"这件事做正。

## 1. v3 的四个根本问题

| # | 问题 | 后果 |
| --- | --- | --- |
| 1 | 权重表**只有哈希没有名字** | 不能枚举 ⇒ 优化器映射 / LoRA 挂载 / 部分加载 / diff 全部做不了 |
| 2 | 头是**定长 128B 的大结构体** | 加一个字段就要改所有读者（含 C 引擎）|
| 3 | **没有分区概念** | 训练态（动量/step/scaler/rng/数据游标）无处安放 |
| 4 | 无分区表/无校验 | 未知扩展只能靠猜位置（V2.7 的 `ext_off` 就因此误判过）|

## 2. V4 布局

```
┌─ Header 128B（定长，字段固定，只增不改语义）
│  magic "TGAI" | version=4 | header_bytes | flags
│  n_layers d_model n_heads head_dim d_ff vocab max_seq n_kv_heads
│  n_experts n_activated arch weight_dtype | rope_theta(f32) | pad/bos/eos/unk
│  tok_off tok_len | sect_off sect_cnt | body_sha256(前 8B)
├─ Section table（自描述，**新加分区不需要改头**）
│  [type:4][flags:4][offset:8][length:8][align:4][crc32:4] × N   （48B/条）
│  已知类型：TOK 词表 / NAM 名字表 / WGT 权重 / ARC 架构扩展
│            TRAIN 训练态 / OPT 优化器 / HASH 校验 / META 聊天模板
├─ TOK  词表（同 v3 编码，便于复用）
├─ NAM  名字表：n × {name_off:u32, name_len:u16, ndim:u8, dtype:u8,
│                   shape[8]:u32}  + 名字字符串池
├─ WGT  权重数据（每个张量按 align 对齐 → 可直接 mmap 单张量）
├─ ARC  架构扩展：per-layer MoE 配置、rope scaling、activation、bias 位、
│       `_qb_bias` 语义标记（v3 只能靠引擎注释猜）
├─ TRAIN  step/epoch/loss/best_ppl/scaler/rng_state/data_cursor
├─ OPT   优化器：每张量的 m/v（int8 分块 + 块 scale，或 fp32），与 NAM 对齐
└─ HASH  逐张量 sha256（可选；训练中防止静默读坏权重）
```

**四条设计原则**

1. **自描述**：读者按 section table 跳过不认识的块 ⇒ 以后加字段**不用改 C 引擎**；
2. **可枚举**：`NAM` 带名字 ⇒ 优化器/LoRA/部分加载/diff 全都成立；
3. **mmap 优先**：每个张量地址对齐 ⇒ 加载一个张量 = 一次切片（V2.7 实测：v3 mmap 解析 0.02s）；
4. **训练态与推理态可分离**：`OPT/TRAIN` 允许落**独立文件**（同布局、自己的头），
   推理文件因此仍然小（3B 权重 6 GB，不含 12~24 GB 优化器态）。

## 3. 迁移路径（零风险，分四步）

| 步 | 内容 | 对产线影响 |
| --- | --- | --- |
| **S1** | 只写 **Python** 侧 `tg4.py`（writer+reader）+ `export.save_tg4()`；`.TG` v3 输出不变 | 无 |
| **S2** | 用 V2.7 做**双向往返**：`.pt` → v4 → 读回，982 张量逐位比对；再由 v4 → v3，与现有产线文件比对 | 无 |
| **S3** | ORV 训练全面切 v4（名字 + `OPT` + `TRAIN`）；产线继续吃 v3 | 无 |
| **S4** | 给 C 引擎加 v4 读取（或只用 v4→v3 导出器供货） | 需引擎改动/验证 |

**S4 之前绝不把 v4 交给产线** —— 这是硬约束。

## 4. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 两套格式并行维护 | `v4→v3` 导出器做成**无损**（权重逐位一致），产线始终可回退 |
| 名字表与权重错位 | 名字表里带 `off/len/shape/dtype`，加载时**校验边界**；`body_sha256` + 逐张量 sha256 |
| 训练态写坏导致续训错乱 | `TRAIN/OPT` 各自 crc32；不一致就拒绝续训（宁可报错不可静默） |
| 头字段不够用 | section table 自描述 ⇒ 新需求=新 section，不动头 |

## 5. 与 sidecar 方案的关系

已经落地的 `<model>.TG.names.json`（v3 旁挂，实测 V2.7 982/982=100%）**保留**：
它是 V4 落地前的过渡，也是 V4 名字表的**交叉校验源**（同一批 FNV 哈希可比对）。