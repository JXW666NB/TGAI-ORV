# GGUF → .TG 转换

把 llama.cpp 生态的 **GGUF** 权重转成 TGAI 引擎自己的 **.TG v3** 格式，纯 numpy、不需要 PyTorch。
转换后的文件可被 GO 引擎直接加载，也可作为训练起点。

```bash
python tools/gguf_to_tg.py --input qwen2.5-0.5b-fp16.gguf --output qwen25.TG --dtype fp16
```

实测（2026-10-30，本机 CPU）：

| 模型 | 输入 | 输出 | 耗时 | 权重数 |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-instruct | `qwen2.5-0.5b-fp16.gguf` 1.2 GB | `qwen25.TG` **944.2 MB** (FP16) | **16.6 s** | 290 |

## 转换做了什么

1. **解析 GGUF v3**：header → 元数据 KV → 张量表 → 权重数据；
2. **张量命名映射**（`map_weight_name`，见 `ARCH_CONFIG`）：
   `token_embd → token_embedding`、`blk.N.attn_q → blocks.N.attn.q_proj`、
   `blk.N.ffn_up → blocks.N.ffn.w2`、`blk.N.ffn_down → blocks.N.ffn.w3` ……
   `.TG` 的权重表**不存名字、只存 FNV-1a 哈希**，所以映射错一个字符，引擎就取不到那块权重。
3. **布局**：GGML 的 `dims=[ne0,ne1]`（ne0 最快变化）对应行主序 `[out, in]`，数据顺序本来就是对的，
   **只交换记录的 dims、不动字节**（早期版本错误地 `reshape().T`，导致 token_embedding/k_proj/v_proj 全乱）。
4. **RoPE 维度重排**：Qwen/LLaMA 的 rotate-half 布局 → 引擎原生交错式，
   **只作用于 Q/K 及其 bias**，其余张量原样。
5. **权重绑定**：`output.weight` 与 `token_embedding` 共享，转换器有意不重复写入（返回 `None`）。
6. **分词器**：词表 + BPE merges 写入 `.TG`；当词表超过 65535（Qwen2.5 是 151936）时，
   merge 对的 uint16 编码装不下，转换器会**额外导出 sidecar `tokenizer.json`**，
   聊天接口检测到 sidecar 就用 Python `tokenizers` 做编解码，绕过引擎内置 uint16 分词器的限制。

## 转换后怎么验证

```bash
python tools/verify_gguf_tg.py --gguf qwen2.5-0.5b-fp16.gguf --tg qwen25.TG
```

它会：
- 用**转换器同一套命名映射**（直接 import `map_weight_name`，绝不另写一套）算出哈希，
  按 `read_tg.load_tensor()` 的同一寻址方式把张量读回来 ——
  这等价于验证「引擎能不能按名字取到这块权重」；
- 对非 Q/K 张量做**逐位比对**；对 Q/K（被重排过）做**值集合比对**（排序后比较），
  证明是"重排"而不是"写坏"；
- 跳过 `output.weight`（权重绑定，本就不写）。

实测结果（Qwen2.5-0.5B，fp16）：

```
arch=qwen2（arch_type=3）
== .TG ==  version=3 dtype=1 vocab=151936 layers=24 张量表=290 词表=151936 merges=114236
   token_embedding.weight   逐位  shape=(151936, 896)  max|Δ|=0.000e+00
   blocks.0.ln1.weight      逐位  shape=(896,)         max|Δ|=0.000e+00
   blocks.0.ffn.w1.weight   逐位  shape=(896, 4864)    max|Δ|=0.000e+00
结果：一致 290 / 不符 0 / 有意跳过 1
结论：PASS
```

**fp16 转换是逐位无损的**（max|Δ| = 0）。

## 已踩过的两个坑（都修了，写在这里防复发）

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| `UnicodeDecodeError: unexpected end of data`（读 .TG 时） | token 的 UTF-8 被 `[:255]` 硬截断，**切在多字节字符中间** | 截断后 `decode('utf-8','ignore').encode()` 落在字符边界；读取端同时改成 `errors='replace'` |
| 校验脚本「291 个张量全部取不到」 | 我另写了一套命名映射，与转换器不一致 | 校验脚本直接 import 转换器的 `map_weight_name`，单一真源 |

## 支持的架构

`llama` / `mistral` / `qwen2` / `qwen3` / `gemma2` / `phi3`（见 `ARCH_CONFIG`）。
新增架构只需在 `ARCH_CONFIG` 里补一条映射，并在 `tests/test_gguf_mapping.py` 里加对应断言。

## 相关工具

| 工具 | 用途 |
| --- | --- |
| `tools/gguf_to_tg.py` | GGUF → .TG |
| `tools/verify_gguf_tg.py` | 转换正确性验证（哈希寻址 + 逐位比对） |
| `tools/inspect_gguf.py` | 只读查看 GGUF 元数据与张量清单 |
| `tools/tg_to_gguf.py` | 反向：.TG → GGUF |
