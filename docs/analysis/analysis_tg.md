# `.TG` "加载正确性缺陷" 根因分析（只读分析 · 2026-11-01）

> 分析对象：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`（下称"本仓"）
> 分析方式：**只读**（read/grep + 在 `%TEMP%` 建临时 `.TG` 做探针），**未修改任何被 git 跟踪的文件**，未连远程。
> ⚠ 并发提示：本次分析进行中，工作区里有**其它进程**在改 AMP/分块 CE 相关文件
> （`docs/AMP_PLAN.md`、`tools/amp_readiness.py`、`tests/test_amp_readiness.py`、
> `tests/test_checkpoint_cuda.py`、`python/tgai_go/train/__init__.py`、`CHANGELOG.md`），
> 与本文结论无关，但 **`CHANGELOG.md` 因此整体下移 2 行**：本文引用的 `CHANGELOG.md:46-67`
> 对应现在的 `48-69`（内容未变，标题仍是"⚠ 更正（2026-11-01 晚）：先前「`.TG` 读错」的判断是**错的**"）。
> 其它被引用的文件（`read_tg.py`、`checkpoints.py`、`gguf_to_tg.py`、`tg_check.py`、`tg_audit.py`、
> `qwen.py` 等）在分析期间**未被改动**，行号可直接用。
> 证据分级：**[实测]** = 我在本机跑出来的；**[代码]** = 从源码直接读出的事实；**[仓内自述]** = 仓库文档/CHANGELOG 记录、我无法独立复核（本机没有官方 safetensors）；**[推断]** = 我的推理。

---

## 0. 结论速览

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | 那 6 个"离谱"的归一化权重是**读错**吗？ | **不是。** 它们是 Qwen2.5-0.5B 的真实值（`final_norm`≈7.1、`ln1`≈0.032）。本仓自己在 `CHANGELOG.md:46-60` 与 `docs/FINETUNE_RESULTS.md:507-513` 已把该判断**推翻**；`tools/tg_check.py:4-9` 把这个教训写进了 docstring | [仓内自述] + [实测]（读路径逐位无损，见 §5 E1） |
| 2 | `read_tg` / `load_tg` 有"读错 1-D 张量"的 bug 吗？ | **没有。** 结构自洽时 1-D 与 2-D 都是**逐位无损**；题面里的"6/49 离谱"是**误判**（判据错，不是加载错） | [实测] E1；[仓内自述] `CHANGELOG.md:55` |
| 3 | 那当前仓库里真正存在的"加载正确性缺陷"是什么？ | **两个**：<br>**D1（确定的代码缺陷）** `load_tg` 的"归一化体检"仍**硬抛 `CheckpointError`**（`checkpoints.py:646-657`），而文档说已降级为"只记录"。后果：**官方 Qwen2.5 的 `.TG` 默认加载不了**，`tools/tg_audit.py` 因此永远拿不到 TG 侧结果（它调用 `C.load_tg(path)` 默认参数）。<br>**D2（未收口的真问题）** `.TG` 与 safetensors 逐张量相关系数只有 **0.886~0.939**（`v_proj` 0.939，而 `v_proj` **不参与** RoPE 重排）⇒ 这是一份"同模型、被扰动"的副本，CE 8.30 vs 3.66 由此而来。**问题不在读端，在"这份 `.TG` 是从哪个源/哪个版本转出来的"** | D1 [代码] `checkpoints.py:646`、`tg_audit.py:78-79`、`CHANGELOG.md:61`、`FINETUNE_RESULTS.md:512`；D2 [仓内自述] `CHANGELOG.md:65-67` |
| 4 | 格式层面为什么这类问题**查不出来** | 表项**没有 name、没有内容指纹**（`tg_patch.py:34-40` 是权威字段表）。于是"读错"与"源错"在容器层**不可区分**，只能靠外部对拍 | [代码] §4 |

> **一句话根因**：这不是"加载器读错"的缺陷，而是**判据错**（用"归一化 γ≈1"这种常识阈值去判 `.TG` 读错）+ **格式缺自描述元数据**（无 name、无内容摘要）导致"读错 / 源错 / 判据错"三者无法区分。真问题（r≈0.9 的扰动副本）至今没有收口。

---

## 1. Q1：`read_tg` 怎么把"名字"映射到表项？会不会错位（尤其 1-D）？

### 1.1 名字不在文件里 —— 名字来自**读端硬编码清单**

- 表项只有 7 个字段，**没有任何名字/字符串**：
  `read_tg.py:53-60` 按 52 字节步长解析 `{hash, off, len, dtype, ndim, shape[4], scale}`；
  权威字段表见 `tools/tg_patch.py:34-40`（`0 u64 hash / 8 u64 offset / 16 u64 length / 24 u32 dtype / 28 u32 ndim / 32 u32 shape[4] / 48 f32 scale`）。
- 查找是**线性扫表 + 哈希相等**（`read_tg.py:86-95`）：

```python
def load_tensor(info, name):        # read_tg.py:86
    want = fnv1a64(name)            # 名字 → FNV-1a 64
    for e in info["table"]:         # 8592: 线性扫，取**第一个**命中
        if e["hash"] == want:
            start = info["data_off"] + e["off"]
            buf = info["raw"][start:start + e["len"]]
            return np.frombuffer(buf, dtype=np_t).reshape(e["shape"]).astype(np.float32)
    return None                     # 8595: 查不到 → None（不抛错）
```

- 名字的**唯一来源是读端代码里写死的那张清单**：`checkpoints.py:573-621`（`token_embedding.weight`、`final_norm.weight`、`blocks.{i}.ln1/ln2/attn.q_proj/... /ffn.w1..w3`，共 `24×12+2 = 290` 个，与实测 290 项吻合）。写端对称的是 `gguf_to_tg.py:1228-1328`（`map_weight_name`）+ `1462` 的 `'hash': fnv1a_64(mapped)`。
- 本仓自己也把这件事写进了文档：`tools/verify_gguf_tg.py:4-6`「`.TG` 的权重表**不存名字**，只存 FNV-1a 哈希 ⇒ 必须用转换器同一套命名映射」；`docs/GGUF_TO_TG.md:22` 同义。
- HF 侧是另一套名字，由 `checkpoints.py:304-319` 的正则表转成同一套（`model.norm.weight → final_norm.weight` 等）。

### 1.2 三种真实的"错位"失效模式（代码里都能复现）

| 模式 | 机制 | 后果 | 实测 |
|---|---|---|---|
| **M1 查不到 → 静默跳过** | `load_tensor` 返回 `None`，`load_tg` 在 `612-614` 直接 `continue`（注释都没有） | 缺张量**不报错**；`fix_linear_layout` 只遍历**已存在**的键（`509 for key in list(out)`），所以缺的层根本不会被发现。直到模型层 `qwen.py:117/119/154` 硬索引 `self.p[...]` 才 `KeyError` | E5/E3 相关路径已验证 |
| **M2 同名重复 → 取第一条** | `read_tg.py:89-90` 第一个命中即返回；表里允许两条同 hash 的项 | 静默取到**另一条**数据；`shape/len` 全一致，读端无法分辨 | [实测] E3：两条同 hash 项，永远返回 `off=0` 那条 |
| **M3 同长度错张量 → 完全静默** | 若 `off` 指到别的张量中间，`len` 与 `shape` 仍自洽 ⇒ `reshape` 不报错 | 拿到"合理量级"的错误数据 | [实测] E4：`ln1` 指进 `q_proj` 中间，读成功、无报错、`σ=0.06` |

### 1.3 为什么 1-D 更容易中招（这是提问里的关键点，且确实成立）

1. **1-D 完全没有形状校验**。`_expect_linear_shapes`（`checkpoints.py:477-493`）只列了 8 个 **2-D** 线性层；`fix_linear_layout`（`496-532`）只对命中这张表的 key 做 `[out,in]↔[in,out]` 判定/转置。1-D 张量**一个都不在表里** ⇒ 布局/形状类护栏对 1-D 全部失效。
2. **1-D 的唯一"护栏"是统计判据** `norm_sanity`（`535-561`），而它判的是"像不像 γ"（均值 ∈ [0.2, 3]、极值 >-2 且 <10），**不是格式正确性**。判据本身错了 → 真值被当成坏值（本 case 就是），坏值也可能落在阈值内（`blocks.0.ln1` 范围 [-0.311,0.360]、均值 0.033 —— 一个被写坏的向量也完全可能长这样）。
3. **动态维度靠反推**：`DH = q0.shape[0] // NH`（`checkpoints.py:577`，注释自认"头里没有 head_dim，只能从 q_proj 行数推"）。q_proj 是**方阵**（896×896）时，`fix_linear_layout` 的"转置判定"也失效（`got == want` 与 `got == want[::-1]` 同时成立，`519-524` 只走 `continue`）⇒ 方阵布局错**不可检测**。
4. **长度自洽 ≠ 正确**：只要 `len == prod(shape)*dtype_size`，任何内容都能通过 `reshape`（我在 E5 里用 `len=4*64 + dtype=fp16 + shape=[64]` 才算触发 `ValueError`）。所以"`len` 与 shape 自洽"这条实测事实，**只能排除长度错，不能排除内容错**。

### 1.4 对这份文件的判断

- 名字错位（M1）**可以排除**：290 个期望名字若有一个查不到，`qwen.py` 的硬索引会 `KeyError`，而两边都能前向出 CE。
- 同名重复/区间重叠（M2）**可以排除**：[仓内自述] `CHANGELOG.md:55` 已用 `tg_check` 验过「290 hash 全唯一、零重叠、零空洞、不越界」；我另给了独立复核命令（§6 Step 0/1）。
- **剩下唯一可能的就是"字节内容本身"** —— 见 §3 H0/H1/H2。

---

## 2. Q2：`hash` 是什么算法？读取时校验了吗？加校验能不能定位坏张量？

### 2.1 算法：FNV-1a 64（对**名字字符串**，不是数据）

```python
def fnv1a64(name):                       # read_tg.py:18-23
    h = 0xCBF29CE484222325               # offset basis
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF   # prime
    return h
```

同一算法在本仓有 **7 份独立实现**（必须逐字节一致，否则全部查不到）：

| 位置 | 用途 |
|---|---|
| `python/tgai_go/read_tg.py:18-23` | 读端查表 |
| `python/tgai_go/pack.py:35-40` | 打包器写表（`pack.py:145`） |
| `tools/gguf_to_tg.py:177-183` | GGUF→TG 写表（`1462`） |
| `tools/tgai_convert.py:91` | `.pt`→TG 写表（`781`） |
| `tools/tg_patch.py:53-58` | 注入工具（`117-126 find_by_name`） |
| `src/tgai_engine.c:1057`（`find_weight` 在 `1388`） | C 引擎按名取权重 |
| `embedded/TGAI_GO/src/TGAI_GO.cpp:219`（`451`） | 嵌入式同款 |

**关键点：这是"名字哈希"，不是"内容校验和"。** 它与张量的字节没有任何关系，因此：

- 它能回答的问题只有一个：「这个名字在表里存在吗？」；
- 它**不能**回答「这块字节是不是我想要的字节」「是不是被写坏了」「是不是来自另一个版本」。

### 2.2 读取时**零校验**（我逐项验过）

`read_tg` / `load_tensor` 全部**不做**下列任何一项：

| 应有检查 | 现状 | 实测 |
|---|---|---|
| 表项是否越界（`data_off+off+len <= 文件大小`） | 无。只有 `raw[start:start+len]` 短读，短读后靠 `reshape` **偶然**报错 | E5：`off` 越 EOF → `ValueError: cannot reshape array of size 0`（是被动报错，不是主动检查） |
| 区间是否重叠 | 无（外挂 `tools/tg_check.py:32-39` 才有） | E2：`gap=0 overlap=0` 但读端不检查 |
| hash 是否重复 | 无（`tg_check.py:27-30` 才有） | E3：重复项静默取第一条 |
| 是否有人读 `header.weight_data_len` | **完全没有**（`read_tg.py:96` 只用 `data_off`） | — |
| 期望名字是否都命中 / 是否有"孤儿表项" | 无。`load_tg` 不统计加载了多少张量（无 `assert len(weights)==290`），缺项直接 `continue`（`612-614`） | — |
| 内容摘要（crc32/sha256/crc64） | **格式里根本没有这个字段**（52 字节表项被 `hash/off/len/dtype/ndim/shape[4]/scale` 占满） | `tg_patch.py:34-40` |

### 2.3 所以：加"校验"能不能直接定位坏张量？

- **加名字哈希校验 ⇒ 不能。** 名字哈希只能发现"名字对不上"（而本 case 名字全对上了）。它连"字节被写坏"这个概念都不表达。
- **加内容指纹 ⇒ 能，而且正好切中要害。** 建议两种不改坏 v3 兼容性的做法：
  1. **sidecar**（最快落地）：`X.TG.manifest.json` 存 `{name: {offset, length, crc32, sha256_16}}`，工具链（`tg_check`/`tg_audit`/`verify_gguf_tg`）读它做逐张量比对。缺点：不随文件走，易丢。
  2. **扩展段**（格式内、向后兼容）：`.TG` 已有 TLV 扩展机制（`write_chat_meta.py:158-201` 的 `extension_off` 负偏移编码、`tgai_convert.py:528-571` 的 `build_v3_ext`，`type=1` 是 V3 配置）。新增 `type=2 = per-tensor digest`：`{hash(u64), crc32(u32), len(u32)}` 数组。老引擎/老读端完全不读（`read_tg.py:63-73` 只认 `CHAT_MAGIC`，其它 `ext` 段会被当 chat meta 解不出而忽略——所以**要新增一个独立 magic 或 version 分支**，否则 `read_tg` 会 `assert m == CHAT_MAGIC` 直接炸，这点必须注意）。
- **但请注意**：内容指纹只能证明"文件里的字节 ≠ 源里的字节"，**证明不了"源本身是错的"**。本 case 恰恰是"字节 == 源（自洽）但源与 safetensors 不同"（r≈0.9）。所以指纹是**必要但不充分**的，必须同时保留 `verify_gguf_tg.py` 那条"对源逐张量比"的链。

---

## 3. Q3：≥4 个假设，按可能性排序，每个给一条能证伪的检查

> 前提数据（题面 + 仓内）：2-D 统计正常（embed 0.0152 / w1 0.0230 / q_proj 0.0667）；`final_norm` 均值 +7.139、范围 [-0.011, 16.75]；`blocks.0.ln1` 均值 0.033、范围 [-0.311, 0.360]；`blocks.3.ln2` 均值 2.612；49 个归一化里 6 个"离谱"；CE 8.30 vs 3.66；逐张量 r = 0.886~0.939（`v_proj` 0.939）。

### H0（最可能，已被本仓推翻原判断支持）：**根本不是加载缺陷，是判据错**
- 内容：`final_norm≈7.1`、`ln1≈0.032` 是 Qwen2.5-0.5B 的真实值；`norm_sanity` 的"γ 应≈1"是错的常识。
- **证伪检查（一条命令）**：
  ```python
  from safetensors.numpy import load_file
  w = load_file(r"<Qwen2.5-0.5B-Instruct>\model.safetensors")
  print(w["model.norm.weight"].mean(), w["model.layers.0.input_layernorm.weight"].mean())
  # 期望 ≈ 7.1 与 ≈ 0.032；若得到 ≈1.0 与 ≈1.0，则 H0 被证伪（真读错了）
  ```
- 现状：本机**没有**官方 safetensors（我搜过 `D:\DSH-Workspace`、`D:\tgai_nlp`、HF cache 均为空），所以只能引 [仓内自述] `CHANGELOG.md:56-57`、`FINETUNE_RESULTS.md:510`。**这是本报告唯一无法独立复核的关键事实，建议优先补这一条。**

### H1：这份 `.TG` 是从**量化过的 GGUF**（Q8_0/Q4_K/Q4_0）转来的 ⇒ 值域被扰动
- 为什么像：r 在**所有**张量上都 <1（含不参与 RoPE 重排的 `v_proj`），且容器自洽、名字全对 —— 这是"值被压缩过"的典型形状。
- **证伪检查**：
  ```bash
  python tools/inspect_gguf.py <源>.gguf      # 看每个 tensor 的 ggml dtype
  # 若全部是 F16/F32 ⇒ H1 倒（量化不是来源）；若 2-D 是 Q4_K/Q8_0 ⇒ H1 成立
  ```
- **定量旁证（重要）**：量化噪声有硬上界。均匀量化步长 `q` 时 `std(Δ) ≈ q/√12`，`r ≈ σ/√(σ²+q²/12)`。
  - Q8_0（127 级）⇒ `r ≈ 0.9997`；Q4_K ⟹ `r ≈ 0.999`；即使 Q4_0 ⟹ `r ≈ 0.99`。
  - 实测 r = 0.886~0.939 对应 `std(Δ) ≈ 0.48σ`，**比任何 4-bit 量化大 4~5 倍**。
  ⇒ **[推断] 单纯量化解释不了 r=0.9**，除非反量化写错了（见 H3）。这条数值判据请在 §6 Step 2 里先算 `std(Δ)/std(w)` 再下结论。

### H2：`.TG` 与 safetensors **不是同一份权重**（base↔instruct、合并、或多训了一段）
- 为什么像：r 0.89~0.94 正是"强微调/合并后的同架构模型"的典型相关度；容器完全自洽也符合"源就是另一个合法模型"。
- **证伪检查**：算 `Δ = w_tg − w_ref` 的**结构**：
  ```python
  d = w_tg - w_ref
  print(d.mean(), d.std()/w_ref.std(), np.abs(d).max()/np.abs(w_ref).max())
  # 量化/反量化错：d 近零均值、|d| 有明确上界(≈q/2)、且 d 与 w 无关、lag=32 自相关有尖峰
  # 换模型：      d 近似"两个高斯之差"，std(d)/std(w) ≈ 0.5~1.4，|d|max 与 max|w| 同量级
  ```
  再加一条**分层扫描**：不同 checkpoint 的差异通常随深度/模块变化明显（embed 与 q/k 差异不一样），而系统性的格式/精度错在**所有**张量上比例一致。若 `std(Δ)/std(w)` 在各张量上的离散度 < 20% ⇒ 更像系统性问题（H1/H3）；> 2× ⇒ 更像 H2。

### H3：转换器的**反量化/量化实现**有 bug（scale 用错/二次应用）
- 为什么像：`tools/gguf_to_tg.py:724+` 的 `_dequant_q4_k` 是**手写的 6-bit scale/min 位拆包**（`scales[12]` 里 8×6bit scale + 8×6bit min），这种代码最容易"整体偏一个比例"；一旦 scale 被多乘/少乘一个常数，值域被整体缩放，r 会明显 <1 而**所有张量同时中招**。同类风险点还有 `_dequant_q8_0/q4_0/q5_0/q5_1`（`563-721`）与 `tg_requantize.py:127-163` 的另一套实现。
- **证伪检查**：用**第三方参考**（`gguf-py` 的 `dequantize`，或 `tools/inspect_gguf.py` 的既有路径）反量化同一个 Q4_K 张量，与 `.TG` 里对应块逐元素比：
  ```bash
  python - <<'PY'
  import gguf, numpy as np
  r = gguf.GGUFReader(r"<源>.gguf")
  t = [x for x in r.tensors if x.name=="blk.0.ffn_down.weight"][0]
  ref = gguf.quants.dequantize(t.data, t.tensor_type)   # 官方参考
  # 与 .TG 里 blocks.0.ffn.w3.weight 比：rel = max|Δ|/max|ref|
  PY
  ```
  若 `rel < 1e-6` ⇒ H3 倒（转换器没错，问题在源）；若 `rel ≈ 0.1` 且 **Δ 在每 256 元素块内呈常数比例** ⇒ H3 成立（scale 用错）。

### H4：容器结构错位（off/len 写坏、重复项、越界）
- **已被排除** [仓内自述] `CHANGELOG.md:55`；但给一条**独立复核**（不依赖他们的结论）：
  ```bash
  python tools/tg_check.py <X>.TG      # 期望 exit 0 / "PASS"
  ```
  外加我 §6 Step 1 的"名字覆盖双射"（290 个期望名字 ↔ 290 个 hash，必须一一对应；有孤儿/缺失就说明映射错位）。

### H5（原假设，已被我的实测排除）：字节被以错误 dtype 解释（fp32/bf16/int8 当成 fp16）
- 我把 6 种误读都跑了一遍（§5 E6）：fp32 字节当 fp16 → **出现 inf/nan**（896 个里 20 个）且 mean≈1.5；bf16 当 fp16 → mean 1.873（全是 1.87±0.01）；Q8_0/Q4_0 载荷当 fp16 → inf/nan；文本当 fp16 → mean 3444。
- **证伪检查**：实测签名是"**全有限**、mean +7.139、min −0.011、max +16.75"。只要统计里**没有 inf/nan**，就能排除"fp32/量化载荷被标成 fp16"这一整类（因为 fp32 低 16 位当 fp16 必然有 ~2~17% 的指数位全 1 → inf/nan）。

### H6（历史假设，用于解释"为什么当初会误判"）：读端把 1-D 做了布局变换
- `_tg_depermute`（`checkpoints.py:464-474`）会把 `q_norm/k_norm` 按 head 重排，`load_tg:609` 传 `nh=1`；Qwen2.5 **没有** q_norm/k_norm（`has_qk_norm = load_tensor(...) is not None`，`578` → False），所以这条**不适用**。而且重排只改顺序、不改均值 ⇒ 不可能把 1.0 变成 7.139。
- **证伪检查**：`python -c "from tgai_go.read_tg import *; i=read_tg(p); a=load_tensor(i,'final_norm.weight'); print(a.mean(), a.min(), a.max())"` —— 若与 `checkpoints.load_tg` 出来的数一致，说明 `load_tg` 没做任何变换（正常路径本就该一致）。

---

## 4. Q4：为什么 PyTorch/HF 不会遇到，而我们遇到了 —— 我们缺哪些元数据

### 4.1 两边的组织方式对比

| 维度 | safetensors（HF） | `.TG` v3（本仓） |
|---|---|---|
| 名字 | **JSON header 的键就是参数名**（`model.layers.0.input_layernorm.weight`） | **完全没有**（只有 64-bit FNV-1a 名字哈希） |
| 每张量元数据 | `{dtype, shape, data_offsets:[begin,end]}`，**绝对偏移、显式 end** | `{hash, off(相对数据区), len, dtype, ndim, shape[4], scale}` |
| 偏移语义 | `data_offsets` 是文件绝对偏移，天然可校验 | `off` 相对 `header.weight_data_off`，**读端不加校验** |
| 名字映射 | 无（名字本身就是身份） | **读端硬编码 + 写端 `map_weight_name` 两套知识必须完全一致**（`docs/GGUF_TO_TG.md:65` 记录过"我另写一套映射 ⇒ 291 个张量全取不到"） |
| 布局约定 | shape 就是真实形状，无隐含约定 | 有隐含约定：`[out,in]`（`checkpoints.py:477-493`）、Q/K 被 RoPE 预置换（`gguf_to_tg.py:1449`）、`dims[0]!=dims[1]` 才交换记录（`1443-1444`） |
| 内容校验 | 官方无（但有 `__metadata__` 可放任意摘要，生态里常放 hash） | 无，且**没有空位**（52B 已占满） |
| 结构参数 | `config.json` 独立文件（`head_dim`、`tie_word_embeddings` 等**显式**） | header 里没有 `head_dim`，读端**反推**（`checkpoints.py:577`）；tie 靠"有没有 lm_head"**猜**（`598-600`） |

本仓读 safetensors 的代码正好印证了差别：`checkpoints.py:331-360` 用 `safetensors.numpy.load_file`（键=名字），再用 `_HF_PATTERNS`（`304-319`）映射；`364-390` 读 `config.json` 拿 `head_dim`/`rope_scaling`/`tie`。**PyTorch 之所以"不会遇到"，不是因为算法强，而是因为名字是数据的一部分、形状是权威声明、结构参数在独立 config 里显式给出。**

### 4.2 我们**缺失的元数据**（按危害排序）

1. **`name` 字符串**（致命）。没有它，"读错"与"名字对错"不可区分；任何映射偏差只能静默变成 `None` 或"别人的张量"。理论上可用名字哈希反查，但**哈希不可逆**，只能靠读端猜一张名字清单。
2. **内容指纹**（致命）。没有它，无法回答"文件字节 == 源字节？"；r≈0.9 的调查因此卡住（`CHANGELOG.md:65-67` 自认"源头在哪个转换器仍待查"）。
3. **布局/变换声明**（高）。`transposed: bool`、`rope_permuted: bool`（每张量一位即可）。现在只能靠形状猜，且**方阵不可判**（`checkpoints.py:519-524`），Q/K 预置换只写在转换器注释里（`gguf_to_tg.py:1125-1137`）——**文件本身不声明**，读端只能靠"猜 + 事后还原"（`_tg_depermute`）。
4. **结构参数进 header**（高）。`head_dim`、`rms_norm_eps`、`rope_scaling`、`tie_embeddings`。现在 `DH` 反推、`eps` 写死 `1e-6`（`checkpoints.py:627`）、`rope_scaling` 完全没有（`load_tg` 不读，只有 safetensors 路径读 `checkpoints.py:388`）。**同一份权重换 `eps`/`scaling` 就是另一个模型，且不报错。**
5. **量化块布局版本**（中）。`dtype=4/5`（Q8_0/Q4_0）的 block 结构、`scale` 是 f32 还是 u32 重解释（`tg_requantize.py:158-162` 与 `dequant_to_fp32` 的 `scale_u32` 就是踩过这个坑的痕迹），格式里没有版本位。
6. **对齐/填充声明**（低）。`gguf_to_tg.py:1492-1504` 只对齐段起点，张量之间**不填充**；若将来加对齐，`off/len` 的语义需要显式声明（现在隐含"紧邻"）。

---

## 5. 我在本机做的只读实验（`%TEMP%\tg_diag\diag2.py`，不碰仓库）

用本仓**自己的** `tgai_go.pack.TGWriter` 造合法 `.TG`、再用 `read_tg/load_tensor` 读回，并手工拼畸形表。

| 编号 | 目的 | 结果 |
|---|---|---|
| **E1** | 合法 `.TG` 的 1-D/2-D 是否逐位无损 | **26/26 张量 mismatch=0**；`final_norm` mean 1.0036、`ln1` 0.9951、`ln2` 0.9980 ⇒ **读路径对 1-D 完全正确**，缺陷不在读端 |
| **E2** | 读端有无结构自检 | 表 26 项、hash 唯一 26、`gap=0 overlap=0`；但**读端不检查**这些（只有外挂 `tg_check` 检查） |
| **E3** | 同名重复项取谁 | 永远取**第一条**（`off=0`，mean 1.0004），第二条（全 7.139）永远读不到，**无任何报错** |
| **E4** | `off` 指进别的张量 | 读成功无报错，`mean +0.0071 / min -0.1384 / max +0.1127 / σ=0.0604` —— **量级是"权重"而不是 7.1** ⇒ 与实测签名不符（可排除"偏移指到邻居张量"） |
| **E5** | `len` 与 shape 不自洽 / `off` 越 EOF | 两者都抛 `ValueError: cannot reshape ...` ⇒ **这类错是"响"的**，反向说明真文件里 `len`/`shape`/结构是自洽的 |
| **E6** | 哪种"误读"能产生实测签名 (7.139 / -0.011 / 16.75) | 见下表 |

E6 明细（对 896 维 γ≈1.0 的向量做各种误读，再按 fp16 解释）：

| 误读方式 | mean | min | max | 与实测签名 |
|---|---|---|---|---|
| **实测** | **+7.1390** | **−0.0110** | **+16.7500** | — |
| fp16 正常 | +0.9989 | +0.9385 | +1.0576 | 不符（这是"应该"的样子） |
| fp32 字节当 fp16（前半） | — | — | — | **出现 20 个 inf/nan** ⇒ 排除 |
| fp32 字节当 fp16（+1B 错位） | — | — | — | **152 个 inf/nan** ⇒ 排除 |
| bf16 字节当 fp16 | +1.8733 | +1.8594 | +1.8818 | 不符（且极差过小） |
| Q8_0 载荷当 fp16 | — | — | — | 25/476 inf/nan ⇒ 排除 |
| Q4_0 nibble 当 fp16 | — | — | — | 8/224 inf/nan ⇒ 排除 |
| 文本（BPE token 表）当 fp16 | +3444.41 | +0.0083 | +16896 | 不符（大 3 个数量级） |
| uint16 token id 当 fp16 | ≈0 | ≈0 | ≈0 | 不符 |

**E6 的结论（这是本节最有价值的推论）**：实测签名是**全有限的、量级 7 上下的、连贯数值向量**。任何"字节级误解释"都会要么产生 inf/nan、要么给出 1.87 / 3444 这种完全不同的量级。⇒ **那 6 个张量里装的是"以某种方式算出来的、量级 ≈7 的合法数值"，不是"错位的字节"**。这把矛头从"读端/偏移"彻底转向"**写端拿到/算出的值**"（H1/H2/H3）。

---

## 6. 可执行的验证步骤（按顺序做，每步都有明确判定）

> 全部只读；除 Step 4 会写一个临时输出文件外不改任何东西。

### Step 0 · 容器自洽性（30 秒，先排除 H4）
```bash
python tools/tg_check.py "D:\...\qwen2.5-0.5b-instruct-fp16.TG"
# 期望：结论 PASS（exit 0）
# 若 FAIL：直接按它点名的问题走（重复 hash / 区间重叠 / 越界），问题到此为止，不用往下
```

### Step 1 · 名字覆盖双射（验证 M1/M2 不存在，也顺便验证映射表）
```python
# audit_names.py —— 用**转换器的** map_weight_name 造期望名字，不另写一套
import sys, importlib.util, os
sys.path.insert(0, r"<repo>\python")
from tgai_go.read_tg import read_tg, fnv1a64
spec = importlib.util.spec_from_file_location("g2t", r"<repo>\tools\gguf_to_tg.py")
g2t = importlib.util.module_from_spec(spec); sys.modules["g2t"]=g2t; spec.loader.exec_module(g2t)
info = read_tg(r"<X>.TG")
have = {e["hash"]: e for e in info["table"]}
h = info["header"]
# 期望名字（Qwen2.5 结构；带 q_norm/k_norm 的模型要加）
names = ["token_embedding.weight", "final_norm.weight"]
for i in range(h["n_layers"]):
    b = "blocks.%d." % i
    names += [b+"ln1.weight", b+"ln2.weight",
              b+"attn.q_proj.weight", b+"attn.k_proj.weight", b+"attn.v_proj.weight",
              b+"attn.out_proj.weight", b+"ffn.w1.weight", b+"ffn.w2.weight",
              b+"ffn.w3.weight", b+"attn.q_proj.bias", b+"attn.k_proj.bias", b+"attn.v_proj.bias"]
want = {fnv1a64(n): n for n in names}
print("表项", len(have), "期望", len(want))
print("缺失(查不到→None→静默跳过):", [want[k] for k in want if k not in have][:5])
print("孤儿(表里有、没人认领):", len([k for k in have if k not in want]))
print("重复 hash:", len(have) - len({e["hash"] for e in info["table"]}))
```
判定：`缺失=0 且 孤儿=0` ⇒ 名字映射没问题（本 case 预期如此）；否则先修映射，别查数值。

### Step 2 · 值域指纹（区分 H1/H2/H3，**最关键的一步**）
```python
# audit_corr.py —— 需要官方 safetensors 目录
import numpy as np, sys, importlib.util
from safetensors.numpy import load_file
# 1) TG 侧：用 checkpoints.load_tg（先绕开 D1：allow_suspicious_norms=True）
sys.path.insert(0, r"<repo>\python")
from tgai_go.train import checkpoints as C
ck = C.load_tg(r"<X>.TG", allow_suspicious_norms=True)      # ★ 必须显式放行，见 §7 D1
# 2) 参考侧
hf = load_file(r"<Qwen2.5-0.5B-Instruct>\model.safetensors")
pat = {"token_embedding.weight":"model.embed_tokens.weight",
       "final_norm.weight":"model.norm.weight",
       "blocks.0.ln1.weight":"model.layers.0.input_layernorm.weight",
       "blocks.0.ln2.weight":"model.layers.0.post_attention_layernorm.weight"}
for i in (0, 1, 12, 23):
    pat["blocks.%d.attn.v_proj.weight"%i] = "model.layers.%d.self_attn.v_proj.weight"%i
    pat["blocks.%d.ffn.w1.weight"%i]      = "model.layers.%d.mlp.gate_proj.weight"%i
for k, ref_name in pat.items():
    a = np.asarray(ck.weights[k], np.float64).ravel()
    b = np.asarray(hf[ref_name], np.float64).ravel()
    if a.size != b.size: print(k, "SIZE MISMATCH", a.size, b.size); continue
    r = float(np.corrcoef(a, b)[0,1]); d = a-b
    print("%-32s r=%.4f  std(d)/std(w)=%.4f  max|d|/max|w|=%.4f  mean(d)=%+.2e"
          % (k, r, d.std()/(b.std()+1e-12), np.abs(d).max()/(np.abs(b).max()+1e-12), d.mean()))
```
**判读表（把 `std(d)/std(w)` 和 `max|d|/max|w|` 对上号）**：

| 观测 | 结论 |
|---|---|
| `r≈1` 且 `max|d|/max|w| < 1e-3` | 两份等价（fp16 无损），问题在别处（配置/eps/permutation） |
| `max|d|/max|w| ≈ 1/127` 或 `1/7`，`d` 近似零均值且与 `w` 无关，**`d` 的 lag-32 自相关有尖峰** | **H3（反量化/scale bug）** 或 H1，且能定位到具体 block |
| `std(d)/std(w)` 在 0.4~1.4，且**各张量比例差异大（>2×）**，`max|d|` 与 `max|w|` 同量级 | **H2（不是同一份权重）** —— 去查源路径/版本 |
| `r≈0.94` 且 `d` 与 `w` **线性相关**（`d ≈ α·w`，`α≈0.33`） | 值被**整体缩放**过（scale 二次应用/eps 类），也是 H3 的变体 |

### Step 3 · 源 dtype 体检（H1）
```bash
python tools/inspect_gguf.py <源>.gguf | head -80     # 看每个 tensor 的 ggml type
```
若源是 F16/F32 全量 ⇒ 量化排除，转 Step 2 的 H2 分支。

### Step 4 · 可复现性（回答"这份 .TG 到底是谁产出的"）
```bash
python tools/gguf_to_tg.py -i <源>.gguf -o %TEMP%\repro.TG --dtype fp16
python tools/verify_gguf_tg.py --gguf <源>.gguf --tg %TEMP%\repro.TG     # 期望 290/290 max|Δ|=0
certutil -hashfile "%TEMP%\repro.TG" SHA256
certutil -hashfile "<X>.TG" SHA256      # 两份 sha256 不同 ⇒ 这份 .TG 不是这条链路/这个版本的产物
```
这一步是**分水岭**：若 `repro.TG` 与 `<X>.TG` 逐字节/逐张量等价 → 问题在源；若不等价 → 问题在"当初那次转换"（不同的 GGUF / 不同的转换器版本 / 被 `tg_patch`/`tg_requantize` 重写过）。

### Step 5 · 复核 H0（唯一能一锤定音判定"是不是误判"的步骤）
同 Step 2 里 `final_norm` / `ln1` 两行的 safetensors 均值：期望 `≈7.1` / `≈0.032`。

---

## 7. 顺带查出的工程缺陷（建议一并修，都有行号）

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| **D1** | 归一化体检**仍然硬抛错**：`if bad and not allow_suspicious_norms: raise CheckpointError(...)` | `checkpoints.py:646-657` | 与 `CHANGELOG.md:61`、`FINETUNE_RESULTS.md:512`、`tools/tg_check.py:5-9` 三处文档（"已降级为只记录"）**矛盾**。按代码读：官方 Qwen2.5 的 `.TG` 只要 `final_norm` 均值 7.1 就**默认加载失败** |
| **D2** | `tools/tg_audit.py` 用默认参数调 `C.load_tg` ⇒ 必然撞上 D1 | `tg_audit.py:78-79`（`_build` 里 `loader(path)`） | 这个"两路比 CE"的工具在当前代码状态下**拿不到 TG 侧结果**，只会打印"tg 失败：CheckpointError"并判 FAIL。用它做判据前必须先修 D1 |
| **D3** | `load_tg` **从不校验加载了多少张量**；缺张量静默 `continue` | `checkpoints.py:602-621`（`612-614`） | 少一个 `ffn.w3` 也不报错，直到模型层 `qwen.py:117/119/154` 硬索引才 `KeyError`，错误信息远离根因 |
| **D4** | `read_tg` 读 `weight_data_len` 却**从不使用**；无越界/重叠/重复检查 | `read_tg.py:96`、`86-95` | 结构损坏只能靠 `reshape` 偶然报错；正确做法是把 `tools/tg_check.py:21-48` 的逻辑内置进 `read_tg`（或在 `load_tg` 开头强制调用） |
| **D5** | `docs/FINETUNE_RESULTS.md:471-494` 仍以"**根因（2026-11-01 定位）**"的口径写着已被推翻的结论，只在 `507-513` 追加了更正 | 同文件 | 同一文档前后矛盾，后来人按 §10.2 行动会重犯同一个误判；建议把 471-494 直接标注"已推翻，见 §10.2.1" |
| **D6** | `eps` 写死 `1e-6`、`rope_scaling` 完全不读（只有 safetensors 路径读 `388`） | `checkpoints.py:627` | 同一份 `.TG` 若源模型 eps/scaling 不同（如 Qwen2.5 是 1e-6 但 Llama-3 不是、Mistral-v0.3 需要 scaling）⇒ **静默算错**，与 `.TG` 读端无关 |
| **D7** | 扩展段只有 `CHAT_MAGIC` 一种，`read_tg.py:66-68` 直接 `assert m == CHAT_MAGIC` | `read_tg.py:63-73` | 将来加"类型=2 的内容指纹段"时，**必须先改这个 assert**，否则任何非 chat 的 ext 段都会让整份文件读不出来 |

---

## 8. 复现清单（本次分析产生的临时文件，均在仓库外）

| 文件 | 内容 |
|---|---|
| `%TEMP%\tg_diag\diag2.py` | §5 的 E1~E6 全部实验（只 import 仓库的 `read_tg`/`pack`，只写 temp） |
| `%TEMP%\tg_diag\good.TG / dup.TG / off.TG / badlen.TG / eof.TG` | 实验用的手工/合成容器 |

运行：`cd %TEMP%\tg_diag && python diag2.py`（需要 numpy；已实测 Python 3.12.10 + numpy 2.5.1 通过）。

---

## 9. 参考行号索引

| 主题 | 位置 |
|---|---|
| `.TG` 表项字段（权威） | `tools/tg_patch.py:34-40` |
| 表解析（52B 步长、无 name） | `python/tgai_go/read_tg.py:53-60` |
| 按名查表（FNV-1a、first-match、None） | `python/tgai_go/read_tg.py:18-23, 86-95` |
| 期望名字清单（读端硬编码） | `python/tgai_go/train/checkpoints.py:573-621` |
| 缺张量静默跳过 | 同上 `612-614` |
| 2-D 布局校验（1-D 无校验） | 同上 `477-532` |
| 归一化统计判据（误判来源） | 同上 `535-561` |
| 归一化硬抛错（D1） | 同上 `646-657` |
| `DH` 反推 / `eps` 写死 | 同上 `577` / `627` |
| HF 名字映射表 | 同上 `304-319`；`load_safetensors_dir` `331-360` |
| 写端名字映射 | `tools/gguf_to_tg.py:1228-1328` |
| 写端 hash 与表项 | 同上 `1462`、`1542-1554` |
| 写端 layout / shape 交换 hack | 同上 `1492-1504`、`1443-1444` |
| RoPE 重排（Q/K，文件不声明） | 同上 `1125-1222` |
| GGUF 读张量字节（block size 表） | 同上 `485-510`、`75-87` |
| C 引擎按名取权重 | `src/tgai_engine.c:1057, 1388` |
| 结构自检工具 | `tools/tg_check.py:21-48` |
| 两路比 CE 工具（受 D1/D2 影响） | `tools/tg_audit.py:39-49, 78-79` |
| GGUF↔TG 逐张量比对 | `tools/verify_gguf_tg.py:41-45, 78-92` |
| 官方文档：表不存名字 | `tools/verify_gguf_tg.py:4-6`、`docs/GGUF_TO_TG.md:22` |
| 更正记录 | `CHANGELOG.md:46-67`（现为 `48-69`，见文首并发提示）、`docs/FINETUNE_RESULTS.md:507-513` |
| 未同步的旧结论 | `docs/FINETUNE_RESULTS.md:471-494` |
