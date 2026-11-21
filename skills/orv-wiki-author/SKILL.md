---
name: orv-wiki-author
description: 为 TGAI ORV 写/审/翻译 Wiki 词条（TXT 协议）。当用户要给 ORV 的算子/API/内核/环境变量/插件补词条、翻译中英文、核对某个条目的正确性、或校验自己写的词条能不能被引擎加载时使用。
---

# 给 ORV 写词条（TXT 协议）

**你（AI 助手）的任务**：帮用户产出**能被引擎直接加载**的词条 TXT 文件。
引擎会热加载，改完立刻生效，**不需要重装、不需要改 Python 代码**。

> 铁律：**词条文字只能写在 `wiki/*.txt` 里，绝不写进 Python。**
> 你要做的是"写文件"，不是"改代码"。若你发现自己在编辑 `.py` 来加文案，停下。

---

## 1. 一条词条 = 一个 TXT 文件

```txt
[ID] op:mul
[分类] compute
[中文名] 逐元素相乘
[英文名] Element-wise multiply
[作者] 张三

=== 中文 ===
逐元素相乘。`a.mul(b)` 与 `a * b` 等价。

=== English ===
Element-wise multiply. `a.mul(b)` is the same as `a * b`.

=== 示例 ===
y = a.mul(b)

=== 见也 ===
op:matmul
```

**只有 `[ID]` 是必填。** 其余字段与区块都可以缺；只写中文也合法。

## 2. `[ID]` 是唯一键：`类型:名字`

| 类型 | 例子 |
| --- | --- |
| `op` | `op:mul` / `op:matmul` / `op:sum` |
| `api` | `api:DUO` / `api:help` / `api:checkpoint` |
| `kernel` | `kernel:cast_f32_to_bf16` |
| `cli` | `cli:--dp` / `cli:--amp` |
| `env` | `env:TG_DP_OVERLAP` |
| `plugin` | `plugin:你的包名:条目名`（插件自带）|

## 3. 严格的解析规则（写错会被拒绝加载）

1. 头部只允许这五个键：`ID / 分类 / 中文名 / 英文名 / 作者`。**多一个键就报错**（防拼写错静默失效）。
2. 区块只允许这四个：`=== 中文 ===`、`=== English ===`、`=== 示例 ===`、`=== 见也 ===`（大小写/空格容忍）。未知区块 ⇒ 报错。
3. 头部之后、第一个区块之前不能有游离文本。
4. `[ID]` 必须是引擎里真实存在的条目；**不存在的 ID 会报错**并给出最近的 3 个候选。
5. `[分类]` 必须在分类表里：`getting-started / tensor / model / memory / compute / optim /
   distributed / plugin / debug / compat / other`。不在表内 ⇒ 归 `other` 且被 `check()` 列出。
6. 编码：**UTF-8 或 GBK 都能读**（引擎自动识别）。推荐 UTF-8。
7. 文件名随意，但**建议** `类型.名字.txt`（如 `op.mul.txt`）；以 `[ID]` 为准，不一致只警告。
8. `=== 见也 ===` 里引用的 ID 必须存在，否则算**死链**，`check()` 会报。

## 4. 三种工作流

### A) 用户想写一条新词条
1. 先查真实签名（**不要凭记忆写**）：
   ```bash
   python -c "import tgai_go.train as T; T.help('mul')"
   python -c "import tgai_go.train as T; T.help('mul', format='json')"
   ```
2. 让引擎生成骨架（已填好 ID/分类/区块）：
   ```bash
   python -c "import tgai_go.train as T; T.help.contrib('op:mul')"
   ```
3. 在骨架上补文字（中文 + English + 示例 + 见也），存成 `wiki/op.mul.txt`。
4. 自检：`python -c "import tgai_go.train as T; T.help.check()"`，确认 0 错误、无死链。

### B) 用户想知道"还缺什么"
```bash
python -c "import tgai_go.train as T; T.help.missing()"
```
按 kind/分类分组输出待写清单（含真实签名与自动示例）。挑一批写，别一次写十条半成品。

### C) 用户要翻译已有词条
* 已有 `.txt` 里只有 `=== 中文 ===` ⇒ 补一个 `=== English ===` 区块即可，**不要另建文件**。
* 翻译要**照原文事实**，不许自行发挥或添加未经验证的结论。

## 5. 写作要求（这决定词条质量）

* **示例必须能跑**：用真实 API；参数名与真实签名一致（用 `T.help(id, format='json')` 核对 `usage`）。
* **写坑**：ORV 与 PyTorch 的差异、静默错误的写法、显存陷阱 —— 这类信息比"它是什么"更值钱。
* **不要写**"这个函数很有用"之类的废话；一条词条至少给一条**别人不知道的信息**。
* 中英内容**事实一致**，不是逐字直译。
* 未验证的东西标 `⚠️ 未验证`，不要伪装成已实测。

## 6. 一眼看懂的判定表

| 情况 | 引擎行为 |
| --- | --- |
| `[ID]` 缺失 / 写成不存在的条目 | **报错**（加载失败，列出候选）|
| 未知头部键 / 未知区块名 | **报错** |
| 只有中文 | 正常加载，页面显示"暂无英文" |
| 条目在引擎里有、但没有词条 | **仍然显示**：名字 + 引擎自动生成的调用示例 + `未编写` 徽标 |
| `[分类]` 不在表里 | 归 `other`，`check()` 里列出 |
| 编码是 GBK | 正常加载（自动识别）|
| 文件名与 ID 不符 | 警告，以 `[ID]` 为准 |

## 7. 插件作者

插件包内放 `wiki/`（同一套 TXT 协议，ID 用 `plugin:<包名>:<条目>`），或声明入口点：

```toml
[project.entry-points."tgai_go.help"]
mypkg = "mypkg.help:WIKI# 目录路径或 {id: 字段字典}"
```

不需要把插件 import 进引擎主路径。两个插件抢同一 ID ⇒ 加载期报错（指出两个来源）。
