# 词条规范（TXT 协议）—— 写 Wiki 只需要记事本

本文件是给**写词条的人**看的。你不需要会 Python，也不需要懂 ORV 内部结构。
你只要写一个 `.txt` 文件，放进 `wiki/` 目录，**运行中的引擎会自己认出来**（热更新，不用重装）。

---

## 1. 一个文件 = 一条词条

文件名随便起（建议 `类型.名字.txt`，例如 `op.mul.txt`），**真正决定它是哪一条的是文件里的 `[ID]` 行**。

```
[ID] op:mul
```

## 2. `[ID]` 怎么写

格式固定是 **`类型:名字`**：

| 类型 | 用途 | 例子 |
| --- | --- | --- |
| `op` | 张量算子（对 Tensor 调用的运算）| `op:mul`、`op:matmul`、`op:sum` |
| `api` | 顶层函数 / 类 | `api:DUO`、`api:help`、`api:checkpoint` |
| `kernel` | 原生 CUDA 内核 | `kernel:cast_f32_to_bf16` |
| `cli` | 命令行参数 | `cli:--dp`、`cli:--amp` |
| `env` | 环境变量 | `env:TG_DP_OVERLAP` |
| `plugin` | 插件提供的条目（**插件自带**）| `plugin:mypkg:myop` |

**不确定 ID 是什么？** 运行下面这条，它会把可写清单列给你：

```bash
python -c "import tgai_go.train as T; T.help.missing()"
```

---

## 3. 文件长什么样（照抄这份改）

```txt
# 井号开头是注释，引擎会忽略
[ID] op:mul
[分类] compute
[中文名] 逐元素相乘
[英文名] Element-wise multiply
[作者] 你的名字

=== 中文 ===
逐元素相乘。`a.mul(b)` 与 `a * b` 等价。

⚠️ 坑：ORV 的 `*` 是**逐元素**乘法，矩阵乘要用 `a.matmul(b)`。
广播规则与 numpy 一致。

=== English ===
Element-wise multiply. `a.mul(b)` is the same as `a * b`.

⚠️ Gotcha: in ORV `*` is element-wise; use `a.matmul(b)` for matrix product.

=== 示例 ===
y = a.mul(b)
z = a.matmul(b)

=== 见也 ===
op:matmul
op:add
```

## 4. 字段与区块（全部写明，照着填就行）

**头部字段**（每行一个 `[键] 值`，顺序随意）：

| 键 | 必填 | 说明 |
| --- | --- | --- |
| `[ID]` | ✅ **必填** | 见第 2 节 |
| `[分类]` | 建议 | 决定网页侧边栏分组，见第 5 节 |
| `[中文名]` | 建议 | 一行短名，显示在列表里 |
| `[英文名]` | 建议 | 同上（英文）|
| `[作者]` | 可选 | 会显示在页面角落 |

**正文区块**（`=== 名字 ===` 开始，到下一个区块为止）：

| 区块 | 必填 | 说明 |
| --- | --- | --- |
| `=== 中文 ===` | 二选一 | 中文说明，可以多段、可以用反引号包代码 |
| `=== English ===` | 二选一 | 英文说明 |
| `=== 示例 ===` | 建议 | 直接写能跑的代码片段（不需要额外标注语言）|
| `=== 见也 ===` | 可选 | 一行一个 ID，做交叉链接 |

规则很宽松：

* **只写中文也行**（页面会显示"暂无英文"）。中英各一份最理想。
* 少写某个区块**不算错**；**只有 `[ID]` 是必填**。
* 中英区块的名字写成 `=== 中文 ===` / `=== English ===`（大小写、前后空格都容忍）。
* 反引号 `` ` `` 里是代码样式；`**加粗**` 也可以用。

## 5. 分类（决定网页侧边栏）

`getting-started` 入门 / `tensor` 张量 / `model` 模型 / `memory` 显存 / `compute` 计算 /
`optim` 优化器 / `distributed` 分布式 / `plugin` 插件 / `debug` 调试 / `compat` 兼容性 / `other` 其他

想加新分类：编辑 `wiki/_meta.txt`（同 TXT 协议），追加一行
```
[分类] benchmark | 基准测试 | Benchmark | 75
```
（最后一个数字是排序权重，小的排前面。）

## 6. 编码：UTF-8 或 GBK 都能读

用记事本存成 **ANSI/GBK** 也没关系，引擎会自动识别（UTF-8 优先，失败回退 GB18030）。
**推荐 UTF-8**（VSCode / Notepad++ / 记事本的"UTF-8"选项）。

## 7. 校验自己的词条

**现在就能用**（独立脚本，不需要引擎加载器）：
```bash
python scripts/wiki_check.py                 # 校验仓库里的 wiki/
python scripts/wiki_check.py --dir 我的目录    # 校验你自己的目录
python scripts/wiki_check.py --strict        # 警告也算失败（提交前建议开）
```

它会检查：编码（UTF-8/BOM/GBK 自动识别）、`[ID]` 是否存在且格式正确、
头部字段与区块名是否合法、字段是否重复、`[分类]` 是否在分类表里、
中英是否缺边、以及 `=== 见也 ===` 里的引用**是否已有词条**（还没写的算"待写"、不算错）。

加载器落地后（1.0.17+）还会有引擎内版本：
```bash
python -c "import tgai_go.train as T; T.help.check()"
```

写之前可以先让引擎给你生成骨架（需要 1.0.17+）：
```bash
python -c "import tgai_go.train as T; T.help.contrib('op:mul')"
```

## 8. 插件作者：请自带词条

插件包内放 `wiki/` 目录（同样的 TXT 协议，`[ID]` 用 `plugin:你的包名:条目名`），
或在 `pyproject.toml` 里声明入口点：

```toml
[project.entry-points."tgai_go.help"]
mypkg = "mypkg.help:WIKI"     # WIKI = 词条目录路径，或 'id -> 字段字典' 的 mapping
```

**不需要**把插件 import 进引擎主路径，引擎通过入口点就能发现你的词条。
两个插件抢同一个 ID 会在加载期**报错**（指出两个来源），不会静默覆盖。
