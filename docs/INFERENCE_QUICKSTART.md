# 训练完怎么用：ORV 直接推理（推训一体）

> 一句话：**ORV 训出来的 `.TG` 文件就是推理输入，中间不需要任何格式转换。**
> 这篇文档只讲"拿到权重之后怎么跑起来"，不讲训练（训练见 `QUICKSTART_TRAIN.md`）。

---

## 1. 最短路径（3 条命令）

```bash
pip install -U tgai-orv

# ① 这个模型是什么（不加载权重，秒回）
python -m tgai_go info -m model.TG

# ② 它自己声明的对话配置（系统提示/停止串/模板标记，同样不加载权重）
python -m tgai_go meta -m model.TG

# ③ 开始对话
python -m tgai_go chat -m model.TG
```

单次生成（适合脚本、冒烟测试）：

```bash
python -m tgai_go gen -m model.TG "用一句话解释 MoE"
python -m tgai_go gen -m model.TG -t 0.2 -n 512 --stop "<|end|>" "写个冒泡排序"
```

其它：`python -m tgai_go version` 打印版本号 + 原生引擎是否就位。

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m/--model` | — | `.TG` 模型路径（**不给就报错**，不会瞎猜） |
| `-s/--system` | 空 | 系统提示词 |
| `-n/--max-tokens` | **256** | 生成长度；回答被截断先看这个（它是默认值，不是上限） |
| `-t/--temperature` | 0.6 | 采样温度；要稳定/确定性任务用 0.1~0.2 |
| `--top-k` / `--top-p` | 40 / 0.9 | 采样截断 |
| `--threads` | 引擎默认 | CPU 线程数 |
| `--stop` | 模型内嵌 | 自定义停止串，**可重复传** |
| `--preload` | 关 | 预解码全部权重：首 token 不卡，但更占内存 |

---

## 2. Python 里怎么用

```python
import tgai_go

# 引擎（持有原生库与设备上下文；一个进程建一个就够）
e = tgai_go.TGEngine()

# 加载模型：训练产物直接加载，extend_ctx 可按需扩上下文（NTK/YaRN）
m = tgai_go.load("model.TG", extend_ctx=0, threads=8, preload=False)

# 多轮对话：连续 ask() 自动拼上下文
c = tgai_go.Chat(e, system="你是 TGAI")
print(c.ask("你好"))
c.ask("我叫小明")
print(c.ask("我叫什么？"))          # 它记得
print(c.ask("天气如何", keep=False))  # 单次调用，不进历史

# 一次性生成（不进历史）
print(c.ask("1+1 等于几", keep=False))
```

要更底层（自定义采样/评测/批处理）：

```python
p = tgai_go.Params()          # dict 子类，也可直接传裸 dict
p["temperature"] = 0.2
ids = ...                     # 自己分词 + 自己套模板（这是最容易出错的环节）
out = tgai_go.generate(e, ids, p, eos_id=2)
```

---

## 3. 关键概念：`.TG` 到底是什么

- `.TG` 是 ORV **自己定义的模型容器**：训练产物 = 推理输入。
- magic `0x49414754`；头固定 **128 字节**；权重表每条 52 字节，用**权名 FNV-1a 64** 查表
  （按名字而不是顺序）。
- dtype 码：`0=fp32 1=fp16 2=int8 3=int32 4=q8_0 5=q4_0 6=bf16`。
  ⚠️ **常见错误**：把 `6`（bf16）当 fp16 解释 —— 都是 2 字节，**不报错**，直接出 NaN/乱码。
- 还有一种容器 `.TG4`（自描述分区 + 名字表），magic `0x00344754`，`read_tg` 自动按 magic 分派。
- 详细字段见 `docs/TG_V4_FORMAT.md`、`docs/TG对话配置元数据_20261010.md`，词条见 Wiki 的 `fmt:TG`。

---

## 4. 后端：CPU 与 GPU 的真实情况

| 事实 | 说明 |
|---|---|
| **wheel 里只带 CPU 引擎** | `pip install tgai-orv` 装的是 CPU 版动态库（`_native/`）。GPU 二进制有平台/驱动/架构依赖，放进 wheel 会让包体积暴涨且大概率不匹配。 |
| **GPU 版在目标机上自举** | `tgai_go.gpu` 负责检测 + 编译 + 安装 GPU 后端（CUDA 需按 `sm_75/sm_86/sm_89…` 编译，Vulkan 需目标机环境）。 |
| **三层兜底** | 编不出来/检测不到就**继续用 CPU**，不会因此跑不起来。 |
| 自查 | `tgai_go.native_available()`、`tgai_go.native_dir()`、`python -m tgai_go version` |

---

## 5. 常见坑（按踩中概率排序）

1. **模板/标记用错**：`.TG` 只存权重与配置，**不会替你套对话模板**。模板与训练时不一致的典型表现是
   **乱码或复读** —— 先用 `meta` 看模型声明的标记体系。
2. **`-n 256` 把回答截断**：默认生成 256 token，长回答要显式加大。
3. **温度设了却没变**：模型内嵌默认值优先，用 `describe_params()` 看引擎注册了哪些键。
4. **bf16 当 fp16**：见第 3 节，静默出错。
5. **`ext_off` 处的对话 magic 对不上**：真实 V2.7 文件就是这样，但**文件是好的**；
   早期解析器在这里 assert 太严会误报"文件损坏"。
6. **一进程多引擎**：每个引擎都占一份设备上下文/显存，多实例不共享。

---

## 6. 想查更多

```bash
python -c "import tgai_go.train as T; T.help('load')"     # 终端查某条
python -c "import tgai_go.train as T; T.help(open='web')" # 生成离线 Wiki（单文件，可搜索）
python -c "import tgai_go.train as T; T.help.check()"     # 词条健康度/覆盖率
```

Wiki 里推理相关的词条 ID 前缀：`api:go:*`（推理 API）、`cli:go:*`（命令行）、`fmt:TG`（文件格式）。

---

## 7. 边界（写清楚，避免误会）

- 推理侧条目是 **CPU 引擎**为基准写的；GPU 后端的性能取决于目标机自举结果，**未在此文档给出数字**。
- 同一引擎上的**并发 decode 不在已验证范围内**；需要并发请开多进程。
- 采样是概率的：`--temperature > 0` 时**不要指望逐位可复现**。
