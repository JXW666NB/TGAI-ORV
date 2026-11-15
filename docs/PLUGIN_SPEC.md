# 插件与插件包规范（v1）

> 2026-11-15。实现见 `python/tgai_go/train/plugin_meta.py`，校验+打包：
> `T.plugin.install(path)` / `tools/build_bundle.py`。
> 与既有文档的关系：本文只定义**元信息**；可移植性三档见 `PLUGIN_DSL.md`，
> 后端能力纪律见 `PLUGIN_DISPATCH.md`，底层装载见 `PLUGIN_NATIVE.md`。

## 一、为什么要规范

插件原先长在三条互不相通的轨道上，元信息各写各的：

| 轨道 | 入口 | 原有字段 |
| --- | --- | --- |
| 组合式 Python 算子 | `ops_plugin.register(...)` | 只有 `kind`、`doc` |
| 原生库 + JSON 声明 | `plugin_manifest.load_manifest` | `name/version/api_version/library/ops` |
| 算子插件包（zip） | `ops_bundle.load` | `schema/name/arity/backends` |

结果是：**装了什么、什么版本、能跑在哪些后端、别的后端有没有回退，用户看不出来。**
规范把三者统一到一份 schema，并把这些信息做成**注册时校验**而不是文档里的口头约定。

## 二、一行装完整个包

```python
import tgai_go.train as T
T.plugin.install("svs-ops-0.2.0.tgop")     # ← 就这一行
T.svs.stft_mag(wav, 1024, hop_length=256)
```

路径可以是：

* `.tgop` / `.zip`（就是个 zip，根目录放 `manifest.json`）
* 含 `manifest.json` 的**目录**（开发时直接指向目录，免打包）

`manifest.json` 里声明 `namespace`，装完后就有 `T.<namespace>`；不声明则只进
`ops_plugin` 注册表（可按名字取用）。

## 三、元信息字段

| 字段 | 必填 | 规则 |
| --- | --- | --- |
| `schema` | ✅ | 目前只能是 `1`；不匹配直接报错（提示升级 ORV 或改 manifest）|
| `name` | ✅ | `^[a-z][a-z0-9-]{1,63}$`（要能当文件名）|
| `version` | ✅ | 语义化版本 `MAJOR.MINOR.PATCH` |
| `api_version` | ✅ | 插件 API 版本，目前 `"1.0"` |
| `backends` | ✅ | 非空列表，取值限 `cpu/cuda/hip/musa/vulkan/metal` |
| `ops` | ✅ | 算子列表，见下 |
| `summary` | 建议 | 一句话（≤120 字），`T.plugin.summary()` 会显示 |
| `namespace` | 可选 | 装了就有 `T.<namespace>` |
| `description` / `author` / `license` / `homepage` / `tags` / `requires` | 可选 | `requires` 形如 `{"orv": ">=0.9.65"}` |

### 算子项（`ops[]`）

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | ✅ | `^[a-z_][a-z0-9_]{1,63}$`（要能当 Python 属性）|
| `form` | ✅ | `composed` / `dsl` / `native`，对应 `PLUGIN_DSL.md` 的 A/B/C 三档 |
| `impl` | ✅ | 按 `form` 不同，见下表 |
| `arity` / `kind` / `doc` | 可选 | 元信息，供 UI 与校验用 |

| `form` | `impl` 长什么样 |
| --- | --- |
| `composed`（A 档）| `{"module": "ops.py", "python": "函数名"}` —— 纯组合式，梯度由 autograd 给 |
| `dsl`（B 档）| `{"expr": "...", "inputs": [...]}` —— 受限表达式，ORV 生成各后端代码 |
| `native`（C 档）| `{"backends": {"cuda": {"source": "k.cu", "fwd": "...", "bwd": "..."}}, "library": "..."}` |

## 四、纪律：C 档必须有回退

沿用 `PLUGIN_DSL.md`：

> 每个后端原生插件（C 档）都必须同时提供一个 A 或 B 档的回退，
> 否则该插件在别的后端上是"不存在"，而不是"跑出错结果"。

**这条已做成校验**：manifest 里只有 `native` 算子而没有 `composed`/`dsl` 回退时，
`validate()` 直接抛错并说明要补什么。

## 五、为 DSL 铺的路

`dsl` 这一档现在就能**声明**（`form: "dsl"` + `impl.expr`/`impl.inputs`），
装载器已经能识别并按 Python 实现装进去；等 DSL 编译器落地后，
同一份 manifest 不需要改字段 —— 只是 `impl.expr` 会被拿去生成各后端代码，
而不是去找 `impl.python`。

这样做的原因：**规范先稳定，引擎再跟上**。否则每加一种实现形态就要改一遍格式，
用户写好的包就废了。

## 六、第一个包：`svs-ops`

虚拟歌手（DDSP/STFT）项目缺的音频算子，已打成本仓第一个插件包：

```
examples/bundles/svs-ops/
├── manifest.json      # 12 个算子，form 全是 composed
└── svs_ops.py         # 实现（Hann 窗、STFT 幅度谱、cumsum、pad、损失…）
```

构建与验证：

```bash
python tools/build_bundle.py examples/bundles/svs-ops     # → dist/svs-ops-0.2.0.tgop
python tools/smoke_svs_bundle.py                          # 一行安装 + 与 torch 对拍
```

实测（V100）：元信息通过 → **18 个算子**安装 → `stft_mag` 与 `torch.stft` 形状一致、
相对差 **1.99e-07**；`cumsum` 正确。

**已知性能缺口**：组合式 `stft_mag` 比 cuFFT 慢 33×（171.9 ms vs 5.2 ms，
B=2、L=96000、6 档分辨率、前向+反向）。差距来自算子调度次数（一次 STFT≈40 次调用），
不是数学量；追平需要把 STFT 做成带反向的原生算子（cuFFT）。
