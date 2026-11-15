# -*- coding: utf-8 -*-
"""把「原生算子免写反向」与「一条命令装完环境」写进 README / PLUGIN_SPEC / skill。"""
import io
import os

SPEC_ADD = '''### 原生算子也可以不写反向（`impl.fallback`）

`form: "native"` 的算子只要在 `impl` 里多声明一个 `fallback`，`install()` 就会用
`register_auto` 装它 —— **前向走你的原生快路径，反向由 fallback 自动导出**：

```json
{
  "name": "my_kernel",
  "form": "native",
  "impl": {
    "module": "impl.py",
    "python": "fast",          // 快路径（module 里调 .dll/.so 的函数）
    "fallback": "fallback_fn"  // 与 fast 数学等价、用 ORV 算子写
  }
}
```

* 训练：前向 `fast`，反向把 `fallback_fn` 重算一遍再反传（梯度精确，代价是多一次 fallback 前向）
* 推理 / `no_grad`：只走 `fast`，不建图
* 默认 `check=true`：首次反向比对两者输出，超 `rtol`（默认 2e-3，可在算子项里覆盖）直接报错

这条也满足「C 档必须有回退」的纪律 —— 校验器现在把 `impl.fallback` 视为合法回退。

### 一条命令装完整个环境

```python
T.plugin.install_all("bundles/")     # 装目录下所有 .tgop / .zip / 含 manifest.json 的子目录
```

任一个包出错会**指出是哪个包**并带上原始错误（不静默跳过 —— "环境没配好"最难查的就是静默）。

'''

README_ADD = '''
### 一条命令装完整个环境

```python
T.plugin.install_all("bundles/")     # 目录下所有插件包一次装完
```

原生算子也**不用写反向**：manifest 里给 `form: "native"` 加一个 `impl.fallback`
（一份用已有算子写的等价实现），ORV 就自动把反向导出 —— 前向仍是你编译好的原生 kernel。
详见 [`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md)。
'''


def _insert_before(path, anchor, text):
    s = io.open(path, encoding="utf-8").read()
    if text.strip().splitlines()[0] in s:
        print("已存在，跳过", path)
        return
    assert anchor in s, "锚点未命中 %s" % path
    s = s.replace(anchor, text.strip() + "\n\n" + anchor, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("已更新", path)


def main():
    # PLUGIN_SPEC：插在「六、第一个包」之前
    _insert_before("docs/PLUGIN_SPEC.md", "## 六、第一个包", SPEC_ADD)

    # README：插在「从 PyTorch 迁移」之前
    _insert_before("README.md", "## 从 PyTorch 迁移", README_ADD)

    # skill：插在「最容易错的几处」之前
    sk = os.path.expanduser("~/.dsh/skills/orv-engine-dev/SKILL.md")
    if os.path.exists(sk):
        _insert_before(sk, "#### 最容易错的几处",
                       "#### 包里装原生算子：`impl.fallback` 自动反向\n\n"
                       "manifest 里 `form: \"native\"` + `impl.fallback`（一份数学等价的可微实现），\n"
                       "`T.plugin.install()` 就会用 `register_auto` 装：前向原生、反向自动导出。\n"
                       "`check=true`（默认）首次反向比对两者，不一致直接报错。\n\n"
                       "一条命令装完整个环境：`T.plugin.install_all(\"bundles/\")`，\n"
                       "失败会指出是哪个包（**不静默跳过**）。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
