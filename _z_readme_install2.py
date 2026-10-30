import io
p = "README.md"
s = io.open(p, encoding="utf-8").read()
SEC = '''---

## 装 / Install

```bash
pip install tgai-orv              # 机器上已有 CUDA Toolkit 或 torch 的 CUDA 库
pip install "tgai-orv[cuda]"      # 裸机器：把 cuBLAS / NVRTC / CUDA runtime 一起装进 venv
```

```bash
python -m tgai_go.train.doctor    # 12 项自检，失败会写清哪个算子/什么形状/怎么修
```

**已验证**：全新的、**没有 torch** 的 venv，不设任何环境变量，`pip install tgai-orv` 后
`doctor` **12/12 PASS**。引擎会自动去本机任何 Python 环境里找 `nvidia/*/lib`；
万一找不到，报错会直接告诉你 `pip install "tgai-orv[cuda]"` 或设 `TGAI_CUDA_BIN`（支持多路径，`:` 分隔）。

> PyPI: https://pypi.org/project/tgai-orv/ · 当前 `0.1.3`

---

## 实力清单 / The Arsenal'''
anchor = "---\n\n## 实力清单 / The Arsenal"
assert anchor in s, "锚点仍未匹配"
s = s.replace(anchor, SEC, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("README 已加安装章节（%d 行）" % s.count("\n"))
