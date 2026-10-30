import io
p = "README.md"
s = io.open(p, encoding="utf-8").read()

SEC = '''---

## 装 / Install

```bash
pip install tgai-orv              # 机器上已有 CUDA Toolkit 或 torch 的 CUDA 库时
pip install "tgai-orv[cuda]"      # 裸机器（不想装 CUDA Toolkit），把 cuBLAS/NVRTC 一起装进 venv
```

然后：

```bash
python -m tgai_go.train.doctor    # 12 项自检
```

**已验证**：全新 venv（**没有 torch**、不需要设任何环境变量）里 `pip install tgai-orv`
→ `doctor` **12/12 PASS**。引擎会自动去本机任何 Python 环境里找 `nvidia/*/lib`；
找不到时会明确告诉你 `pip install "tgai-orv[cuda]"` 或设 `TGAI_CUDA_BIN`（支持多路径，`:` 分隔）。

---

## 这玩意到底是啥 / What the Hell Is This Thing'''

anchor = "---\n\n## 这玩意到底是啥 / What the Hell Is This Thing"
assert anchor in s
s = s.replace(anchor, SEC, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("README 已加安装章节")
