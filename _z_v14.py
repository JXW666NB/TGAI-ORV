import io
for f, a, b in (("pyproject.toml", 'version = "0.1.3"', 'version = "0.1.4"'),):
    s = io.open(f, encoding="utf-8").read().replace(a, b, 1)
    io.open(f, "w", encoding="utf-8", newline="\n").write(s)
print("版本 -> 0.1.4")
