import io, re
src = io.open("pyproject.toml", encoding="utf-8-sig").read()   # 去掉可能的 BOM
io.open("pyproject.toml", "w", encoding="utf-8", newline="\n").write(src)
import tomllib
d = tomllib.loads(io.open("pyproject.toml", encoding="utf-8").read())
print("TOML 解析 OK：name=%s version=%s" % (d["project"]["name"], d["project"]["version"]))
