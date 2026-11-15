# -*- coding: utf-8 -*-
"""让插件包里的原生算子**不用写反向**（走 register_auto），并加「一条命令装完整个环境」。

改动：
1. `install()` 的 native 分支：若 `impl.fallback` 存在且有 `module`，就把
   `impl.python`（快路径）与 `impl.fallback`（可微等价实现）交给 `register_auto` ——
   作者只写 .cu/.dll + 一份 Python 等价实现，反自动导出。
   没声明 fallback 时保持原行为（走 plugin.load_manifest 的低层 C 绑定）。
2. 新增 `install_all(dir)`：把目录下所有 `.tgop`（或含 manifest.json 的子目录）一次装完。
"""
import io
import re

P = "python/tgai_go/train/plugin_meta.py"

OLD = '''            elif form == "native":
                lib = impl.get("library") or impl.get("backends", {}).get("cuda", {}).get("source")
                if not lib:
                    raise PluginMetaError(
                        "算子 %r 是 native 但没给 library/source。修法：补 impl.library" % (op["name"],))
                _plugin.load_manifest(os.path.join(root, lib), replace=replace) \\
                    if os.path.exists(os.path.join(root, lib)) else None
                installed.append(op["name"])'''

NEW = '''            elif form == "native":
                lib = impl.get("library") or impl.get("backends", {}).get("cuda", {}).get("source")
                fb_name = impl.get("fallback")
                mod_name = impl.get("module")
                fast_name = impl.get("python")
                if fb_name:
                    # ★ 轮椅档：原生快路径 + 可微等价实现 ⇒ 反向自动导出，不用写 bwd
                    if not (mod_name and fast_name):
                        raise PluginMetaError(
                            "算子 %r 声明了 fallback，但缺少 module/python（快路径入口）。修法：写成 "
                            '{"module": "impl.py", "python": "forward", "fallback": "fallback_fn"}'
                            % (op["name"],))
                    mpath = os.path.join(root, mod_name)
                    fwd = _load_callable(mpath, fast_name)
                    fbk = _load_callable(mpath, fb_name)
                    ops_plugin.register_auto(
                        op["name"], fwd, fbk,
                        check=bool(op.get("check", True)),
                        rtol=float(op.get("rtol", 2e-3)),
                        kind=op.get("kind", "functional"),
                        doc=op.get("doc", ""), replace=replace)
                    installed.append(op["name"])
                else:
                    if not lib:
                        raise PluginMetaError(
                            "算子 %r 是 native 但没给 library/source，也没给 fallback。"
                            "修法：要么补 impl.library（低层 C 绑定），"
                            "要么补 impl.fallback + module/python（自动反向，推荐）" % (op["name"],))
                    if os.path.exists(os.path.join(root, lib)):
                        _plugin.load_manifest(os.path.join(root, lib), replace=replace)
                    installed.append(op["name"])'''

ALLFN = '''

def install_all(root_dir, *, replace=False, verbose=False):
    """**一条命令装完整个环境**：把目录下所有 `.tgop` 与含 manifest.json 的子目录都装上。

    顺序：先按名字排序装 `.tgop` 与目录。任何一个包的元信息/实现有问题都会报错并指出是哪个包，
    不会静默跳过（静默跳过在"环境没配好"的场景里最难查）。

    返回已安装的元信息列表。
    """
    import glob

    if not os.path.isdir(root_dir):
        raise PluginMetaError(
            "不是目录：%s。修法：给一个存放插件包的目录" % (root_dir,))
    targets = []
    for p in sorted(glob.glob(os.path.join(root_dir, "*.tgop"))):
        targets.append(p)
    for p in sorted(glob.glob(os.path.join(root_dir, "*.zip"))):
        if p not in targets:
            targets.append(p)
    for name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "manifest.json")):
            targets.append(d)

    if not targets:
        raise PluginMetaError(
            "目录里没有可装的插件包：%s。修法：放 .tgop 文件，或含 manifest.json 的子目录"
            % (root_dir,))

    out = []
    for t in targets:
        try:
            out.append(install(t, replace=replace, verbose=verbose))
        except Exception as e:                      # noqa: BLE001
            raise PluginMetaError(
                "装 %s 失败：%s" % (os.path.basename(t), e)) from e
    if verbose:
        print("共装好 %d 个插件包" % len(out))
    return out
'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if "def install_all(" in s:
        print("已经打过补丁")
        return 0
    assert OLD in s, "找不到 native 分支锚点"
    s = s.replace(OLD, NEW, 1)
    s = s.rstrip() + "\n" + ALLFN
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已改：native+fallback 走 register_auto；新增 install_all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
