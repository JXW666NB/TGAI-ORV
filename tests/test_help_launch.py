# -*- coding: utf-8 -*-
"""`T.help(open='web', launch=...)` 的行为契约（2026-11-21 用户实际踩到的 bug）。

我改了 `open_web(payload, launch=None)` 却**没把 launch 接到 T.help 上**，
于是文档写着 `T.help(open='web', launch=True)`、代码直接 TypeError ——
"文档承诺了但没实现"比"没有这个功能"更糟。这文件钉住三件事：

1. 不传 launch ⇒ **不许**打开浏览器（否则又会抢用户前台）；
2. `launch=True` ⇒ 打开一次，且给出的是 file:// URI；
3. `TG_HELP_LAUNCH=1` 环境变量 ⇒ 等价于 launch=True；
4. 未知参数的错误提示里必须列出 `launch`（否则用户不知道该写什么）。
"""
import pytest

from tgai_go.train import apilist as T


@pytest.fixture()
def opened(monkeypatch):
    import webbrowser

    box = []
    monkeypatch.setattr(webbrowser, "open",
                        lambda uri, *a, **k: (box.append(uri), True)[1])
    return box


def test_default_does_not_launch(opened, monkeypatch, capsys):
    monkeypatch.delenv("TG_HELP_LAUNCH", raising=False)
    path = T.help(open="web")
    out = capsys.readouterr().out.strip()
    assert opened == [], "默认不许弹浏览器（会用窗口抢用户前台）"
    assert path and out.endswith("index.html")


def test_launch_true_opens_once(opened, capsys):
    T.help(open="web", launch=True)
    capsys.readouterr()
    assert len(opened) == 1, "launch=True 必须真的打开一次"
    assert opened[0].startswith("file://")


def test_env_var_enables_launch(opened, monkeypatch, capsys):
    monkeypatch.setenv("TG_HELP_LAUNCH", "1")
    T.help(open="web")
    capsys.readouterr()
    assert len(opened) == 1


def test_output_path_is_stable(opened, capsys):
    """固定路径（不再 mkdtemp）：重复调用覆盖同一个文件，不堆积临时目录。"""
    p1 = T.help(open="web", print=False)
    p2 = T.help(open="web", print=False)
    capsys.readouterr()
    assert p1 == p2
    assert p1.replace("\\", "/").endswith("tgai-orv-help/index.html")


def test_unknown_kwarg_message_mentions_launch():
    with pytest.raises(TypeError) as ei:
        T.help(open="web", bogus=1)
    msg = str(ei.value)
    assert "launch" in msg
    assert "修法：" in msg
