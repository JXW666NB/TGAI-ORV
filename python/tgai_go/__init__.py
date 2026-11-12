# -*- coding: utf-8 -*-
"""TGAI GO —— 纯 Python 推理库（TGAI 模型 / 兼容 .TG 格式的模型）

用法
----
>>> import tgai_go
>>> m = tgai_go.load(r"D:\\models\\v27.TG")     # 或 load("v27.TG")
>>> m.generate("什么是机器学习？", max_tokens=128)
'机器学习是一种…'

多轮对话：

>>> with tgai_go.load("v27.TG") as m:
...     s = m.session(system="你是TGAI")
...     print(s.chat("1+1是多少？"))
...     print(s.chat("那2+2呢？"))          # 自动带上历史

自定义停止串 / 采样参数：

>>> m.generate("写首诗", max_tokens=64, stop=["。"], min_tokens=8)

设计要点
--------
* **零编译**：底层 C 引擎以预编译 DLL/.so 形式随包分发（`_native/`），
  用户 `pip install tgai-go` 后即可用，不需要编译器或 Vulkan SDK。
* **配置随模型走**：`.TG` 文件可内嵌对话模板（九标记 / ChatML / 纯文本），
  库会自动读取；第三方模型只要打包时带上配置就能直接对话。
* **纯 numpy 采样**：top-k / top-p / min-p / 重复惩罚 / DRY 防复读。
"""

import logging as _logging
import os as _os

# 版本**单一来源**：安装后从发行版元数据取（与 pyproject.toml 保持一致）；
# 直接在源码目录里跑（没装）时退回下面的字面量。两者由 tests/test_version.py 守着。
try:
    from importlib.metadata import PackageNotFoundError as _PNF
    from importlib.metadata import version as _pkg_version
    try:
        __version__ = _pkg_version("tgai-orv")
    except _PNF:
        __version__ = "0.9.58"
except Exception:  # noqa: BLE001  极端老的 Python / 精简环境
    __version__ = "0.9.58"

# 库默认不往 stdout 输出（脚本版会打印，库不该）
_logging.getLogger('tgai_go').addHandler(_logging.NullHandler())

from ._engine import TGEngine                     # noqa: E402
from ._session import (ChatSession, generate,       # noqa: E402
                        kv_safe_text)
from ._meta import parse_tg_chat_meta             # noqa: E402
from .params import Params, describe as describe_params   # noqa: E402
from . import gpu                                   # noqa: E402
from .chat import Chat, PRESETS                     # noqa: E402
from . import server                                # noqa: E402
from . import launcher                              # noqa: E402
from ._sampling import sample_numpy, apply_penalties   # noqa: E402

__all__ = [
    'TGEngine', 'ChatSession', 'generate', 'load', 'Model',
    'parse_tg_chat_meta', 'sample_numpy', 'apply_penalties',
    'native_dir', 'native_available', 'chat_meta_of', '__version__',
    'kv_safe_text', 'Params', 'describe_params', 'gpu', 'launcher',
    'Chat', 'PRESETS', 'server',
]


def native_dir():
    """返回包内原生库所在目录（诊断用）"""
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '_native')


def native_available():
    """检查原生引擎是否随包就位"""
    d = native_dir()
    if not _os.path.isdir(d):
        return False
    for f in _os.listdir(d):
        if f.endswith(('.dll', '.so', '.dylib')):
            return True
    return False


def chat_meta_of(tg_path):
    """只读 .TG 的对话配置（不加载权重，很快）

    用于在下载/加载大模型前先看看它声明了哪些对话参数。
    """
    return parse_tg_chat_meta(tg_path)


def enable_logging(level=_logging.INFO):
    """打开库日志（默认静默）"""
    lg = _logging.getLogger('tgai_go')
    if not any(isinstance(h, _logging.StreamHandler) for h in lg.handlers):
        lg.addHandler(_logging.StreamHandler())
    lg.setLevel(level)


class Model:
    """模型句柄（`load()` 的返回类型）

    包了 TGEngine，提供更顺手的 API：
      - model.generate(text, **kw)    单轮生成
      - model.session(**kw)           开多轮会话
      - model.encode/decode           分词
      - with 语句自动释放
    """

    def __init__(self, engine):
        self._e = engine
        self.cfg = engine.cfg
        self.chat_meta = getattr(engine, 'chat_meta', {'present': False})

    # ── 元信息 ──
    @property
    def n_layers(self):
        return self.cfg['n_layers']

    @property
    def d_model(self):
        return self.cfg['d_model']

    @property
    def vocab_size(self):
        return self.cfg['vocab_size']

    @property
    def max_seq_len(self):
        return self.cfg['max_seq']

    @property
    def engine(self):
        """底层 TGEngine（需要精细控制时用）"""
        return self._e

    # ── 分词 ──
    def encode(self, text, add_bos=True):
        return self._e.encode(text, add_bos=add_bos)

    def decode(self, token_id):
        return self._e.decode(token_id)

    # ── 生成 ──
    def generate(self, prompt, *, system=None, history=None, **kw):
        """单轮生成（内部建一次会话，不复用 KV）

        prompt   : 问题文本
        system   : 系统提示
        history  : [(q, a), ...] 预置历史
        **kw     : 传给 session.chat 的参数（max_tokens / temperature / stop …）
        """
        s = self.session(system=system)
        if history:
            s.history.extend(history)
        return s.chat(prompt, **kw)

    def stream(self, prompt, *, system=None, **kw):
        """流式生成，逐段 yield 文本"""
        s = self.session(system=system)
        return s.stream(prompt, **kw)

    def session(self, system="", **kw):
        """开一个多轮会话"""
        return ChatSession(self._e, sys_prompt=system or "", **kw)

    # ── 性能 ──
    def set_threads(self, n):
        """设置 CPU 线程数（返回实际生效值）"""
        return self._e.set_threads(n)

    @property
    def threads(self):
        return self._e.get_threads()

    def preload(self):
        """预解码全部权重（首次推理更快）"""
        return self._e.preload()

    # ── 资源 ──
    def close(self):
        try:
            self._e.free()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return ('<tgai_go.Model layers=%d d_model=%d vocab=%d max_seq=%d>'
                % (self.n_layers, self.d_model, self.vocab_size, self.max_seq_len))


def load(path, *, dll=None, extend_ctx=0, threads=None, preload=False):
    """加载 .TG 模型

    path       : .TG 文件路径
    dll        : 指定引擎库（默认用包内的 _native/）
    extend_ctx : 扩展上下文长度（NTK/YaRN），0 = 用模型自带 max_seq
    threads    : CPU 线程数（None = 引擎默认，通常是逻辑核数）
    preload    : True = 预解码全部权重（首次推理不卡顿，但占更多内存）

    返回 Model 对象。

    注意：外部分词器（sidecar）无需手动指定 —— 若模型旁存在
          <model.TG>.tokenizer.json，引擎会自动使用它。
    """
    if not _os.path.exists(path):
        raise FileNotFoundError('模型文件不存在: %s' % path)
    # ★ 2026-11-12（外部报告 P0）：pip 装出来的 wheel 此前**不含** _native/ 原生库，
    #   于是这里直接抛 `FileNotFoundError: DLL not found`，用户不知道是打包问题还是自己缺东西。
    #   另外 ≤0.9.56 的 Windows 引擎 DLL 存在 PE 头错位（ctypes 报 WinError 193），
    #   也不是用户能修的。两种情形都给可执行的指引。
    if dll is None and not native_available():
        raise RuntimeError(
            '找不到推理引擎的原生库（%s 下没有 .dll/.so/.dylib）' % native_dir()
            + '。修法：① 升级到 tgai-orv>=0.9.57（此版本起 _native/ 会随 wheel 分发）；'
              '② 或从源码安装：git clone 后 `pip install -e .`；'
              '③ 或用 dll= 显式指定引擎库路径。'
              '注：训练侧（tgai_go.train）不依赖这个原生库，不受影响')
    try:
        e = TGEngine(dll_path=dll)
        e.load(path, extend_ctx=extend_ctx)
    except OSError as exc:
        _extra = ""
        if "193" in str(exc) or "not a valid Win32" in str(exc):
            _extra = ('其中 WinError 193 表示原生 DLL 本身损坏 —— '
                      'tgai-orv ≤0.9.56 随包的 Windows 引擎 DLL 存在 PE 头错位（已知缺陷，'
                      '需重编，见 docs/EXTERNAL_REVIEW_TRIAGE_2026-11-12.md）。')
        raise RuntimeError(
            '推理引擎原生库加载失败：%s' % str(exc).rstrip('。')
            + '。修法：' + _extra
            + '① 用 dll= 显式指向你自己的引擎库；'
              '② 或在 Linux 上构建/使用；'
              '③ 检查该 .dll 能否被 `ctypes.CDLL(路径)` 单独加载。'
              '注：训练侧（tgai_go.train）不依赖这个原生库，不受影响') from exc
    if threads:
        got = e.set_threads(threads)
        if got > 0:
            _logging.getLogger('tgai_go').info('threads=%d', got)
    if preload:
        e.preload()
    return Model(e)
