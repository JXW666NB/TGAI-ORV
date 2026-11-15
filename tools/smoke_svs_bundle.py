# -*- coding: utf-8 -*-
"""冒烟：一行装完 svs-ops 插件包，并验证算子在 GPU 上正确。

    import tgai_go.train as T
    T.plugin.install("dist/svs-ops-0.2.0.tgop")
    T.svs.stft_mag(...)
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import torch  # noqa: E402

import tgai_go.train as T  # noqa: E402


def main():
    # ★ 这就是用户要的"一行导入"
    meta = T.plugin.install("dist/svs-ops-0.2.0.tgop", replace=True, verbose=True)
    print("包名 %s 版本 %s 后端 %s" % (meta["name"], meta["version"], meta["backends"]))

    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.3, (2, 4096)).astype(np.float32)

    n_fft, hop = 1024, 256
    mo = T.svs.stft_mag(T.from_numpy(wav), n_fft, hop_length=hop,
                        window=T.svs.hann_window(n_fft))
    mt = torch.stft(torch.tensor(wav), n_fft, hop, n_fft,
                    torch.hann_window(n_fft), center=True, pad_mode="reflect",
                    return_complex=True).abs()
    a = mo.to_numpy()
    b = mt.numpy()
    rel = float(np.abs(a - b).max() / (np.abs(b).max() or 1.0))
    print("stft_mag 形状 %s（torch %s）相对差 %.2e %s"
          % (tuple(mo.shape), tuple(mt.shape), rel, "OK" if rel < 1e-5 else "**超差**"))

    cu = T.svs.cumsum(T.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)), -1)
    ok_cu = np.allclose(cu.to_numpy(), np.cumsum(np.arange(12, dtype=np.float32).reshape(3, 4), -1))
    print("cumsum:", "OK" if ok_cu else "**错**")

    print("T.svs 提供的算子数:", len([n for n in dir(T.svs) if not n.startswith("_")]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
