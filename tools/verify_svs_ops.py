# -*- coding: utf-8 -*-
"""svs-ops 插件对拍：每个算子与真 PyTorch 比前向 + 反向。

跑法（需要同一个进程里既有 ORV 又有 torch，用在 V100 上的交叉环境）：
    set PYTHONPATH=D:\\orv_pip_0962\\Lib\\site-packages
    C:\\Python311\\python.exe tools\\verify_svs_ops.py
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import tgai_go.train as T  # noqa: E402
from tgai_go.train import plugin  # noqa: E402

RESULT = []


def load_plugin():
    # 直接装插件包（与用户用法一致），这样拿到的是包里的算子集合
    T.plugin.install("examples/bundles/svs-ops", replace=True)
    return T.svs


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = float(np.abs(a - b).max())
    s = float(np.abs(b).max()) or 1.0
    return d, d / s


def check(name, ours, ref, tol=1e-5, note=""):
    d, r = rel(ours, ref)
    ok = r < tol
    RESULT.append((name, ok, r, note))
    print("%-28s 相对 %.2e  %s %s" % (name, r, "OK" if ok else "**超差**", note), flush=True)
    return ok


def main():
    S = load_plugin()
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)

    # ---------- 逐元素 ----------
    x = rng.normal(0, 2.0, (4, 32)).astype(np.float32)
    xt = torch.tensor(x, requires_grad=True)
    for nm, ours_fn, ref_fn in (
        ("sigmoid", S.sigmoid, torch.sigmoid),
        ("softplus", S.softplus, TF.softplus),
        ("log1p", lambda v: S.log1p(v.abs()), lambda v: torch.log1p(v.abs())),
        ("log10", lambda v: S.log10(v.abs() + 0.5), lambda v: torch.log10(v.abs() + 0.5)),
        ("pow2", lambda v: S.pow(v, 2.0), lambda v: v ** 2),
        ("leaky_relu", lambda v: S.leaky_relu(v, 0.01), lambda v: TF.leaky_relu(v, 0.01)),
    ):
        xo = T.from_numpy(x).requires_grad_(True) if hasattr(T.from_numpy(x), "requires_grad_") else None
        xo = T.from_numpy(x)
        xo.requires_grad = True
        yo = ours_fn(xo)
        (yo * yo).sum().backward()
        yt = ref_fn(xt)
        if xt.grad is not None:
            xt.grad.zero_()
        (yt * yt).sum().backward()
        check(nm + " 前向", yo.to_numpy(), yt.detach().numpy())
        check(nm + " 反向", xo.grad.to_numpy(), xt.grad.numpy(), 2e-5)

    # ---------- pad ----------
    for mode in ("constant", "replicate", "reflect"):
        # torch 要求 reflect 的 pad < 该维长度，所以 reflect 用小一点
        pad = (5, 7) if mode == "reflect" else (37, 41)
        xo = T.from_numpy(x)
        xo.requires_grad = True
        yo = S.pad1d(xo, pad[0], pad[1], mode=mode)
        yo.sum().backward()
        xt2 = torch.tensor(x, requires_grad=True)
        yt = TF.pad(xt2, pad, mode=mode)
        yt.sum().backward()
        check("pad1d[%s] 形状" % mode, np.array(yo.shape), np.array(yt.shape), 1e-9)
        check("pad1d[%s] 前向" % mode, yo.to_numpy(), yt.detach().numpy())
        check("pad1d[%s] 反向" % mode, xo.grad.to_numpy(), xt2.grad.numpy())

    # ---------- cumsum ----------
    xo = T.from_numpy(x)
    xo.requires_grad = True
    yo = S.cumsum(xo, -1)
    (yo * yo).sum().backward()
    xt3 = torch.tensor(x, requires_grad=True)
    yt = torch.cumsum(xt3, -1)
    (yt * yt).sum().backward()
    check("cumsum 前向", yo.to_numpy(), yt.detach().numpy())
    check("cumsum 反向", xo.grad.to_numpy(), xt3.grad.numpy())

    # ---------- losses ----------
    a = rng.normal(0, 1, (4, 16)).astype(np.float32)
    b = rng.normal(0, 1, (4, 16)).astype(np.float32)
    for nm, ours_fn, ref_fn in (
        ("l1_loss", S.l1_loss, TF.l1_loss),
        ("mse_loss", S.mse_loss, TF.mse_loss),
    ):
        ao = T.from_numpy(a); ao.requires_grad = True
        bo = T.from_numpy(b)
        lo = ours_fn(ao, bo); lo.backward()
        at = torch.tensor(a, requires_grad=True); bt = torch.tensor(b)
        lt = ref_fn(at, bt); lt.backward()
        check(nm, np.array([lo.to_numpy()]), np.array([float(lt)]), 1e-6)
        check(nm + " 反向", ao.grad.to_numpy(), at.grad.numpy())

    lg = rng.normal(0, 2, (4, 16)).astype(np.float32)
    tg = (rng.random((4, 16)) > 0.5).astype(np.float32)
    lo = T.from_numpy(lg); lo.requires_grad = True
    l_ours = S.bce_with_logits(lo, T.from_numpy(tg)); l_ours.backward()
    lt2 = torch.tensor(lg, requires_grad=True)
    l_ref = TF.binary_cross_entropy_with_logits(lt2, torch.tensor(tg)); l_ref.backward()
    check("bce_with_logits", np.array([l_ours.to_numpy()]), np.array([float(l_ref)]), 1e-6)
    check("bce_with_logits 反向", lo.grad.to_numpy(), lt2.grad.numpy())

    # ---------- hann_window ----------
    check("hann_window(1024)", S.hann_window(1024).to_numpy(),
          torch.hann_window(1024).numpy(), 1e-6)

    # ---------- STFT ----------
    B, L = 2, 4096
    wav = rng.normal(0, 0.3, (B, L)).astype(np.float32)
    for n_fft, hop in ((1024, 256), (512, 128), (2048, 512)):
        win = torch.hann_window(n_fft)
        xo = T.from_numpy(wav); xo.requires_grad = True
        mo = S.stft_mag(xo, n_fft, hop_length=hop, window=S.hann_window(n_fft))
        (mo * mo).sum().backward()
        xt4 = torch.tensor(wav, requires_grad=True)
        spec = torch.stft(xt4, n_fft, hop, n_fft, win, center=True,
                          pad_mode="reflect", return_complex=True)
        mt = spec.abs()
        (mt * mt).sum().backward()
        check("stft_mag[%d/%d] 前向" % (n_fft, hop), mo.to_numpy(), mt.detach().numpy(), 5e-5)
        check("stft_mag[%d/%d] 形状" % (n_fft, hop),
              np.array(mo.shape), np.array(mt.shape), 1e-9)
        check("stft_mag[%d/%d] 反向" % (n_fft, hop), xo.grad.to_numpy(), xt4.grad.numpy(), 5e-5)

    # ---------- STFT 快路径（hop 块分解）----------
    for n_fft, hop in ((1024, 256), (512, 128), (2048, 512), (256, 64)):
        win = torch.hann_window(n_fft)
        xo = T.from_numpy(wav); xo.requires_grad = True
        mo = S.stft_mag_fast(xo, n_fft, hop_length=hop, window=S.hann_window(n_fft))
        (mo * mo).sum().backward()
        xt5 = torch.tensor(wav, requires_grad=True)
        spec = torch.stft(xt5, n_fft, hop, n_fft, win, center=True,
                          pad_mode="reflect", return_complex=True)
        mt = spec.abs()
        (mt * mt).sum().backward()
        check("fast[%d/%d] 形状" % (n_fft, hop), np.array(mo.shape), np.array(mt.shape), 1e-9)
        check("fast[%d/%d] 前向" % (n_fft, hop), mo.to_numpy(), mt.detach().numpy(), 5e-5)
        check("fast[%d/%d] 反向" % (n_fft, hop), xo.grad.to_numpy(), xt5.grad.numpy(), 5e-5)

    bad = [r for r in RESULT if not r[1]]
    print("\n=== 汇总：%d 项，失败 %d ===" % (len(RESULT), len(bad)))
    for nm, _ok, r, note in bad:
        print("  FAIL %-26s 相对 %.2e %s" % (nm, r, note))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
