# -*- coding: utf-8 -*-
"""扫参分析器的判定逻辑测试（纯 CPU，用合成 CSV）。

为什么值得测：这个分析器会**决定推荐哪个微调参数**。
如果它把"降幅好看但梯度一直在爆"的配置选出来，别人照着用就会训崩 ——
而且会以为是引擎的问题。判定规则必须先被测试钉住。
"""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "tools", "sweep_report.py")
    spec = importlib.util.spec_from_file_location("sweep_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(lr, batch, delta, gnorm, steps=300):
    return {"lr": str(lr), "batch": str(batch), "steps": str(steps),
            "delta_pct": str(delta), "gnorm_med": str(gnorm),
            "sec_per_step": "1.2", "_src": "x.csv"}


def test_recommends_only_stable_configs():
    sr = _load()
    rows = [
        _row(2e-5, 4, -30.0, 60.0),    # 降幅最大，但 gnorm 60 > 5×clip ⇒ 不稳，不能推荐
        _row(1e-5, 4, -12.0, 3.0),     # 降幅及格 + 稳 ⇒ 推荐
        _row(5e-6, 2, -2.0, 2.0),      # 稳但降幅不够 ⇒ 未学到
    ]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is not None
    assert best["lr"] == 1e-5 and best["batch"] == 4, best
    assert second is None, "只有一组达标时不该有次佳"


def test_unstable_config_is_rejected_with_reason():
    sr = _load()
    rows = [_row(2e-5, 4, -40.0, 80.0)]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is None
    reasons = " ".join(out[0]["reasons"])
    assert "gnorm" in reasons and "学习率偏大" in reasons, reasons


def test_no_config_passes_says_check_data_not_relax_criteria():
    """全不达标时的输出必须引导去查数据，而不是"再调调超参"。"""
    sr = _load()
    rows = [_row(5e-6, 2, -1.0, 2.0), _row(1e-5, 2, -0.5, 3.0)]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    md = sr.render(out, best, second, 1.0, 5.0)
    assert "没有一组达标" in md
    assert "dataset_report" in md, "应该指向数据体检，而不是继续调参"


def test_tie_breaks_toward_smaller_lr_then_batch():
    sr = _load()
    rows = [
        _row(2e-5, 2, -15.0, 3.0),
        _row(1e-5, 4, -15.0, 3.0),     # 同降幅，lr 更小 ⇒ 应优先
    ]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best["lr"] == 1e-5 and best["batch"] == 4, best
    assert second is not None and second["lr"] == 2e-5


def test_missing_probe_result_is_not_treated_as_success():
    sr = _load()
    rows = [{"lr": "1e-5", "batch": "4", "steps": "300", "delta_pct": "",
             "gnorm_med": "3.0", "sec_per_step": "", "_src": "x.csv"}]
    out, best, _ = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is None
    assert any("没有探针结果" in r for r in out[0]["reasons"])
