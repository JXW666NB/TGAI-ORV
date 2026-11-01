# -*- coding: utf-8 -*-
"""更新 sweep_report 测试以匹配「按实测校准后」的规则。

校准依据（V100 实测，2026-11-01，见 docs/sweep_q25_v100.csv）：
  lr=5e-6/b2 gnorm 35.2 −8.5% PASS（最优）| lr=2e-5 两组发散 FAIL 而 gnorm 只有 30~36
  ⇒ gnorm 不能当筛选门槛（旧规则 5×clip 会把最优组也判死），
    只能当异常值警告；真正的失败信号是"没有探针终值/verdict=FAIL"。
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


def _row(lr, batch, delta, gnorm, verdict="PASS", steps=120):
    return {"lr": str(lr), "batch": str(batch), "steps": str(steps),
            "delta_pct": str(delta), "gnorm_med": str(gnorm),
            "sec_per_step": "0.8", "verdict": verdict, "_src": "x.csv"}


def test_high_gnorm_alone_no_longer_rejects_the_best_config():
    """回归：实测最优组 gnorm=35、降幅 −8.5%，旧规则（5×clip）会把它判死。"""
    sr = _load()
    rows = [_row(5e-6, 2, -8.5, 35.2)]
    out, best, _ = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is not None, "最优组不该因为 gnorm=35 被判死（实测它就是正常值）"
    assert best["lr"] == 5e-6 and best["batch"] == 2


def test_failed_run_is_rejected_with_reason():
    """真正的失败信号：没有探针终值（发散）⇒ 即使 gnorm 不高也要拒。"""
    sr = _load()
    rows = [{"lr": "2e-05", "batch": "2", "steps": "120", "delta_pct": "",
             "gnorm_med": "29.9", "sec_per_step": "0.8", "verdict": "FAIL",
             "_src": "x.csv"}]
    out, best, _ = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is None
    assert any("发散" in r or "失败" in r for r in out[0]["reasons"]), out[0]["reasons"]


def test_extreme_gnorm_still_warns_and_rejects():
    """gnorm 离谱（>200×clip）仍然要拒 —— 那是真的在爆。"""
    sr = _load()
    rows = [_row(2e-5, 2, -30.0, 5000.0)]
    out, best, _ = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is None
    assert any("gnorm" in r for r in out[0]["reasons"])


def test_recommends_largest_drop():
    sr = _load()
    rows = [_row(5e-6, 1, -7.9, 37.1), _row(5e-6, 2, -8.5, 35.2),
            _row(1e-5, 2, -5.7, 32.2), _row(1e-5, 1, -3.3, 37.2, verdict="未学到")]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert (best["lr"], best["batch"]) == (5e-6, 2), best      # 降幅最大者优先
    assert (second["lr"], second["batch"]) == (5e-6, 1), second


def test_tie_breaks_toward_smaller_lr_then_batch():
    sr = _load()
    rows = [_row(2e-5, 2, -15.0, 35.0), _row(1e-5, 4, -15.0, 35.0)]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best["lr"] == 1e-5, best
    assert second is not None and second["lr"] == 2e-5


def test_no_config_passes_points_to_data_not_hyperparams():
    sr = _load()
    rows = [_row(5e-6, 2, -1.0, 35.0), _row(1e-5, 2, -0.5, 34.0)]
    out, best, second = sr.judge(rows, clip=1.0, min_delta=5.0)
    md = sr.render(out, best, second, 1.0, 5.0)
    assert "没有一组达标" in md
    assert "dataset_report" in md


def test_missing_probe_result_is_failure_not_success():
    sr = _load()
    rows = [{"lr": "1e-5", "batch": "4", "steps": "120", "delta_pct": "",
             "gnorm_med": "3.0", "sec_per_step": "", "verdict": "", "_src": "x.csv"}]
    out, best, _ = sr.judge(rows, clip=1.0, min_delta=5.0)
    assert best is None
