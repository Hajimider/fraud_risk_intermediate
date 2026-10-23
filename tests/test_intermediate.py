import numpy as np
import pandas as pd

from fraud_model import build_features, cost_of_actions, make_demo_data, run, time_split


def test_time_split_keeps_order_and_sizes():
    frame = make_demo_data(rows=1000)
    train, validation, test = time_split(frame)
    assert train.Time.max() < validation.Time.min() < test.Time.min()
    assert len(train) + len(validation) + len(test) == 1000


def test_feature_engineering_uses_only_current_transaction_fields():
    frame = make_demo_data(rows=20)
    features = build_features(frame)
    assert {"LogAmount", "Amount"}.issubset(features.columns)
    assert "Hour" not in features.columns
    assert len(features) == len(frame)


def test_time_split_never_splits_same_timestamp():
    frame = make_demo_data(rows=1000)
    frame["Time"] = (frame["Time"] // 4).astype(int)
    train, validation, test = time_split(frame)
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()


def test_time_split_rejects_too_few_unique_timestamps():
    frame = make_demo_data(rows=12)
    frame["Time"] = np.repeat([1, 2, 3], 4)
    try:
        time_split(frame)
    except ValueError as exc:
        assert "至少需要4个" in str(exc)
    else:
        raise AssertionError("应拒绝少于4个不同时间戳的数据")


def test_cost_counts_actions_and_capacity():
    result = cost_of_actions([0, 1, 0, 1], [0.1, 0.95, 0.6, 0.2], .5, .9, review_capacity=.5)
    assert result["预计漏放欺诈"] == 1
    assert abs(result["预计误拦正常"] - .01) < 1e-9
    assert result["复核笔数"] == 1
    assert abs(result["模拟成本"] - 101.56) < 1e-9


def test_review_is_not_assumed_perfect():
    result = cost_of_actions([0, 1], [.6, .6], .5, .9, review_capacity=1)
    assert abs(result["预计漏放欺诈"] - .1) < 1e-9
    assert abs(result["预计误拦正常"] - .01) < 1e-9
    assert abs(result["模拟成本"] - 13.06) < 1e-9


def test_policy_keeps_a_real_review_band():
    from fraud_model import choose_policy
    result = choose_policy([0, 1, 0, 1, 0, 0], [.1, .95, .4, .7, .2, .3], review_capacity=.5)
    assert result["复核阈值"] < result["拦截阈值"]
    assert result["复核笔数"] > 0 and result["拦截笔数"] > 0


def test_train_path_does_not_silently_use_demo_data(tmp_path):
    import fraud_model
    try:
        fraud_model.run(tmp_path / "missing.csv")
    except FileNotFoundError as exc:
        assert "python run.py demo" in str(exc)
    else:
        raise AssertionError("真实数据路径不存在时不应静默切换为教学数据")


def test_demo_run_writes_report(tmp_path, monkeypatch):
    import fraud_model
    monkeypatch.setattr(fraud_model, "ROOT", tmp_path)
    report = run(rows=1600, trees=12)
    assert report["数据来源"] == "合成教学数据"
    assert report["验证集选定模型"] in {"类别加权逻辑回归", "类别加权XGBoost"}
    assert (tmp_path / "reports" / "latest.json").exists()
    assert (tmp_path / "reports" / "test_scores.csv").exists()
