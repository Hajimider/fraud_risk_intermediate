import numpy as np

from fraud_model import (
    build_features,
    choose_policy,
    cost_of_actions,
    make_demo_data,
    paired_ap_difference_interval,
    rolling_time_splits,
    run,
    stratified_bootstrap_intervals,
    time_split,
)


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


def test_rolling_time_splits_keep_windows_chronological_and_test_last():
    folds, test = rolling_time_splits(make_demo_data(rows=12000))
    assert [fold["窗口"] for fold in folds] == ["40%-55%", "55%-70%", "70%-85%"]
    assert len(folds) == 3
    for fold in folds:
        assert fold["train"].Time.max() < fold["validation"].Time.min()
        assert fold["train"].Class.nunique() == fold["validation"].Class.nunique() == 2
    assert folds[-1]["validation"].Time.max() < test.Time.min()
    assert folds[-1]["train"].Time.max() < folds[-1]["validation"].Time.min()


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
    result = choose_policy([0, 1, 0, 1, 0, 0], [.1, .95, .4, .7, .2, .3], review_capacity=.5)
    assert result["复核阈值"] < result["拦截阈值"]
    assert result["复核笔数"] > 0 and result["拦截笔数"] > 0


def test_full_threshold_search_matches_brute_force_with_amount_costs():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([.1, .9, .4, .7, .2, .3, .8, .6])
    amounts = np.array([4, 12, 3, 200, 7, 1, 5, 18])
    actual = choose_policy(y, scores, review_capacity=.5, amounts=amounts)
    thresholds = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    expected = min(
        (cost_of_actions(y, scores, low, high, review_capacity=.5, amounts=amounts)
         for i, low in enumerate(thresholds[:-1]) for high in thresholds[i + 1:]
         if 0 < ((scores >= low) & (scores < high)).sum()
         and ((scores >= low) & (scores < high)).mean() <= .5
         and (scores >= high).any()),
        key=lambda row: (row["模拟成本"], row["复核率"]),
    )
    assert np.isclose(actual["模拟成本"], expected["模拟成本"])


def test_stratified_bootstrap_returns_finite_intervals():
    y = np.array([0] * 20 + [1] * 4)
    scores = np.r_[np.linspace(.01, .2, 20), [.8, .9, .7, .6]]
    amounts = np.ones(len(y))
    intervals = stratified_bootstrap_intervals(y, scores, amounts, .4, .75, iterations=40)
    assert set(intervals) == {"平均精确率AP", "ROC-AUC", "精确率", "召回率", "模拟成本"}
    assert all(np.isfinite(bounds).all() and bounds[0] <= bounds[1] for bounds in intervals.values())


def test_paired_ap_interval_contains_identical_model_difference():
    y = [0] * 20 + [1] * 4
    scores = np.linspace(0, 1, len(y))
    low, high = paired_ap_difference_interval(y, scores, scores, iterations=40)
    assert low == high == 0


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
    report = run(rows=12000, trees=12, tuning_trials=1, bootstrap_iterations=40)
    assert report["数据来源"] == "合成教学数据"
    assert report["验证集选定模型"] in {
        "类别加权逻辑回归",
        "类别加权XGBoost",
        "类别加权XGBoost（移除LogAmount）",
    }
    assert len(report["滚动验证窗口"]) == 3
    assert len(report["特征消融"]) == 18
    assert report["金额特征方案"] != "both" or "XGBoost" in report["验证集选定模型"]
    assert len(report["校准方案比较"]) == 4
    assert len(report["看板分箱"]["分数分布"]) == 10
    assert all("平均AP" in row and "AP标准差" in row for row in report["模型比较"])
    assert report["SQL数据画像"]["交易总数"] == 12000
    assert (tmp_path / "reports" / "latest.json").exists()
    assert (tmp_path / "reports" / "test_scores.csv").exists()
