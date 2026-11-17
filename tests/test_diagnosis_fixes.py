import itertools
import json

import numpy as np
import pandas as pd
import pytest

from experiments import reliability_bins, training_parts
from fraud_model import (
    CostParameters,
    choose_policy,
    cost_of_actions,
    expected_loss_policy,
    make_demo_data,
    stratified_bootstrap_intervals,
)
from sql_analysis import analyze_transactions


@pytest.mark.parametrize("amount_weighted", [False, True])
def test_search_matches_exhaustive_search_with_custom_costs_and_ties(amount_weighted):
    rng = np.random.default_rng(17)
    params = CostParameters(fraud_loss=47, false_block_cost=11, review_cost=.7,
                            review_capacity=.5, review_recall=.63, review_false_positive_rate=.08,
                            min_fraud_loss=3, fraud_loss_multiplier=1.7)
    for _ in range(20):
        y = rng.integers(0, 2, 20)
        score = rng.choice([.1, .2, .4, .6, .9], 20)
        amounts = rng.uniform(0, 100, 20) if amount_weighted else None
        candidates = []
        thresholds = np.unique(np.r_[score, np.nextafter(score.max(), np.inf)])
        for low, high in itertools.combinations_with_replacement(thresholds, 2):
            row = cost_of_actions(y, score, low, high, amounts=amounts, cost_params=params)
            if (row["复核笔数"] or row["拦截笔数"]) and not row["是否超出复核容量"]:
                candidates.append(row)
        actual = choose_policy(y, score, amounts=amounts, cost_params=params)
        expected = min(candidates, key=lambda row: (row["模拟成本"], row["复核率"]))
        assert actual["模拟成本"] == pytest.approx(expected["模拟成本"])
        assert actual["复核率"] == expected["复核率"]


def test_training_early_stopping_and_calibration_segments_do_not_overlap():
    frame = make_demo_data(12000)
    frame.Time = frame.Time // 120
    fit, early, calibration = training_parts(frame)
    assert fit.Time.max() < early.Time.min() <= early.Time.max() < calibration.Time.min()
    assert len(fit) + len(early) + len(calibration) == len(frame)


@pytest.mark.parametrize("scores,capacity", [([.5] * 4, .5)])
def test_choose_policy_handles_tied_scores_without_nonempty_review_block_split(scores, capacity):
    result = choose_policy([0, 1, 0, 1], scores, review_capacity=capacity)
    assert result["复核笔数"] == 0
    assert result["拦截笔数"] == 4


def test_choose_policy_allows_zero_block_actions_when_costs_prefer_review():
    result = choose_policy(
        [0, 1, 0, 1, 0, 0],
        [.01, .02, .03, .04, .05, .06],
        amounts=[1., 1., 1., 1., 1., 1.],
        review_capacity=.5,
        fraud_loss_multiplier=.01,
        false_block_cost=100.,
    )
    assert result["拦截笔数"] == 0
    assert result["复核笔数"] > 0


def test_expected_loss_policy_reports_capacity_constrained_actions():
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([.01, .8, .02, .7, .03, .6])
    amounts = np.array([10., 100., 20., 80., 30., 60.])
    result = expected_loss_policy(y, scores, amounts, cost_params=CostParameters(review_capacity=.5))
    assert result["策略类型"] == "期望损失策略"
    assert result["复核率"] <= .5
    assert result["放行笔数"] + result["复核笔数"] + result["拦截笔数"] == len(y)


def test_expected_loss_policy_reuses_frozen_shadow_price():
    y = np.array([0, 1, 0, 1])
    scores = np.array([.1, .8, .2, .7])
    amounts = np.array([10., 100., 20., 80.])
    validation = expected_loss_policy(y, scores, amounts, cost_params=CostParameters(review_capacity=.5))
    frozen = expected_loss_policy(y, scores, amounts, cost_params=CostParameters(review_capacity=.5),
                                  shadow_price=validation["容量影子价格λ"])
    assert frozen["容量影子价格λ"] == validation["容量影子价格λ"]
    assert frozen["放行笔数"] == validation["放行笔数"]
    assert frozen["复核笔数"] == validation["复核笔数"]
    assert frozen["拦截笔数"] == validation["拦截笔数"]


def test_bootstrap_uses_custom_cost_parameters():
    y = np.array([0, 1, 0, 1] * 5)
    scores = np.array([.1, .8, .2, .7] * 5)
    amounts = np.array([10., 100., 20., 80.] * 5)
    params = CostParameters(false_block_cost=99, review_cost=7)
    result = stratified_bootstrap_intervals(y, scores, amounts, .2, .7, iterations=20, seed=3, cost_params=params)
    assert result["模拟成本"][1] > result["模拟成本"][0]


def test_aggregate_bins_preserve_all_samples_including_constant_scores():
    for score in (np.zeros(100), np.linspace(0, 1, 100)):
        bins = reliability_bins([0] * 90 + [1] * 10, score)
        assert sum(row["交易笔数"] for row in bins) == 100
        assert sum(row["交易笔数"] * row["实际欺诈比例"] for row in bins) == pytest.approx(10)


def test_mysql_summary_and_quantiles_match_input_data():
    frame = make_demo_data(500)
    frame.loc[0, "Amount"] = np.nan
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    summary, profile = analyze_transactions(frame)
    assert summary["执行后端"] == "mysql"
    assert summary["交易总数"] == len(frame)
    assert summary["缺失值总数"] == 1
    assert summary["重复行数"] == 1
    assert profile["交易笔数"].sum() == len(frame)
    for row in summary["金额分布"]:
        amounts = frame.loc[frame.Class == row["欺诈标签"], "Amount"]
        for key, q in (("金额P25", .25), ("金额中位数", .5), ("金额P75", .75), ("金额P90", .9)):
            assert row[key] == pytest.approx(amounts.quantile(q))


def test_report_contains_cloud_charts_and_no_machine_paths():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "reports" / "latest.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("报告版本") != 2:
        pytest.skip("新报告尚未生成")
    assert sum(row["交易笔数"] for row in report["看板分箱"]["分数分布"]) == report["时间切分"]["测试集"]
    assert report["校准结果"]["方法"].startswith("同模型")
    assert len(report["成本敏感性分析"]) == 9
    assert all("复核阈值" in row and "测试成本" in row for row in report["成本敏感性分析"])
