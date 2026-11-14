import json
import itertools

import numpy as np
import pandas as pd
import pytest

from experiments import reliability_bins, training_parts
from fraud_model import CostParameters, choose_policy, cost_of_actions, make_demo_data
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
        for low, high in itertools.combinations(np.unique(score), 2):
            row = cost_of_actions(y, score, low, high, amounts=amounts, cost_params=params)
            if row["复核笔数"] > 0 and row["拦截笔数"] > 0 and not row["是否超出复核容量"]:
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


@pytest.mark.parametrize("scores,capacity", [([.1, .2, .3, .4], 0.), ([.5] * 4, .5)])
def test_infeasible_nonempty_review_and_block_policy_has_clear_error(scores, capacity):
    with pytest.raises(ValueError, match="没有满足复核容量的策略"):
        choose_policy([0, 1, 0, 1], scores, review_capacity=capacity)


def test_aggregate_bins_preserve_all_samples_including_constant_scores():
    for score in (np.zeros(100), np.linspace(0, 1, 100)):
        bins = reliability_bins([0] * 90 + [1] * 10, score)
        assert sum(row["交易笔数"] for row in bins) == 100
        assert sum(row["交易笔数"] * row["实际欺诈比例"] for row in bins) == pytest.approx(10)


def test_sql_backends_have_identical_summary_and_quantiles():
    frame = make_demo_data(500)
    frame.loc[0, "Amount"] = np.nan
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    mysql, mysql_profile = analyze_transactions(frame, backend="mysql")
    embedded, embedded_profile = analyze_transactions(frame, backend="duckdb")
    for result in (mysql, embedded):
        result.pop("执行后端")
    assert mysql["缺失值总数"] == embedded["缺失值总数"] == 1
    pd.testing.assert_frame_equal(pd.DataFrame(mysql.pop("金额分布")), pd.DataFrame(embedded.pop("金额分布")), check_dtype=False)
    assert mysql == embedded
    pd.testing.assert_frame_equal(mysql_profile, embedded_profile, check_dtype=False)


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
