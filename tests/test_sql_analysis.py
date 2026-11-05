import pandas as pd

from fraud_model import make_demo_data
from sql_analysis import analyze_transactions, wilson_interval


def test_sql_analysis_counts_nulls_and_duplicate_rows():
    frame = make_demo_data(rows=500)
    frame.loc[0, "V1"] = float("nan")
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)

    summary, profile = analyze_transactions(frame)

    assert summary["交易总数"] == len(frame)
    assert summary["欺诈笔数"] == int(frame["Class"].sum())
    assert summary["缺失值总数"] == 1
    assert summary["重复行数"] == 1
    assert set(summary["金额分布"][0]) == {
        "欺诈标签", "交易笔数", "平均金额", "金额中位数", "金额P25", "金额P75", "金额P90",
    }
    assert {"欺诈率下限", "欺诈率上限"}.issubset(profile.columns)
    assert (profile["欺诈率下限"] <= profile["欺诈比例"]).all()
    assert (profile["欺诈比例"] <= profile["欺诈率上限"]).all()


def test_wilson_interval_handles_empty_and_boundary_counts():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 100)
    assert low == 0 and 0 < high < 0.1
    low, high = wilson_interval(100, 100)
    assert 0.9 < low < 1 and high == 1
