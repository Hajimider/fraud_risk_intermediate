import numpy as np
import pandas as pd
import pytest

from ieee_pipeline import (
    feature_screening,
    fit_woe,
    load_ieee,
    prepare_keys,
    psi,
    temporal_features,
    temporal_features_target_encoding,
    transform_woe,
)

pytestmark = pytest.mark.mysql


def sample_transactions():
    n = 40
    return pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.repeat(np.arange(n // 2) * 1000, 2),
                         "TransactionAmt": np.arange(n, dtype=float) + 10, "card1": [7] * n, "addr1": [2] * n,
                         "D1": [0.] * n, "DeviceInfo": ["设备A"] * n, "P_emaildomain": ["a.example"] * n,
                         "isFraud": np.tile([0, 0, 1, 0], n // 4)})


def test_sql_features_do_not_use_peers_future_or_unavailable_labels():
    data = sample_transactions()
    original = temporal_features(data, label_cutoff=9000, label_delay=2000)
    changed = data.copy()
    changed.loc[changed.TransactionDT >= 10000, ["TransactionAmt", "isFraud"]] = [999999., 1]
    updated = temporal_features(changed, label_cutoff=9000, label_delay=2000)
    pd.testing.assert_frame_equal(original.loc[data.TransactionDT < 10000], updated.loc[data.TransactionDT < 10000])
    # 验证标签即使在未来已经下载，也不允许更新后续目标编码。
    changed = data.copy()
    changed.loc[changed.TransactionDT > 9000, "isFraud"] = 1 - changed.loc[changed.TransactionDT > 9000, "isFraud"]
    updated = temporal_features(changed, label_cutoff=9000, label_delay=2000)
    targets = [column for column in original if "目标编码" in column]
    pd.testing.assert_frame_equal(original[targets], updated[targets])
    assert original.shape[1] >= 25
    assert original.iloc[0]["卡_历史频次"] == original.iloc[1]["卡_历史频次"] == 0
    assert original.iloc[2]["卡_历史频次"] == original.iloc[3]["卡_历史频次"] == 2
    assert original.iloc[2]["卡_平滑目标编码"] == 0


def test_target_encoding_only_matches_full_window_encoding():
    data = sample_transactions()
    full = temporal_features(data, label_cutoff=9000, label_delay=2000)
    history = full.drop(columns=[column for column in full if "平滑目标编码" in column])
    lightweight = temporal_features_target_encoding(data, history, label_cutoff=9000, label_delay=2000)
    target_columns = [column for column in full if "平滑目标编码" in column]
    pd.testing.assert_frame_equal(full[target_columns], lightweight[target_columns])


def test_uid_contains_first_relative_day_and_missing_is_explicit():
    frame = sample_transactions()
    frame.loc[0, "D1"] = np.nan
    keyed = prepare_keys(frame)
    assert "未知首日" in keyed.uid.iloc[0]
    assert keyed.uid.iloc[1] != keyed.uid.iloc[0]


def test_missing_entities_remain_null_and_are_not_grouped():
    frame = sample_transactions()
    frame["DeviceInfo"] = pd.NA
    keyed = prepare_keys(frame)
    assert keyed["device_key"].isna().all()
    assert not (keyed["device_key"] == "缺失").any()


def test_ingestion_provenance_and_identity_uniqueness(tmp_path):
    data = sample_transactions().drop(columns="DeviceInfo")
    identity = pd.DataFrame({"TransactionID": data.TransactionID, "DeviceInfo": "设备A"})
    data.to_csv(tmp_path / "transactions.csv", index=False)
    identity.to_csv(tmp_path / "identity.csv", index=False)
    merged = load_ieee(tmp_path / "transactions.csv", tmp_path / "identity.csv", tmp_path / "prepared")
    assert len(merged) == len(data)
    assert (tmp_path / "prepared/ieee.source.json").is_file()
    identity = pd.concat([identity, identity.iloc[[0]]])
    identity.to_csv(tmp_path / "identity.csv", index=False)
    with pytest.raises(ValueError, match="唯一"):
        load_ieee(tmp_path / "transactions.csv", tmp_path / "identity.csv", tmp_path / "invalid")


def test_woe_and_psi_handle_missing_constants_and_new_ranges():
    reference = pd.Series([0., 1., 2., 3., 4., np.nan] * 10)
    labels = np.array([0, 0, 1, 0, 1, 0] * 10)
    fitted = fit_woe(reference, labels)
    assert np.isfinite(transform_woe(pd.Series([-100., 1e8, np.nan]), fitted)).all()
    assert psi(reference, reference) == pytest.approx(0.)
    assert psi(reference.to_numpy(), reference.to_numpy()) == pytest.approx(0.)
    assert psi(reference, reference + 20) > 0
    screening, _ = feature_screening(pd.DataFrame({"常量": np.ones(60), "金额": reference}), labels,
                                     pd.DataFrame({"常量": np.ones(60), "金额": reference + 20}))
    assert screening.loc[screening.特征 == "常量", "筛选结果"].iloc[0] == "常量"


@pytest.mark.slow
def test_monthly_ieee_experiment_on_explicit_synthetic_data(tmp_path):
    from ieee_experiments import monthly_windows, run_ieee
    rng = np.random.default_rng(42)
    n = 2400
    data = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.repeat(np.arange(4), n // 4) * (30 * 86400) + np.tile(np.arange(n // 4) * 3600, 4),
        "isFraud": np.tile(np.arange(n // 4) % 5 == 0, 4).astype(int),
        "card1": rng.integers(1, 30, n), "addr1": rng.integers(1, 10, n), "D1": np.zeros(n),
    })
    data["TransactionAmt"] = np.where(data.isFraud == 1, rng.uniform(70, 150, n), rng.uniform(10, 90, n))
    data["V1"] = data.isFraud + rng.normal(0, .6, n)
    windows, test = monthly_windows(data)
    for _, train, validation in windows:
        assert train.TransactionDT.max() + 7 * 86400 <= validation.TransactionDT.min()
        assert validation.TransactionDT.max() + 7 * 86400 <= test.TransactionDT.min()
    path = tmp_path / "synthetic_ieee.parquet"
    data.to_parquet(path, index=False)
    report = run_ieee(path, tmp_path / "reports", trees=12, trials=1, iterations=10, export_shap=True)
    assert report["总笔数"] == n
    assert len(report["调参记录"]) == 3
    assert len(report["分组消融"]) == 5
    assert report["选定模型"] in {"WOE评分卡", "XGBoost", "LightGBM"}
    assert report["SHAP文件"]
    assert (tmp_path / "reports/ieee_report.json").is_file()
    assert (tmp_path / "reports/images/shap_global.png").is_file()
