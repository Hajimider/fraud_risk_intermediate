import pandas as pd
from fraud_model import make_demo_data
from sql_analysis import analyze_transactions, connect_mysql, wilson_interval


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


def test_ieee_temporal_sql_runs_and_excludes_future_label_information():
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[1] / "sql" / "ieee_temporal_features.sql").read_text(encoding="utf-8")
    frame = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [100, 604900, 604900, 1209800],
        "TransactionAmt": [10.0, 20.0, 30.0, 900.0],
        "card1": [7, 7, 7, 7],
        "addr1": [2, 2, 2, 2],
        "DeviceInfo": ["a", "a", "b", "a"],
        "isFraud": [1, 0, 1, 1],
    })

    def evaluate(data):
        connection = connect_mysql()
        outputs = []
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET @label_cutoff_dt = %s", (100,))
                cursor.execute(
                    "CREATE TEMPORARY TABLE transactions ("
                    "TransactionID BIGINT, TransactionDT BIGINT, TransactionAmt DOUBLE, "
                    "card1 BIGINT, addr1 BIGINT, DeviceInfo VARCHAR(255), isFraud TINYINT)"
                )
                cursor.executemany(
                    "INSERT INTO transactions VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    list(data.itertuples(index=False, name=None)),
                )
            for statement in sql.split(";"):
                if statement.strip():
                    with connection.cursor() as cursor:
                        cursor.execute(statement)
                        columns = [item[0] for item in cursor.description]
                        outputs.append(pd.DataFrame(cursor.fetchall(), columns=columns))
        finally:
            connection.close()
        return outputs

    original = evaluate(frame)
    changed = frame.copy()
    changed.loc[3, ["TransactionAmt", "isFraud"]] = [1.0, 0]
    updated = evaluate(changed)
    assert len(original) == 5
    assert original[0].loc[0, "card_txn_count_1h"] == 0
    assert original[3].loc[0, "prior_uid_txn_count"] == 0
    assert original[4].loc[0, "card_fraud_rate_smoothed"] == updated[4].loc[0, "card_fraud_rate_smoothed"]
    assert original[4].loc[1, "card_fraud_rate_smoothed"] == updated[4].loc[1, "card_fraud_rate_smoothed"]
    assert original[4].loc[1, "card_fraud_rate_smoothed"] > 0
    assert original[4].loc[1, "card_fraud_rate_smoothed"] == original[4].loc[2, "card_fraud_rate_smoothed"]
