"""用 DuckDB SQL 汇总交易数据质量、金额分布和时间画像。"""

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "sql" / "fraud_eda.sql"
ANALYSIS_COLUMNS = ["Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """计算二项比例的 Wilson 95% 置信区间。"""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, 1.0 if successes == total else upper


def analyze_transactions(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """执行项目 SQL 文件并返回报告摘要与按相对小时统计的结果。"""
    missing_columns = set(ANALYSIS_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"SQL分析缺少字段：{', '.join(sorted(missing_columns))}")

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("transactions", frame)
        quoted_columns = [f'"{column.replace(chr(34), chr(34) * 2)}"' for column in ANALYSIS_COLUMNS]
        missing_expression = " + ".join(
            f'COUNT(*) FILTER (WHERE {column} IS NULL)' for column in quoted_columns
        )
        missing_count = connection.execute(
            f"SELECT {missing_expression} FROM transactions"
        ).fetchone()[0]
        connection.execute(SQL_FILE.read_text(encoding="utf-8"))

        summary = connection.execute("SELECT * FROM sql_summary").fetchone()
        duplicate_count = connection.execute(
            "SELECT duplicate_rows FROM sql_duplicate_rows"
        ).fetchone()[0]
        amount_rows = connection.execute(
            "SELECT * FROM sql_amount_by_class"
        ).fetchdf().rename(columns={
            "fraud_label": "欺诈标签",
            "transaction_count": "交易笔数",
            "mean_amount": "平均金额",
            "median_amount": "金额中位数",
            "amount_q1": "金额P25",
            "amount_q3": "金额P75",
            "amount_p90": "金额P90",
        })
        profile = connection.execute("SELECT * FROM sql_hour_profile").fetchdf().rename(columns={
            "relative_hour": "相对小时",
            "transaction_count": "交易笔数",
            "fraud_count": "欺诈笔数",
            "mean_amount": "平均金额",
        })
    finally:
        connection.close()

    profile["欺诈比例"] = profile["欺诈笔数"] / profile["交易笔数"]
    intervals = [
        wilson_interval(int(successes), int(total))
        for successes, total in zip(profile["欺诈笔数"], profile["交易笔数"])
    ]
    profile["欺诈率下限"] = [interval[0] for interval in intervals]
    profile["欺诈率上限"] = [interval[1] for interval in intervals]

    return {
        "交易总数": int(summary[0]),
        "欺诈笔数": int(summary[1]),
        "欺诈比例": float(summary[1] / summary[0]) if summary[0] else 0.0,
        "缺失值总数": int(missing_count or 0),
        "重复行数": int(duplicate_count or 0),
        "金额分布": amount_rows.to_dict(orient="records"),
    }, profile
