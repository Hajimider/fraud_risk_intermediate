"""用 MySQL 8.0+ 汇总交易数据质量、金额分布和时间画像。"""

import os
import re
from pathlib import Path

import pandas as pd
import pymysql
from dotenv import dotenv_values
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "sql" / "fraud_eda.sql"
ANALYSIS_COLUMNS = ["Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]]


def load_mysql_config(environ=None):
    """系统环境变量优先于项目 .env；密码不设代码默认值。"""
    values = {**dotenv_values(ROOT / ".env"), **os.environ} if environ is None else environ
    database = values.get("MYSQL_DATABASE", "fraud_risk")
    if not isinstance(database, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", database):
        raise ValueError("MYSQL_DATABASE 仅支持 1–64 位英文字母、数字和下划线")
    try:
        port = int(values.get("MYSQL_PORT", "3306"))
    except (TypeError, ValueError):
        raise ValueError("MYSQL_PORT 必须是 1–65535 的整数") from None
    if not 1 <= port <= 65535:
        raise ValueError("MYSQL_PORT 必须是 1–65535 的整数")
    return {
        "host": values.get("MYSQL_HOST", "127.0.0.1"),
        "port": port,
        "user": values.get("MYSQL_USER", "root"),
        "password": values.get("MYSQL_PASSWORD") or "",
        "database": database,
    }


def connect_mysql(read_timeout=300):
    """连接项目数据库，不存在时创建；只在该库内使用连接级临时表。"""
    config = load_mysql_config()
    database = config.pop("database")
    connection = None
    try:
        connection = pymysql.connect(
            **config, charset="utf8mb4", autocommit=True,
            connect_timeout=10, read_timeout=read_timeout, write_timeout=300,
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = str(cursor.fetchone()[0])
            if "mariadb" in version.lower() or int(version.split(".")[0]) < 8:
                raise ValueError("SQL 分析需要 MySQL 8.0 或更新版本")
            try:
                connection.select_db(database)
            except pymysql.err.OperationalError as exc:
                if exc.args[0] != 1049:
                    raise
                cursor.execute(f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4")
                connection.select_db(database)
        return connection
    except Exception as exc:
        if connection is not None:
            connection.close()
        if isinstance(exc, pymysql.MySQLError):
            raise RuntimeError(
                f"MySQL 连接失败（错误码 {exc.args[0]}）；请检查服务状态和 .env 中的 MYSQL_* 配置"
            ) from None
        raise


def load_transactions(connection, frame):
    """批量写入连接级临时表，缺失值转换为 SQL NULL；不修改已有永久表。"""
    columns = ", ".join(f"`{column}` DOUBLE NULL" for column in ANALYSIS_COLUMNS)
    names = ", ".join(f"`{column}`" for column in ANALYSIS_COLUMNS)
    placeholders = ", ".join(["%s"] * len(ANALYSIS_COLUMNS))
    with connection.cursor() as cursor:
        cursor.execute(f"CREATE TEMPORARY TABLE transactions ({columns})")
        for start in range(0, len(frame), 1000):
            batch = frame.iloc[start:start + 1000][ANALYSIS_COLUMNS].astype(object)
            batch = batch.where(pd.notna(batch), None)
            cursor.executemany(
                f"INSERT INTO transactions ({names}) VALUES ({placeholders})",
                list(batch.itertuples(index=False, name=None)),
            )


def execute_queries(connection, path):
    """按顺序执行仓库内的 SELECT 脚本；语句以分号结束，注释不包含分号。"""
    results = []
    with connection.cursor() as cursor:
        for statement in Path(path).read_text(encoding="utf-8").split(";"):
            if not statement.strip():
                continue
            cursor.execute(statement)
            columns = [item[0] for item in cursor.description]
            results.append(pd.DataFrame(cursor.fetchall(), columns=columns))
    return results


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

    if frame.empty:
        raise ValueError("交易表不能为空")
    connection = connect_mysql()
    try:
        load_transactions(connection, frame)
        missing_expression = " + ".join(f"COUNT(*) - COUNT(`{column}`)" for column in ANALYSIS_COLUMNS)
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT {missing_expression} FROM transactions")
            missing_count = cursor.fetchone()[0]
        summary_rows, duplicate_rows, amount_rows, profile = execute_queries(connection, SQL_FILE)
        summary = summary_rows.iloc[0]
        duplicate = duplicate_rows.iloc[0]
        amount_rows = amount_rows.rename(columns={
            "fraud_label": "欺诈标签",
            "transaction_count": "交易笔数",
            "mean_amount": "平均金额",
            "median_amount": "金额中位数",
            "amount_q1": "金额P25",
            "amount_q3": "金额P75",
            "amount_p90": "金额P90",
        })
        profile = profile.rename(columns={
            "relative_hour": "相对小时",
            "transaction_count": "交易笔数",
            "fraud_count": "欺诈笔数",
            "mean_amount": "平均金额",
        })
    finally:
        connection.close()

    for column in amount_rows.columns:
        amount_rows[column] = pd.to_numeric(amount_rows[column])
    for column in profile.columns:
        profile[column] = pd.to_numeric(profile[column])
    profile["欺诈比例"] = profile["欺诈笔数"] / profile["交易笔数"]
    intervals = [
        wilson_interval(int(successes), int(total))
        for successes, total in zip(profile["欺诈笔数"], profile["交易笔数"], strict=False)
    ]
    profile["欺诈率下限"] = [interval[0] for interval in intervals]
    profile["欺诈率上限"] = [interval[1] for interval in intervals]
    contingency = profile[["交易笔数", "欺诈笔数"]].copy()
    contingency["正常笔数"] = contingency["交易笔数"] - contingency["欺诈笔数"]
    table = contingency[["正常笔数", "欺诈笔数"]].to_numpy()
    if len(table) > 1 and (table.sum(axis=0) > 0).all():
        chi2, p_value, _, expected = chi2_contingency(table)
        chi_test = {"卡方统计量": float(chi2), "自由度": int(len(profile) - 1), "P值": float(p_value),
                    "期望频数小于5的单元格比例": float((expected < 5).mean()), "说明": "稀疏单元格较多时渐近P值仅供探索"}
    else:
        chi_test = {"卡方统计量": None, "自由度": 0, "P值": None, "说明": "小时或类别不足，不进行检验"}

    return {
        "交易总数": int(summary["total_count"]),
        "欺诈笔数": int(summary["fraud_count"]),
        "欺诈比例": float(summary["fraud_count"] / summary["total_count"]) if summary["total_count"] else 0.0,
        "缺失值总数": int(missing_count or 0),
        "重复行数": int(duplicate["duplicate_rows"] or 0),
        "重复中的欺诈行数": int(duplicate["duplicate_fraud_rows"] or 0),
        "相对小时欺诈率卡方检验": chi_test,
        "执行后端": "mysql",
        "金额分布": amount_rows.to_dict(orient="records"),
    }, profile
