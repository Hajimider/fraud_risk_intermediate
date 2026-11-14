"""IEEE-CIS 本地数据接入、MySQL 历史特征、WOE/IV 与 PSI。

不下载比赛文件，不生成虚构实验结果。目标编码只使用训练截止前且已成熟的标签。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sql_analysis import connect_mysql

ENTITIES = {"卡": "card_key", "地址": "address_key", "邮箱": "email_key", "设备": "device_key", "近似用户": "uid"}


def load_ieee(transaction_path, identity_path, output_dir="data/ieee", sample_rows=None):
    paths = [Path(transaction_path), Path(identity_path)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("缺少IEEE-CIS原始文件，请将已获授权的train_transaction.csv和train_identity.csv放到data/ieee/。")
    transactions = pd.read_csv(paths[0], nrows=sample_rows)
    identity = pd.read_csv(paths[1])
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1", "isFraud"}
    if required - set(transactions) or "TransactionID" not in identity:
        raise ValueError(f"IEEE交易表缺少必要字段或身份表缺少TransactionID：{sorted(required - set(transactions))}")
    if transactions.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any():
        raise ValueError("TransactionID必须唯一，不能进行多对多身份合并")
    data = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    if not data.isFraud.isin([0, 1]).all() or not np.isfinite(data[["TransactionDT", "TransactionAmt"]]).all().all():
        raise ValueError("交易时间/金额必须有限，isFraud必须为0/1")
    data = data.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data.to_parquet(directory / "transactions.parquet", index=False)
    metadata = {"来源": "IEEE-CIS Fraud Detection / Kaggle", "网址": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
                "行数": len(data), "是否样本": sample_rows is not None,
                "文件": []}
    # 分块哈希兼容Python 3.10，避免读入两份大型CSV。
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        metadata["文件"].append({"名称": path.name, "SHA256": digest.hexdigest()})
    (directory / "ieee.source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def prepare_keys(frame):
    data = frame.copy()
    for column in ("card1", "addr1", "P_emaildomain", "DeviceInfo"):
        if column not in data:
            data[column] = pd.NA
    # 首笔交易相对日仅为近似身份信息；不声称是真实用户或真实日历日。
    first_day = np.floor(data.TransactionDT / 86400 - pd.to_numeric(data.get("D1", pd.Series(np.nan, index=data.index)), errors="coerce"))
    data["card_key"] = data.card1.astype("string").fillna("缺失")
    data["address_key"] = data.addr1.astype("string").fillna("缺失")
    data["email_key"] = data.P_emaildomain.astype("string").fillna("缺失")
    data["device_key"] = data.DeviceInfo.astype("string").fillna("缺失")
    data["uid"] = data.card_key + ":" + data.address_key + ":" + first_day.astype("Int64").astype("string").fillna("未知首日")
    if any(data[column].str.len().gt(255).any() for column in ENTITIES.values()):
        raise ValueError("实体键长度超过255，不能静默截断为同一个实体")
    return data


def temporal_features(frame, label_cutoff, label_delay=7 * 86400):
    """执行MySQL窗口函数；同刻不互见，后续验证/测试标签永不进入特征。

    标签延迟是假设，可传入业务实际延迟。即使测试标签已经下载，也按冻结训练截止屏蔽。
    """
    if not np.isfinite(label_cutoff) or not isinstance(label_delay, int) or label_delay < 1:
        raise ValueError("标签截止必须有限，标签延迟必须为正整数秒")
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    if required - set(frame) or frame.TransactionID.duplicated().any():
        raise ValueError("窗口特征需要唯一ID、时间、金额和标签")
    data = prepare_keys(frame).sort_values(["TransactionDT", "TransactionID"], kind="stable")
    connection = connect_mysql()
    outputs = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE TEMPORARY TABLE ieee_transactions (TransactionID BIGINT PRIMARY KEY, TransactionDT BIGINT, TransactionAmt DOUBLE, observed_label DOUBLE, " + ", ".join(f"{key} VARCHAR(255)" for key in ENTITIES.values()) + ")")
            columns = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", *ENTITIES.values()]
            values = data[columns].astype(object)
            values.loc[data.TransactionDT > label_cutoff, "isFraud"] = None
            values = values.where(pd.notna(values), None)
            for start in range(0, len(values), 1000):
                cursor.executemany("INSERT INTO ieee_transactions VALUES (" + ",".join(["%s"] * len(columns)) + ")", list(values.iloc[start:start + 1000].itertuples(index=False, name=None)))
            for name, entity in ENTITIES.items():
                # 每个实体单独运行，避免跨实体窗口排序使中间结果膨胀。
                query = f"""
                SELECT TransactionID,
                    COUNT(*) OVER recent_hour AS `{name}_1小时频次`,
                    COUNT(*) OVER recent_day AS `{name}_24小时频次`,
                    SUM(TransactionAmt) OVER recent_day AS `{name}_24小时金额`,
                    COUNT(*) OVER history AS `{name}_历史频次`,
                    AVG(TransactionAmt) OVER history AS `{name}_历史金额均值`,
                    STDDEV_POP(TransactionAmt) OVER history AS `{name}_历史金额标准差`,
                    (TransactionAmt - AVG(TransactionAmt) OVER history) / NULLIF(STDDEV_POP(TransactionAmt) OVER history, 0) AS `{name}_金额偏离度`,
                    CASE WHEN COUNT(*) OVER history = 0 THEN 1 ELSE 0 END AS `{name}_新颖性`,
                    TransactionDT - MAX(TransactionDT) OVER history AS `{name}_距上次秒数`,
                    COUNT(*) OVER history / NULLIF(COUNT(*) OVER global_history, 0) AS `{name}_历史频率编码`,
                    (COALESCE(SUM(observed_label) OVER mature_labels, 0) + 20 * COALESCE(AVG(observed_label) OVER global_labels, 0)) /
                    (COUNT(observed_label) OVER mature_labels + 20) AS `{name}_平滑目标编码`
                FROM ieee_transactions
                WINDOW
                    recent_hour AS (PARTITION BY {entity} ORDER BY TransactionDT RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),
                    recent_day AS (PARTITION BY {entity} ORDER BY TransactionDT RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
                    history AS (PARTITION BY {entity} ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                    mature_labels AS (PARTITION BY {entity} ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING),
                    global_history AS (ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                    global_labels AS (ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING)
                ORDER BY TransactionID
                """
                cursor.execute(query)
                result = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description]).set_index("TransactionID")
                outputs.append(result.apply(pd.to_numeric))
    finally:
        connection.close()
    result = pd.concat(outputs, axis=1).reindex(frame.TransactionID)
    result.index = frame.index
    result["相对日内小时"] = (frame.TransactionDT // 3600) % 24
    result["相对周内日"] = (frame.TransactionDT // 86400) % 7
    return result


def bin_edges(values, bins=10):
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.array([-np.inf, np.inf])
    return np.r_[-np.inf, np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)[1:-1])), np.inf]


def bin_codes(values, edges):
    return pd.cut(pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan), edges, labels=False).fillna(-1).astype(int)


def fit_woe(values, labels, bins=10):
    edges = bin_edges(values, bins)
    codes = bin_codes(values, edges)
    table = pd.DataFrame({"箱": codes.to_numpy(), "标签": np.asarray(labels)}).groupby("箱").标签.agg(["sum", "count"])
    # 训练段拟合分箱及平滑，不从验证/测试估计任何映射。
    bad = (table["sum"] + .5) / (table["sum"].sum() + .5 * len(table))
    good = (table["count"] - table["sum"] + .5) / ((table["count"] - table["sum"]).sum() + .5 * len(table))
    woe = np.log(bad / good)
    return {"edges": edges, "mapping": woe.to_dict(), "IV": float(((bad - good) * woe).sum())}


def transform_woe(values, fitted):
    return bin_codes(values, fitted["edges"]).map(fitted["mapping"]).fillna(0).astype(float)


def psi(reference, current, bins=10):
    edges = bin_edges(reference, bins)
    categories = np.r_[-1, np.arange(len(edges) - 1)]
    a = bin_codes(reference, edges).value_counts().reindex(categories, fill_value=0).to_numpy() + .5
    b = bin_codes(current, edges).value_counts().reindex(categories, fill_value=0).to_numpy() + .5
    a, b = a / a.sum(), b / b.sum()
    return float(((b - a) * np.log(b / a)).sum())


def feature_screening(train, labels, validation):
    rows, mappings = [], {}
    for column in train:
        fitted = fit_woe(train[column], labels)
        drift = psi(train[column], validation[column])
        missing = float(train[column].isna().mean())
        reason = "缺失率>95%" if missing > .95 else "常量" if train[column].nunique() < 2 else "IV<0.02" if fitted["IV"] < .02 else "PSI>0.25" if drift > .25 else "保留"
        rows.append({"特征": column, "缺失率": missing, "IV": fitted["IV"], "PSI": drift, "筛选结果": reason})
        mappings[column] = fitted
    return pd.DataFrame(rows), mappings


def uid_diagnostics(frame):
    keyed = prepare_keys(frame)
    groups = keyed.groupby("uid").isFraud.agg(["count", "mean", "nunique"])
    return {"近似用户数": len(groups), "交易数中位数": float(groups["count"].median()),
            "标签不一致用户占比": float((groups["nunique"] > 1).mean()),
            "D1缺失率": float(frame.D1.isna().mean()) if "D1" in frame else 1.,
            "说明": "card1+addr1+floor(TransactionDT/86400-D1)，参考比赛常见公开构造；不是真实用户标识，缺失键会混组。"}


def prepare_ieee(transaction_path, identity_path, output_dir="data/ieee", sample_rows=None):
    data = load_ieee(transaction_path, identity_path, output_dir, sample_rows)
    times = np.sort(data.TransactionDT.unique())
    if len(times) < 10:
        raise ValueError("需要更多交易时间才能进行时间验证")
    cutoff = times[int(len(times) * .7) - 1]
    engineered = temporal_features(data, cutoff)
    engineered.to_parquet(Path(output_dir) / "temporal_features.parquet", index=False)
    diagnostics = uid_diagnostics(data)
    diagnostics.update({"新增特征数": len(engineered.columns), "标签冻结截止秒": int(cutoff), "标签延迟秒": 7 * 86400,
                        "状态": "数据接入与特征已生成；尚未完成建模评估", "时间口径": "相对时间，不声称真实星期或真实钟点"})
    (Path(output_dir) / "preparation.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics
