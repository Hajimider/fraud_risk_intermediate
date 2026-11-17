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
    suffix = f".sample-{sample_rows}" if sample_rows is not None else ""
    data.to_parquet(directory / f"transactions{suffix}.parquet", index=False)
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
    (directory / f"ieee{suffix}.source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def prepare_keys(frame):
    data = frame.copy()
    for column in ("card1", "addr1", "P_emaildomain", "DeviceInfo"):
        if column not in data:
            data[column] = pd.NA
    # 首笔交易相对日仅为近似身份信息；不声称是真实用户或真实日历日。
    first_day = np.floor(data.TransactionDT / 86400 - pd.to_numeric(data.get("D1", pd.Series(np.nan, index=data.index)), errors="coerce"))
    data["card_key"] = data.card1.astype("string")
    data["address_key"] = data.addr1.astype("string")
    data["email_key"] = data.P_emaildomain.astype("string")
    data["device_key"] = data.DeviceInfo.astype("string")
    data["uid"] = data.card_key + ":" + data.address_key.fillna("?") + ":" + first_day.astype("Int64").astype("string").fillna("未知首日")
    data.loc[data.card_key.isna(), "uid"] = pd.NA
    if any(data[column].str.len().gt(255).any() for column in ENTITIES.values()):
        raise ValueError("实体键长度超过255，不能静默截断为同一个实体")
    return data


def _target_encoding_features(frame, label_cutoff, label_delay):
    """按成熟标签截止计算实体平滑目标编码，避免把整表结果从 SQL 拉回 Python。"""
    data = prepare_keys(frame[[
        column for column in (
            "TransactionID", "TransactionDT", "isFraud", "card1", "addr1", "D1", "P_emaildomain", "DeviceInfo",
        ) if column in frame
    ]]).sort_values(["TransactionDT", "TransactionID"], kind="stable")
    observed = pd.to_numeric(data["isFraud"], errors="coerce").where(data.TransactionDT <= label_cutoff)
    time_totals = pd.DataFrame({"TransactionDT": data.TransactionDT, "observed": observed}).groupby(
        "TransactionDT", sort=True, as_index=True
    ).agg(fraud_count=("observed", "sum"), label_count=("observed", "count"))
    times = time_totals.index.to_numpy(dtype=np.int64)
    cutoff_times = np.searchsorted(times, times - label_delay, side="right") - 1
    fraud_cumulative = time_totals.fraud_count.to_numpy(dtype=float).cumsum()
    label_cumulative = time_totals.label_count.to_numpy(dtype=float).cumsum()
    mature_fraud = np.where(cutoff_times >= 0, fraud_cumulative[np.maximum(cutoff_times, 0)], 0.0)
    mature_labels = np.where(cutoff_times >= 0, label_cumulative[np.maximum(cutoff_times, 0)], 0.0)
    global_rate = np.divide(mature_fraud, mature_labels, out=np.zeros_like(mature_fraud), where=mature_labels > 0)
    global_frame = pd.DataFrame({"TransactionDT": times, "global_mature_rate": global_rate})
    outputs = []
    for name, entity in ENTITIES.items():
        entity_data = data.loc[data[entity].notna(), ["TransactionID", "TransactionDT", entity]].copy()
        if entity_data.empty:
            outputs.append(pd.DataFrame(index=frame.TransactionID, columns=[f"{name}_平滑目标编码"], dtype=float))
            continue
        entity_data["observed"] = observed.loc[entity_data.index]
        entity_times = entity_data.groupby([entity, "TransactionDT"], sort=False).agg(
            fraud_count=("observed", "sum"), label_count=("observed", "count")
        ).reset_index().sort_values([entity, "TransactionDT"], kind="stable")

        def mature(group):
            group = group.copy()
            group_times = group.TransactionDT.to_numpy(dtype=np.int64)
            indices = np.searchsorted(group_times, group_times - label_delay, side="right") - 1
            fraud = group.fraud_count.to_numpy(dtype=float).cumsum()
            labels = group.label_count.to_numpy(dtype=float).cumsum()
            valid = np.maximum(indices, 0)
            group["mature_fraud"] = np.where(indices >= 0, fraud[valid], 0.0)
            group["mature_labels"] = np.where(indices >= 0, labels[valid], 0.0)
            return group

        entity_times = entity_times.groupby(entity, sort=False, group_keys=False).apply(mature).reset_index(drop=True)
        entity_times = entity_times.merge(global_frame, on="TransactionDT", how="left", validate="many_to_one")
        entity_times[f"{name}_平滑目标编码"] = (
            entity_times.mature_fraud + 20 * entity_times.global_mature_rate
        ) / (entity_times.mature_labels + 20)
        values = entity_data[["TransactionID", entity, "TransactionDT"]].merge(
            entity_times[[entity, "TransactionDT", f"{name}_平滑目标编码"]],
            on=[entity, "TransactionDT"], how="left", validate="many_to_one",
        )
        outputs.append(values.set_index("TransactionID")[[f"{name}_平滑目标编码"]])
    result = pd.concat(outputs, axis=1).reindex(frame.TransactionID)
    result.index = frame.index
    return result


def _history_only_features(data):
    """使用分组聚合和向量化滚动窗口计算只依赖历史的特征。"""
    data = data[["TransactionID", "TransactionDT", "TransactionAmt", *ENTITIES.values()]].copy()
    for entity in ENTITIES.values():
        codes, _ = pd.factorize(data[entity], sort=False)
        data[entity] = codes.astype(np.int64)
        data.loc[codes < 0, entity] = None
    outputs = []
    global_times = data.groupby("TransactionDT", sort=True).size()
    global_history = global_times.cumsum().shift(1, fill_value=0)
    global_history.name = "global_history_count"
    global_history.index.name = "TransactionDT"

    for name, entity in ENTITIES.items():
        entity_data = data.loc[data[entity].notna(), ["TransactionID", "TransactionDT", "TransactionAmt", entity]]
        entity_times = entity_data.groupby([entity, "TransactionDT"], sort=False).agg(
            transaction_count=("TransactionID", "size"),
            amount_sum=("TransactionAmt", "sum"),
            amount_square_sum=("TransactionAmt", lambda values: np.square(values).sum()),
        ).reset_index().sort_values([entity, "TransactionDT"], kind="stable")
        by_entity = entity_times.groupby(entity, sort=False)
        entity_times["prior_count"] = by_entity.transaction_count.cumsum() - entity_times.transaction_count
        entity_times["prior_amount"] = by_entity.amount_sum.cumsum() - entity_times.amount_sum
        entity_times["prior_amount_squares"] = by_entity.amount_square_sum.cumsum() - entity_times.amount_square_sum
        entity_times["prior_time"] = by_entity.TransactionDT.shift(1)
        time_index = pd.to_timedelta(entity_times.TransactionDT, unit="s")
        indexed = entity_times.set_index(time_index)
        indexed.index.name = "window_time"
        for seconds, value_column, result_column in (
            (3600, "transaction_count", "hour_count"),
            (86400, "transaction_count", "day_count"),
            (86400, "amount_sum", "day_amount"),
        ):
            rolled = indexed.groupby(entity, sort=False)[value_column].rolling(
                f"{seconds}s", closed="left"
            ).sum()
            entity_times[result_column] = rolled.reset_index(level=0, drop=True).to_numpy()
        entity_times = entity_times.join(global_history, on="TransactionDT")
        count = entity_times.prior_count
        mean = entity_times.prior_amount / count.replace(0, np.nan)
        variance = (entity_times.prior_amount_squares / count.replace(0, np.nan) - mean.pow(2)).clip(lower=0)
        feature_columns = [
            f"{name}_历史频次", f"{name}_历史金额均值", f"{name}_历史金额标准差", f"{name}_新颖性",
            f"{name}_距上次秒数", f"{name}_历史频率编码", f"{name}_1小时频次", f"{name}_24小时频次",
            f"{name}_24小时金额",
        ]
        entity_times[feature_columns] = pd.concat([
            count,
            mean,
            np.sqrt(variance),
            (count == 0).astype(float),
            entity_times.TransactionDT - entity_times.prior_time,
            count / entity_times.global_history_count.replace(0, np.nan),
            entity_times.hour_count,
            entity_times.day_count,
            entity_times.day_amount,
        ], axis=1).to_numpy()
        entity_features = entity_times.set_index([entity, "TransactionDT"])[[
            *feature_columns,
        ]]
        transaction_features = entity_data.merge(entity_features.reset_index(), on=[entity, "TransactionDT"], how="left", validate="many_to_one")
        transaction_features[f"{name}_金额偏离度"] = (
            transaction_features.TransactionAmt - transaction_features[f"{name}_历史金额均值"]
        ) / transaction_features[f"{name}_历史金额标准差"].replace(0, np.nan)
        outputs.append(transaction_features.set_index("TransactionID")[feature_columns + [f"{name}_金额偏离度"]])
    return outputs


def temporal_features(frame, label_cutoff, label_delay=7 * 86400, target_only=False, history_only=False):
    """执行MySQL窗口函数；同刻不互见，后续验证/测试标签永不进入特征。

    标签延迟是假设，可传入业务实际延迟。即使测试标签已经下载，也按冻结训练截止屏蔽。
    ``target_only`` 只物化平滑目标编码，供已缓存历史特征的实验复用。
    """
    if not np.isfinite(label_cutoff) or not isinstance(label_delay, int) or label_delay < 1:
        raise ValueError("标签截止必须有限，标签延迟必须为正整数秒")
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    if required - set(frame) or frame.TransactionID.duplicated().any():
        raise ValueError("窗口特征需要唯一ID、时间、金额和标签")
    if target_only and history_only:
        raise ValueError("target_only与history_only不能同时启用")
    source_columns = [column for column in (
        "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
        "card1", "addr1", "D1", "P_emaildomain", "DeviceInfo",
    ) if column in frame]
    data = prepare_keys(frame[source_columns]).sort_values(["TransactionDT", "TransactionID"], kind="stable")
    if history_only:
        outputs = _history_only_features(data)
        result = pd.concat(outputs, axis=1).reindex(frame.TransactionID)
        result.index = frame.index
        result["相对日内小时"] = (frame.TransactionDT // 3600) % 24
        result["相对周内日"] = (frame.TransactionDT // 86400) % 7
        return result
    connection = connect_mysql(read_timeout=1800)
    outputs = []
    try:
        with connection.cursor() as cursor:
            entity_columns = ", ".join(f"{key} VARCHAR(255)" for key in ENTITIES.values())
            entity_indexes = ", ".join(f"INDEX `ix_{key}_time` (`{key}`, TransactionDT)" for key in ENTITIES.values())
            cursor.execute(
                "CREATE TEMPORARY TABLE ieee_transactions (TransactionID BIGINT PRIMARY KEY, "
                "TransactionDT BIGINT, TransactionAmt DOUBLE, observed_label DOUBLE, "
                f"{entity_columns}, {entity_indexes})"
            )
            columns = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", *ENTITIES.values()]
            values = data[columns].astype(object)
            values.loc[data.TransactionDT > label_cutoff, "isFraud"] = None
            values = values.where(pd.notna(values), None)
            for start in range(0, len(values), 10000):
                cursor.executemany("INSERT INTO ieee_transactions VALUES (" + ",".join(["%s"] * len(columns)) + ")", list(values.iloc[start:start + 10000].itertuples(index=False, name=None)))
            if target_only:
                cursor.execute(f"""
                    CREATE TEMPORARY TABLE ieee_global_features AS
                    SELECT TransactionDT,
                        COALESCE(SUM(fraud_count) OVER global_labels /
                            NULLIF(SUM(label_count) OVER global_labels, 0), 0) AS global_mature_rate
                    FROM (
                        SELECT TransactionDT,
                            SUM(observed_label) AS fraud_count,
                            COUNT(observed_label) AS label_count
                        FROM ieee_transactions
                        GROUP BY TransactionDT
                    ) AS time_totals
                    WINDOW global_labels AS (
                        ORDER BY TransactionDT
                        RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING
                    )
                """)
            else:
                cursor.execute(f"""
                    CREATE TEMPORARY TABLE ieee_global_features AS
                    SELECT TransactionDT,
                        SUM(transaction_count) OVER global_history AS global_history_count,
                        COALESCE(SUM(fraud_count) OVER global_labels /
                            NULLIF(SUM(label_count) OVER global_labels, 0), 0) AS global_mature_rate
                    FROM (
                        SELECT TransactionDT, COUNT(*) AS transaction_count,
                            SUM(observed_label) AS fraud_count, COUNT(observed_label) AS label_count
                        FROM ieee_transactions
                        GROUP BY TransactionDT
                    ) AS time_totals
                    WINDOW
                        global_history AS (ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                        global_labels AS (ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING)
                """)
            cursor.execute("ALTER TABLE ieee_global_features ADD PRIMARY KEY (TransactionDT)")
            for name, entity in ENTITIES.items():
                if target_only:
                    cursor.execute(f"""
                        CREATE TEMPORARY TABLE ieee_entity_times AS
                        SELECT `{entity}` AS entity_key, TransactionDT,
                            SUM(observed_label) AS fraud_count,
                            COUNT(observed_label) AS label_count
                        FROM ieee_transactions
                        WHERE `{entity}` IS NOT NULL
                        GROUP BY `{entity}`, TransactionDT
                    """)
                else:
                    cursor.execute(f"""
                        CREATE TEMPORARY TABLE ieee_entity_times AS
                        SELECT `{entity}` AS entity_key, TransactionDT,
                            COUNT(*) AS transaction_count,
                            SUM(TransactionAmt) AS amount_sum,
                            SUM(TransactionAmt * TransactionAmt) AS amount_square_sum,
                            SUM(observed_label) AS fraud_count,
                            COUNT(observed_label) AS label_count
                        FROM ieee_transactions
                        WHERE `{entity}` IS NOT NULL
                        GROUP BY `{entity}`, TransactionDT
                    """)
                cursor.execute("ALTER TABLE ieee_entity_times ADD PRIMARY KEY (entity_key, TransactionDT)")
                if target_only:
                    cursor.execute(f"""
                        CREATE TEMPORARY TABLE ieee_entity_features AS
                        SELECT entity_key, TransactionDT,
                            (COALESCE(mature_fraud, 0) + 20 * global_mature_rate) /
                                (COALESCE(mature_labels, 0) + 20) AS `{name}_平滑目标编码`
                        FROM (
                            SELECT entity_times.entity_key, entity_times.TransactionDT,
                                SUM(entity_times.fraud_count) OVER mature_labels AS mature_fraud,
                                SUM(entity_times.label_count) OVER mature_labels AS mature_labels,
                                global.global_mature_rate
                            FROM ieee_entity_times AS entity_times
                            JOIN ieee_global_features AS global USING (TransactionDT)
                            WINDOW mature_labels AS (
                                PARTITION BY entity_times.entity_key
                                ORDER BY entity_times.TransactionDT
                                RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING
                            )
                        ) AS target_values
                    """)
                    cursor.execute("ALTER TABLE ieee_entity_features ADD PRIMARY KEY (entity_key, TransactionDT)")
                    cursor.execute(f"""
                        SELECT transactions.TransactionID, features.`{name}_平滑目标编码`
                        FROM ieee_transactions AS transactions
                        LEFT JOIN ieee_entity_features AS features
                            ON features.entity_key = transactions.`{entity}`
                            AND features.TransactionDT = transactions.TransactionDT
                    """)
                    result = pd.DataFrame(
                        cursor.fetchall(),
                        columns=[column[0] for column in cursor.description],
                    ).set_index("TransactionID")
                    outputs.append(result.apply(pd.to_numeric))
                    cursor.execute("DROP TEMPORARY TABLE ieee_entity_features, ieee_entity_times")
                    continue
                cursor.execute(f"""
                    CREATE TEMPORARY TABLE ieee_entity_windows AS
                    SELECT entity_key, TransactionDT,
                        SUM(transaction_count) OVER recent_hour AS hour_count,
                        SUM(transaction_count) OVER recent_day AS day_count,
                        SUM(amount_sum) OVER recent_day AS day_amount,
                        SUM(transaction_count) OVER entity_history AS cumulative_count,
                        SUM(amount_sum) OVER entity_history AS cumulative_amount,
                        SUM(amount_square_sum) OVER entity_history AS cumulative_amount_squares,
                        SUM(fraud_count) OVER mature_labels AS mature_fraud,
                        SUM(label_count) OVER mature_labels AS mature_labels
                    FROM ieee_entity_times
                    WINDOW
                        recent_hour AS (PARTITION BY entity_key ORDER BY TransactionDT RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),
                        recent_day AS (PARTITION BY entity_key ORDER BY TransactionDT RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
                        entity_history AS (PARTITION BY entity_key ORDER BY TransactionDT ROWS UNBOUNDED PRECEDING),
                        mature_labels AS (PARTITION BY entity_key ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING)
                """)
                cursor.execute("ALTER TABLE ieee_entity_windows ADD PRIMARY KEY (entity_key, TransactionDT)")
                cursor.execute("""
                    CREATE TEMPORARY TABLE ieee_entity_previous AS
                    SELECT entity_key, TransactionDT, hour_count, day_count, day_amount, mature_fraud, mature_labels,
                        LAG(cumulative_count, 1, 0) OVER entity_history AS prior_count,
                        LAG(cumulative_amount, 1, 0) OVER entity_history AS prior_amount,
                        LAG(cumulative_amount_squares, 1, 0) OVER entity_history AS prior_amount_squares,
                        LAG(TransactionDT, 1, 0) OVER entity_history AS prior_time
                    FROM ieee_entity_windows
                    WINDOW entity_history AS (PARTITION BY entity_key ORDER BY TransactionDT)
                """)
                cursor.execute("ALTER TABLE ieee_entity_previous ADD PRIMARY KEY (entity_key, TransactionDT)")
                cursor.execute(f"""
                    CREATE TEMPORARY TABLE ieee_entity_features AS
                    SELECT cur.entity_key, cur.TransactionDT,
                        cur.hour_count AS hour_count,
                        cur.day_count AS day_count,
                        cur.day_amount AS day_amount,
                        cur.prior_count AS `{name}_历史频次`,
                        cur.prior_amount / NULLIF(cur.prior_count, 0) AS `{name}_历史金额均值`,
                        SQRT(GREATEST(cur.prior_amount_squares / NULLIF(cur.prior_count, 0)
                            - POW(cur.prior_amount / NULLIF(cur.prior_count, 0), 2), 0)) AS `{name}_历史金额标准差`,
                        CASE WHEN cur.prior_count = 0 THEN 1 ELSE 0 END AS `{name}_新颖性`,
                        CASE WHEN cur.prior_count = 0 THEN NULL ELSE cur.TransactionDT - cur.prior_time END AS `{name}_距上次秒数`,
                        cur.prior_count / NULLIF(g.global_history_count, 0) AS `{name}_历史频率编码`,
                        (COALESCE(cur.mature_fraud, 0) + 20 * g.global_mature_rate) /
                            (COALESCE(cur.mature_labels, 0) + 20) AS `{name}_平滑目标编码`
                    FROM ieee_entity_previous AS cur
                    JOIN ieee_global_features AS g USING (TransactionDT)
                """)
                cursor.execute("ALTER TABLE ieee_entity_features ADD PRIMARY KEY (entity_key, TransactionDT)")
                cursor.execute(f"""
                    SELECT transactions.TransactionID, features.*,
                        features.hour_count AS `{name}_1小时频次`,
                        features.day_count AS `{name}_24小时频次`,
                        features.day_amount AS `{name}_24小时金额`,
                        (transactions.TransactionAmt - features.`{name}_历史金额均值`) /
                            NULLIF(features.`{name}_历史金额标准差`, 0) AS `{name}_金额偏离度`
                    FROM ieee_transactions AS transactions
                    LEFT JOIN ieee_entity_features AS features
                        ON features.entity_key = transactions.`{entity}`
                        AND features.TransactionDT = transactions.TransactionDT
                """)
                result = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description]).set_index("TransactionID")
                outputs.append(result.drop(columns=["entity_key", "TransactionDT", "hour_count", "day_count", "day_amount"]).apply(pd.to_numeric))
                cursor.execute("DROP TEMPORARY TABLE ieee_entity_features, ieee_entity_previous, ieee_entity_windows, ieee_entity_times")
    finally:
        connection.close()
    result = pd.concat(outputs, axis=1).reindex(frame.TransactionID)
    result.index = frame.index
    result["相对日内小时"] = (frame.TransactionDT // 3600) % 24
    result["相对周内日"] = (frame.TransactionDT // 86400) % 7
    return result


def temporal_features_history(frame, output_path=None):
    """生成只依赖历史交易的特征，并可持久化复用。"""
    result = temporal_features(
        frame,
        label_cutoff=int(frame.TransactionDT.max()) + 1,
        label_delay=1,
        history_only=True,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        persisted = pd.concat([frame.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
        persisted.to_parquet(output_path, index=False)
    return result


def temporal_features_target_encoding(frame, history_features, label_cutoff, label_delay=7 * 86400):
    """仅计算截止期目标编码，并与已持久化的历史特征拼接。"""
    target = _target_encoding_features(frame, label_cutoff=label_cutoff, label_delay=label_delay)
    target_columns = [column for column in target if "平滑目标编码" in column]
    if len(target_columns) != len(ENTITIES):
        raise RuntimeError("目标编码查询未返回全部实体列")
    return pd.concat([history_features.loc[frame.index], target.loc[:, target_columns]], axis=1)


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
    suffix = f".sample-{sample_rows}" if sample_rows is not None else ""
    engineered = temporal_features(data, cutoff)
    engineered.to_parquet(Path(output_dir) / f"temporal_features{suffix}.parquet", index=False)
    diagnostics = uid_diagnostics(data)
    diagnostics.update({"新增特征数": len(engineered.columns), "标签冻结截止秒": int(cutoff), "标签延迟秒": 7 * 86400,
                        "状态": "数据接入与特征已生成；尚未完成建模评估", "时间口径": "相对时间，不声称真实星期或真实钟点"})
    (Path(output_dir) / f"preparation{suffix}.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics
