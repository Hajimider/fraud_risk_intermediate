-- MySQL 8.0+，适用于 IEEE-CIS 交易与身份字段合并后的 transactions 表。
-- 同一 TransactionDT 的交易不互相提供特征，避免并列时间戳泄漏。

-- 1. 同卡过去 1 小时交易量和金额（TransactionDT 单位为秒）。
SELECT
    TransactionID,
    card1,
    COUNT(*) OVER (
        PARTITION BY card1 ORDER BY TransactionDT
        RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
    ) AS card_txn_count_1h,
    SUM(TransactionAmt) OVER (
        PARTITION BY card1 ORDER BY TransactionDT
        RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
    ) AS card_amount_sum_1h
FROM transactions;

-- 2. 同卡此前交易金额偏离度，窗口只包含更早的交易。
SELECT
    TransactionID,
    card1,
    (TransactionAmt - AVG(TransactionAmt) OVER prior_txns)
        / NULLIF(STDDEV_POP(TransactionAmt) OVER prior_txns, 0) AS card_amount_zscore,
    COUNT(*) OVER prior_txns AS card_history_count
FROM transactions
WHERE card1 IS NOT NULL
WINDOW prior_txns AS (
    PARTITION BY card1 ORDER BY TransactionDT
    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
);

-- 3. 新设备标记和距该设备上次出现的时间。
SELECT
    TransactionID,
    DeviceInfo,
    CASE WHEN COUNT(*) OVER device_history = 0 THEN 1 ELSE 0 END AS is_new_device,
    TransactionDT - MAX(TransactionDT) OVER device_history AS seconds_since_device_seen
FROM transactions
WHERE DeviceInfo IS NOT NULL
WINDOW device_history AS (
    PARTITION BY DeviceInfo ORDER BY TransactionDT
    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
);

-- 4. 简化账户键示例，不等同真实用户。完整管线另加入D1推导的首笔相对日。
WITH keyed AS (
    SELECT *, CONCAT(CAST(card1 AS CHAR), ':', COALESCE(CAST(addr1 AS CHAR), 'NA')) AS approximate_uid
    FROM transactions
    WHERE card1 IS NOT NULL
)
SELECT
    TransactionID,
    approximate_uid,
    COUNT(*) OVER (
        PARTITION BY approximate_uid ORDER BY TransactionDT
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_uid_txn_count
FROM keyed;

-- 5. 同卡欺诈率目标编码。调用方须设置 @label_cutoff_dt 为基础模型拟合截止秒。
-- 未设置截止时安全地不使用任何标签。固定7天成熟延迟只是演示假设。
WITH prior AS (
    SELECT
        *,
        SUM(CASE WHEN TransactionDT <= COALESCE(@label_cutoff_dt, -1) THEN isFraud END) OVER (
            PARTITION BY card1 ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 604800 PRECEDING
        ) AS prior_fraud_count,
        COUNT(CASE WHEN TransactionDT <= COALESCE(@label_cutoff_dt, -1) THEN isFraud END) OVER (
            PARTITION BY card1 ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 604800 PRECEDING
        ) AS prior_labeled_count,
        AVG(CASE WHEN TransactionDT <= COALESCE(@label_cutoff_dt, -1) THEN isFraud END) OVER (
            ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 604800 PRECEDING
        ) AS prior_global_rate
    FROM transactions
    WHERE card1 IS NOT NULL
)
SELECT
    TransactionID,
    (COALESCE(prior_fraud_count, 0) + 20 * COALESCE(prior_global_rate, 0))
        / (COALESCE(prior_labeled_count, 0) + 20) AS card_fraud_rate_smoothed
FROM prior;
