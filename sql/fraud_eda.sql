-- MySQL 8.0+。四条查询按顺序返回规模、重复、金额与相对小时画像。
SELECT
    COUNT(*) AS total_count,
    COALESCE(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END), 0) AS fraud_count
FROM transactions;

SELECT
    COALESCE(SUM(copy_count - 1), 0) AS duplicate_rows,
    COALESCE(SUM(CASE WHEN Class = 1 THEN copy_count - 1 ELSE 0 END), 0) AS duplicate_fraud_rows
FROM (
    SELECT Class, COUNT(*) AS copy_count
    FROM transactions
    GROUP BY `Time`, Amount, Class,
        V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
        V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
        V21, V22, V23, V24, V25, V26, V27, V28
    HAVING COUNT(*) > 1
) AS repeated_transactions;

-- 连续分位数采用线性插值，位置为 1 + (非空笔数 - 1) * q。
WITH ranked AS (
    SELECT
        Class,
        Amount,
        ROW_NUMBER() OVER (PARTITION BY Class ORDER BY Amount IS NULL, Amount) AS amount_rank,
        COUNT(Amount) OVER (PARTITION BY Class) AS amount_count
    FROM transactions
),
quantiles AS (
    SELECT
        Class,
        COUNT(*) AS transaction_count,
        AVG(Amount) AS mean_amount,
        MAX(amount_count) AS amount_count,
        MAX(CASE WHEN amount_rank = FLOOR(1 + (amount_count - 1) * 0.50) THEN Amount END) AS q50_lower,
        MAX(CASE WHEN amount_rank = CEIL(1 + (amount_count - 1) * 0.50) THEN Amount END) AS q50_upper,
        MAX(CASE WHEN amount_rank = FLOOR(1 + (amount_count - 1) * 0.25) THEN Amount END) AS q25_lower,
        MAX(CASE WHEN amount_rank = CEIL(1 + (amount_count - 1) * 0.25) THEN Amount END) AS q25_upper,
        MAX(CASE WHEN amount_rank = FLOOR(1 + (amount_count - 1) * 0.75) THEN Amount END) AS q75_lower,
        MAX(CASE WHEN amount_rank = CEIL(1 + (amount_count - 1) * 0.75) THEN Amount END) AS q75_upper,
        MAX(CASE WHEN amount_rank = FLOOR(1 + (amount_count - 1) * 0.90) THEN Amount END) AS q90_lower,
        MAX(CASE WHEN amount_rank = CEIL(1 + (amount_count - 1) * 0.90) THEN Amount END) AS q90_upper
    FROM ranked
    GROUP BY Class
)
SELECT
    CAST(Class AS SIGNED) AS fraud_label,
    transaction_count,
    mean_amount,
    q50_lower + (1 + (amount_count - 1) * 0.50 - FLOOR(1 + (amount_count - 1) * 0.50)) * (q50_upper - q50_lower) AS median_amount,
    q25_lower + (1 + (amount_count - 1) * 0.25 - FLOOR(1 + (amount_count - 1) * 0.25)) * (q25_upper - q25_lower) AS amount_q1,
    q75_lower + (1 + (amount_count - 1) * 0.75 - FLOOR(1 + (amount_count - 1) * 0.75)) * (q75_upper - q75_lower) AS amount_q3,
    q90_lower + (1 + (amount_count - 1) * 0.90 - FLOOR(1 + (amount_count - 1) * 0.90)) * (q90_upper - q90_lower) AS amount_p90
FROM quantiles
ORDER BY Class;

SELECT
    CAST(FLOOR(`Time` / 3600) AS SIGNED) AS relative_hour,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    AVG(Amount) AS mean_amount
FROM transactions
GROUP BY relative_hour
ORDER BY relative_hour;
