CREATE OR REPLACE TEMP TABLE sql_summary AS
SELECT
    COUNT(*)::BIGINT AS total_count,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)::BIGINT AS fraud_count
FROM transactions;

CREATE OR REPLACE TEMP TABLE sql_duplicate_rows AS
SELECT COALESCE(SUM(copy_count - 1), 0)::BIGINT AS duplicate_rows
FROM (
    SELECT *, COUNT(*) AS copy_count
    FROM transactions
    GROUP BY ALL
    HAVING COUNT(*) > 1
);

CREATE OR REPLACE TEMP TABLE sql_amount_by_class AS
SELECT
    Class::INTEGER AS fraud_label,
    COUNT(*)::BIGINT AS transaction_count,
    AVG(Amount)::DOUBLE AS mean_amount,
    MEDIAN(Amount)::DOUBLE AS median_amount,
    QUANTILE_CONT(Amount, 0.25)::DOUBLE AS amount_q1,
    QUANTILE_CONT(Amount, 0.75)::DOUBLE AS amount_q3,
    QUANTILE_CONT(Amount, 0.90)::DOUBLE AS amount_p90
FROM transactions
GROUP BY Class
ORDER BY Class;

CREATE OR REPLACE TEMP TABLE sql_hour_profile AS
SELECT
    FLOOR(Time / 3600)::INTEGER AS relative_hour,
    COUNT(*)::BIGINT AS transaction_count,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)::BIGINT AS fraud_count,
    AVG(Amount)::DOUBLE AS mean_amount
FROM transactions
GROUP BY relative_hour
ORDER BY relative_hour;
