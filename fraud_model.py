"""按时间切分交易数据，比较模型并评估双阈值策略。"""

from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
RAW_FEATURES = ["Amount", *[f"V{i}" for i in range(1, 29)]]
MODEL_FEATURES = ["Amount", "LogAmount", *[f"V{i}" for i in range(1, 29)]]
ROLLING_RATIOS = (0.4, 0.55, 0.7, 0.85, 1.0)


def make_demo_data(rows=12000, seed=42):
    x, y = make_classification(
        n_samples=rows,
        n_features=29,
        n_informative=12,
        n_redundant=5,
        weights=[0.995, 0.005],
        class_sep=1.4,
        random_state=seed,
    )
    frame = pd.DataFrame(x, columns=RAW_FEATURES)
    frame["Amount"] = np.exp(frame["Amount"] * 0.7 + 3)
    frame.insert(0, "Time", np.arange(rows, dtype=float) * 60)
    frame["Class"] = y
    return frame


def build_features(frame, include_log_amount=True):
    """保留原始特征，并对金额取对数。"""
    features = frame[RAW_FEATURES].copy()
    if include_log_amount:
        features["LogAmount"] = np.log1p(features["Amount"].clip(lower=0))
        return features[MODEL_FEATURES]
    return features[RAW_FEATURES]


def time_split(frame):
    """按唯一时间戳顺序切分，确保同一时刻不会跨数据集。"""
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    times = np.sort(ordered["Time"].unique())
    first = max(1, int(len(times) * 0.7))
    second = max(2, int(len(times) * 0.85))
    if len(times) < 4 or second >= len(times):
        raise ValueError("至少需要4个不同交易时间才能进行时间切分")
    first_time, second_time = times[first - 1], times[second - 1]
    train = ordered.loc[ordered.Time <= first_time]
    validation = ordered.loc[(ordered.Time > first_time) & (ordered.Time <= second_time)]
    test = ordered.loc[ordered.Time > second_time]
    return train, validation, test


def rolling_time_splits(frame):
    """构造三个扩展训练窗口，并保留最后15%作为唯一测试集。"""
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    times = np.sort(ordered["Time"].unique())
    boundaries = [int(len(times) * ratio) for ratio in ROLLING_RATIOS]
    if len(times) < 10 or any(a >= b for a, b in zip([0, *boundaries[:-1]], boundaries, strict=False)):
        raise ValueError("滚动验证至少需要10个不同交易时间")

    windows = []
    for index in range(len(boundaries) - 2):
        train_end = boundaries[index]
        validation_end = boundaries[index + 1]
        train_cutoff = times[train_end - 1]
        validation_cutoff = times[validation_end - 1]
        train = ordered.loc[ordered["Time"] <= train_cutoff]
        validation = ordered.loc[
            (ordered["Time"] > train_cutoff) & (ordered["Time"] <= validation_cutoff)
        ]
        if train["Class"].nunique() < 2 or validation["Class"].nunique() < 2:
            raise ValueError("每个滚动训练和验证窗口都需要包含正常与欺诈样本")
        windows.append({
            "窗口": f"{int(ROLLING_RATIOS[index] * 100)}%-{int(ROLLING_RATIOS[index + 1] * 100)}%",
            "train": train,
            "validation": validation,
        })

    _, _, test = time_split(ordered)
    if test["Class"].nunique() < 2:
        raise ValueError("最后15%的测试集需要同时包含正常与欺诈样本")
    return windows, test




@dataclass(frozen=True)
class CostParameters:
    fraud_loss: float = 100.0
    false_block_cost: float = 6.0
    review_cost: float = 1.5
    review_capacity: float = 0.02
    review_recall: float = 0.9
    review_false_positive_rate: float = 0.01
    min_fraud_loss: float = 1.0
    fraud_loss_multiplier: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"成本参数 {key} 必须为有限非负数")
        for key in ("review_capacity", "review_recall", "review_false_positive_rate"):
            if getattr(self, key) > 1:
                raise ValueError(f"成本参数 {key} 必须在0到1之间")


def action_cost_arrays(y_true, scores, amounts, parameters):
    y, score = np.asarray(y_true), np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y):
        raise ValueError("标签和分数必须是等长非空一维数组")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("标签必须为0/1，分数必须为有限数值")
    if amounts is None:
        loss = np.full(len(y), parameters.fraud_loss * parameters.fraud_loss_multiplier)
    else:
        amount = np.asarray(amounts, dtype=float)
        if amount.shape != y.shape or not np.isfinite(amount).all():
            raise ValueError("金额必须是等长有限数组")
        loss = np.maximum(amount, parameters.min_fraud_loss) * parameters.fraud_loss_multiplier
    allow = np.where(y == 1, loss, 0.0)
    review = np.where(y == 1, loss * (1 - parameters.review_recall),
                      parameters.review_false_positive_rate * parameters.false_block_cost) + parameters.review_cost
    block = np.where(y == 0, parameters.false_block_cost, 0.0)
    return y.astype(int), score, allow, review, block


def cost_of_actions(
    y_true,
    scores,
    review_threshold,
    block_threshold,
    amounts=None,
    cost_params=None,
    **overrides,
):
    """按动作估算成本；提供金额时漏放损失按金额加权。"""
    parameters = replace(cost_params or CostParameters(), **overrides)
    y, score, allow_cost, review_cost, block_cost = action_cost_arrays(y_true, scores, amounts, parameters)
    if np.isnan(review_threshold) or np.isnan(block_threshold) or review_threshold > block_threshold:
        raise ValueError("复核阈值不能大于拦截阈值")
    action = np.where(
        score >= block_threshold,
        "拦截",
        np.where(score >= review_threshold, "复核", "放行"),
    )
    review = action == "复核"
    block = action == "拦截"
    # 复核会漏判欺诈，也会误判正常交易，不能按完美审核计算。
    missed_mask = (action == "放行") & (y == 1)
    reviewed_fraud = review & (y == 1)
    fraud_cost = float(allow_cost[missed_mask].sum() + allow_cost[reviewed_fraud].sum() * (1 - parameters.review_recall))
    missed = int(missed_mask.sum()) + int(reviewed_fraud.sum()) * (1 - parameters.review_recall)
    false_blocks = (
        int((block & (y == 0)).sum())
        + int((review & (y == 0)).sum()) * parameters.review_false_positive_rate
    )
    review_count = int(review.sum())
    block_count = int(block.sum())
    cost = float(allow_cost[action == "放行"].sum() + review_cost[review].sum() + block_cost[block].sum())
    review_rate = review_count / len(y)
    return {
        "放行笔数": int((action == "放行").sum()),
        "复核笔数": review_count,
        "拦截笔数": block_count,
        "预计漏放欺诈": float(missed),
        "预计误拦正常": float(false_blocks),
        "复核率": review_rate,
        "是否超出复核容量": review_rate > parameters.review_capacity,
        "模拟成本": float(cost),
        "漏放损失金额": float(fraud_cost),
    }


def choose_policy(y_true, scores, review_capacity=None, amounts=None, cost_params=None, **overrides):
    """遍历全体唯一分数阈值，利用前缀成本和滑动最大值精确搜索。"""
    if review_capacity is not None:
        overrides["review_capacity"] = review_capacity
    parameters = replace(cost_params or CostParameters(), **overrides)
    y, score, allow_cost, review_cost, block_cost = action_cost_arrays(y_true, scores, amounts, parameters)
    order = np.argsort(score, kind="stable")
    y, score = y[order], score[order]
    thresholds = np.unique(np.r_[score, np.nextafter(score[-1], np.inf)])
    starts = np.searchsorted(score, thresholds, side="left")
    cap = int(np.floor(len(y) * parameters.review_capacity))
    allow_cost, review_cost, block_cost = allow_cost[order], review_cost[order], block_cost[order]
    review_prefix = np.r_[0.0, np.cumsum(review_cost - allow_cost)]
    block_prefix = np.r_[0.0, np.cumsum(block_cost - allow_cost)]
    best = None
    base = float(allow_cost.sum())
    candidates = deque()
    next_review = 0
    for block_threshold, block_start in zip(thresholds, starts, strict=False):
        if block_start > len(y):
            break
        while next_review < len(starts) and starts[next_review] < block_start:
            idx = int(starts[next_review])
            if idx < block_start:
                while candidates and review_prefix[starts[candidates[-1]]] <= review_prefix[idx]:
                    candidates.pop()
                candidates.append(next_review)
            next_review += 1
        min_start = block_start - cap
        while candidates and starts[candidates[0]] < min_start:
            candidates.popleft()
        if candidates:
            review_idx = candidates[0]
            review_start = int(starts[review_idx])
            incremental = review_prefix[block_start] - review_prefix[review_start]
            block_increment = block_prefix[-1] - block_prefix[block_start]
            total_cost = base + incremental + block_increment
            review_count = block_start - review_start
            row = {
                "复核阈值": float(thresholds[review_idx]),
                "拦截阈值": float(block_threshold),
                "放行笔数": review_start,
                "复核笔数": review_count,
                "拦截笔数": len(y) - block_start,
                "复核率": review_count / len(y),
                "是否超出复核容量": review_count / len(y) > parameters.review_capacity,
                "模拟成本": float(total_cost),
            }
            if row["复核笔数"] or row["拦截笔数"]:
                if best is None or (row["模拟成本"], row["复核率"]) < (best["模拟成本"], best["复核率"]):
                    best = row
        if block_start < len(y):
            zero_review = {
                "复核阈值": float(block_threshold),
                "拦截阈值": float(block_threshold),
                "放行笔数": block_start,
                "复核笔数": 0,
                "拦截笔数": len(y) - block_start,
                "复核率": 0.0,
                "是否超出复核容量": False,
                "模拟成本": float(base + block_prefix[-1] - block_prefix[block_start]),
            }
            if best is None or (zero_review["模拟成本"], zero_review["复核率"]) < (best["模拟成本"], best["复核率"]):
                best = zero_review
    if best is None:
        raise ValueError("没有满足复核容量的策略")
    return {
        "复核阈值": best["复核阈值"], "拦截阈值": best["拦截阈值"],
        **cost_of_actions(y_true, scores, best["复核阈值"], best["拦截阈值"], amounts=amounts, cost_params=parameters),
    }


def expected_loss_policy(
    y_true, scores, amounts, cost_params=None, review_capacity=None,
    shadow_price=None, **overrides
):
    """按逐笔概率与金额最小化期望损失，并约束复核容量。

    ``shadow_price`` 由验证集搜索后可冻结传入测试集，避免按测试集分布重新配额。
    """
    if review_capacity is not None:
        overrides["review_capacity"] = review_capacity
    parameters = replace(cost_params or CostParameters(), **overrides)
    y, score, allow_cost, review_cost, block_cost = action_cost_arrays(y_true, scores, amounts, parameters)
    probability = np.clip(score.astype(float), 0.0, 1.0)
    loss = np.maximum(np.asarray(amounts, dtype=float), parameters.min_fraud_loss) * parameters.fraud_loss_multiplier
    review_base = (
        probability * loss * (1 - parameters.review_recall)
        + (1 - probability) * parameters.review_false_positive_rate * parameters.false_block_cost
        + parameters.review_cost
    )
    allow_expected = probability * loss
    block_expected = (1 - probability) * parameters.false_block_cost

    def actions(lambda_price):
        costs = np.column_stack((allow_expected, review_base + lambda_price, block_expected))
        return np.argmin(costs, axis=1)

    if shadow_price is None:
        actions_at_zero = actions(0.0)
        capacity = parameters.review_capacity
        if (actions_at_zero == 1).mean() <= capacity:
            shadow_price = 0.0
        else:
            upper = float(max(np.max(loss), parameters.false_block_cost, parameters.review_cost, 1.0) * 2)
            while (actions(upper) == 1).mean() > capacity and upper < 1e12:
                upper *= 2
            lower = 0.0
            for _ in range(60):
                middle = (lower + upper) / 2
                if (actions(middle) == 1).mean() > capacity:
                    lower = middle
                else:
                    upper = middle
            shadow_price = upper
    elif not np.isfinite(shadow_price) or shadow_price < 0:
        raise ValueError("冻结影子价格必须是非负有限数")
    chosen = actions(shadow_price)
    action_names = np.array(["放行", "复核", "拦截"])
    selected = action_names[chosen]
    review = selected == "复核"
    block = selected == "拦截"
    missed_mask = (selected == "放行") & (y == 1)
    reviewed_fraud = review & (y == 1)
    fraud_cost = float(allow_cost[missed_mask].sum() + allow_cost[reviewed_fraud].sum() * (1 - parameters.review_recall))
    missed = int(missed_mask.sum()) + int(reviewed_fraud.sum()) * (1 - parameters.review_recall)
    false_blocks = float((block & (y == 0)).sum() + (review & (y == 0)).sum() * parameters.review_false_positive_rate)
    cost = float(allow_cost[selected == "放行"].sum() + review_cost[review].sum() + block_cost[block].sum())
    return {
        "策略类型": "期望损失策略",
        "容量影子价格λ": float(shadow_price),
        "放行笔数": int((selected == "放行").sum()),
        "复核笔数": int(review.sum()),
        "拦截笔数": int(block.sum()),
        "预计漏放欺诈": float(missed),
        "预计误拦正常": false_blocks,
        "复核率": float(review.mean()),
        "是否超出复核容量": bool(review.mean() > parameters.review_capacity),
        "模拟成本": cost,
        "漏放损失金额": fraud_cost,
    }




def stratified_bootstrap_intervals(
    y_true, scores, amounts, review_threshold, block_threshold,
    iterations=1000, seed=42, cost_params=None,
):
    """按类别分层重采样，返回测试指标和冻结策略成本的百分位区间。"""
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    amount = np.asarray(amounts, dtype=float)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("平均精确率AP", "ROC-AUC", "精确率", "召回率", "模拟成本")}
    for _ in range(iterations):
        idx = np.r_[rng.choice(groups[0], len(groups[0]), replace=True), rng.choice(groups[1], len(groups[1]), replace=True)]
        metric = classification_metrics(y[idx], score[idx], threshold=review_threshold)
        cost = cost_of_actions(y[idx], score[idx], review_threshold, block_threshold,
                               amounts=amount[idx], cost_params=cost_params)
        for key in samples:
            samples[key].append(metric.get(key) if key != "模拟成本" else cost[key])
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in samples.items()
    }


def paired_ap_difference_interval(y_true, first_scores, second_scores, iterations=1000, seed=42):
    y = np.asarray(y_true, dtype=int)
    first, second = np.asarray(first_scores), np.asarray(second_scores)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iterations):
        idx = np.r_[rng.choice(groups[0], len(groups[0]), replace=True), rng.choice(groups[1], len(groups[1]), replace=True)]
        diffs.append(average_precision_score(y[idx], first[idx]) - average_precision_score(y[idx], second[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def classification_metrics(y_true, scores, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "平均精确率AP": float(average_precision_score(y, scores)) if both_classes else None,
        "ROC-AUC": float(roc_auc_score(y, scores)) if both_classes else None,
        "精确率": float(precision_score(y, pred, zero_division=0)),
        "召回率": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "真阴性": int(tn),
        "假阳性": int(fp),
        "假阴性": int(fn),
        "真阳性": int(tp),
    }


def run(data_path=None, rows=12000, trees=80, seed=42, output_dir=None, tuning_trials=3, bootstrap_iterations=1000):
    from experiments import run_experiment
    return run_experiment(data_path, rows, trees, seed, output_dir or ROOT / "reports", tuning_trials, bootstrap_iterations)
