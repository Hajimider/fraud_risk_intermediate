"""按时间切分交易数据，比较模型并评估双阈值策略。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sql_analysis import analyze_transactions

ROOT = Path(__file__).resolve().parent
RAW_FEATURES = ["Amount", *[f"V{i}" for i in range(1, 29)]]
MODEL_FEATURES = ["Amount", "LogAmount", *[f"V{i}" for i in range(1, 29)]]
ROLLING_RATIOS = (0.4, 0.55, 0.7, 0.85, 1.0)
MODEL_CANDIDATES = (
    ("类别加权逻辑回归", "logistic", True),
    ("类别加权XGBoost", "xgboost", True),
    ("类别加权XGBoost（移除LogAmount）", "xgboost", False),
)


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
    if len(times) < 10 or any(a >= b for a, b in zip([0, *boundaries[:-1]], boundaries)):
        raise ValueError("滚动验证至少需要10个不同交易时间")

    windows = []
    previous_end = 0
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

    test_cutoff = times[boundaries[-2] - 1]
    test = ordered.loc[ordered["Time"] > test_cutoff]
    if test["Class"].nunique() < 2:
        raise ValueError("最后15%的测试集需要同时包含正常与欺诈样本")
    return windows, test


def make_model(kind, train_labels, trees, seed):
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    positive_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    return XGBClassifier(
        n_estimators=trees,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
    )


def cost_of_actions(
    y_true,
    scores,
    review_threshold,
    block_threshold,
    fraud_loss=100,
    false_block_cost=6,
    review_cost=1.5,
    review_capacity=0.02,
    review_recall=0.9,
    review_false_positive_rate=0.01,
):
    """按放行、复核、拦截统计笔数，并估算漏放和误拦成本。"""
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    action = np.where(
        score >= block_threshold,
        "拦截",
        np.where(score >= review_threshold, "复核", "放行"),
    )
    review = action == "复核"
    block = action == "拦截"
    # 复核会漏判欺诈，也会误判正常交易，不能按完美审核计算。
    missed = (
        int(((action == "放行") & (y == 1)).sum())
        + int((review & (y == 1)).sum()) * (1 - review_recall)
    )
    false_blocks = (
        int((block & (y == 0)).sum())
        + int((review & (y == 0)).sum()) * review_false_positive_rate
    )
    review_count = int(review.sum())
    block_count = int(block.sum())
    cost = missed * fraud_loss + false_blocks * false_block_cost + review_count * review_cost
    review_rate = review_count / len(y)
    return {
        "放行笔数": int((action == "放行").sum()),
        "复核笔数": review_count,
        "拦截笔数": block_count,
        "预计漏放欺诈": float(missed),
        "预计误拦正常": float(false_blocks),
        "复核率": review_rate,
        "是否超出复核容量": review_rate > review_capacity,
        "模拟成本": float(cost),
    }


def choose_policy(y_true, scores, review_capacity=0.02):
    """在验证集上选成本最低的双阈值策略，成本相同时优先少复核。"""
    quantiles = np.r_[np.linspace(0, 0.95, 20), 0.97, 0.98, 0.99, 0.995, 1.0]
    thresholds = np.unique(np.r_[
        np.quantile(scores, quantiles),
        np.nextafter(float(np.max(scores)), 2.0),
    ])
    best = None
    for i, review_threshold in enumerate(thresholds[:-1]):
        for block_threshold in thresholds[i + 1:]:
            result = cost_of_actions(
                y_true, scores, review_threshold, block_threshold,
                review_capacity=review_capacity,
            )
            if result["是否超出复核容量"] or not result["复核笔数"] or not result["拦截笔数"]:
                continue
            row = {
                "复核阈值": float(review_threshold),
                "拦截阈值": float(block_threshold),
                **result,
            }
            if best is None or (row["模拟成本"], row["复核率"]) < (best["模拟成本"], best["复核率"]):
                best = row
    if best is None:
        raise ValueError("没有满足复核容量的策略")
    return best


def classification_metrics(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(scores) >= 0.5
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


def run(data_path=None, rows=12000, trees=80, seed=42):
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到真实数据文件：{path}\n如果只是学习流程，请运行 python run.py demo")
        frame = pd.read_csv(path)
        missing = {"Time", "Class", *RAW_FEATURES} - set(frame.columns)
        if missing:
            raise ValueError(f"数据缺少字段：{', '.join(sorted(missing))}")
        source = "公开信用卡CSV"
    else:
        frame = make_demo_data(rows, seed)
        source = "合成教学数据"
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    frame[RAW_FEATURES] = frame[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    if frame["Time"].isna().any() or not np.isfinite(frame["Time"]).all():
        raise ValueError("Time必须是无缺失的有限数值")
    if frame[RAW_FEATURES].isna().any().any():
        raise ValueError("特征列包含缺失值或非数值，请先清洗数据")
    if frame["Class"].isna().any() or not frame["Class"].isin([0, 1]).all():
        raise ValueError("Class必须是无缺失的0/1标签")
    sql_summary, profile = analyze_transactions(frame)
    windows, test = rolling_time_splits(frame)
    scores_by_candidate = {name: [] for name, _, _ in MODEL_CANDIDATES}
    window_results = []
    window_summaries = []
    last_models = {}
    last_validation_scores = {}

    for window in windows:
        train, validation = window["train"], window["validation"]
        baseline_ap = float(validation["Class"].mean())
        window_summaries.append({
            "窗口": window["窗口"],
            "训练集笔数": len(train),
            "验证集笔数": len(validation),
            "欺诈笔数": int(validation["Class"].sum()),
            "欺诈率基线AP": baseline_ap,
        })
        for name, kind, include_log_amount in MODEL_CANDIDATES:
            feature_columns = MODEL_FEATURES if include_log_amount else RAW_FEATURES
            scaler = StandardScaler()
            x_train = scaler.fit_transform(build_features(train, include_log_amount))
            x_validation = scaler.transform(build_features(validation, include_log_amount))
            model = make_model(kind, train["Class"].to_numpy(), trees, seed)
            model.fit(x_train, train["Class"])
            validation_score = model.predict_proba(x_validation)[:, 1]
            ap = float(average_precision_score(validation["Class"], validation_score))
            scores_by_candidate[name].append(ap)
            window_results.append({"窗口": window["窗口"], "模型": name, "AP": ap})
            if window is windows[-1]:
                last_models[name] = (model, scaler, feature_columns)
                last_validation_scores[name] = validation_score

    comparison = []
    for name, scores in scores_by_candidate.items():
        comparison.append({
            "模型": name,
            "最近窗口AP": scores[-1],
            "平均AP": float(np.mean(scores)),
            "AP标准差": float(np.std(scores, ddof=1)),
        })
    winner = max(comparison, key=lambda row: row["平均AP"])["模型"]
    winning_model, scaler, selected_features = last_models[winner]
    last_validation = windows[-1]["validation"]
    policy = choose_policy(last_validation["Class"], last_validation_scores[winner])
    x_test = scaler.transform(build_features(test, include_log_amount="LogAmount" in selected_features))
    test_score = winning_model.predict_proba(x_test)[:, 1]
    test_cost = cost_of_actions(test.Class, test_score, policy["复核阈值"], policy["拦截阈值"])
    allow_all = cost_of_actions(test.Class, test_score, 2, 2)
    test_cost["全部放行模拟成本"] = allow_all["模拟成本"]
    test_cost["相对全部放行成本变化"] = (
        test_cost["模拟成本"] / allow_all["模拟成本"] - 1
        if allow_all["模拟成本"] else None
    )

    if "XGBoost" in winner:
        importance_values = winning_model.feature_importances_
        explanation = "XGBoost原生特征重要性（不代表因果关系）"
    else:
        importance_values = np.abs(winning_model.coef_[0])
        explanation = "逻辑回归标准化系数绝对值（不代表因果关系）"
    report = {
        "数据来源": source,
        "总笔数": len(frame),
        "欺诈笔数": int(frame.Class.sum()),
        "时间切分": {
            "训练集": len(windows[-1]["train"]),
            "验证集": len(last_validation),
            "测试集": len(test),
        },
        "滚动验证窗口": window_summaries,
        "滚动验证结果": window_results,
        "模型比较": comparison,
        "验证集选定模型": winner,
        "特征消融": [
            {
                "窗口": window["窗口"],
                "完整XGBoost_AP": scores_by_candidate["类别加权XGBoost"][index],
                "移除LogAmount_AP": scores_by_candidate["类别加权XGBoost（移除LogAmount）"][index],
                "AP变化": (
                    scores_by_candidate["类别加权XGBoost（移除LogAmount）"][index]
                    - scores_by_candidate["类别加权XGBoost"][index]
                ),
            }
            for index, window in enumerate(windows)
        ],
        "测试集指标": classification_metrics(test.Class, test_score),
        "验证集选定策略": policy,
        "测试集冻结策略表现": test_cost,
        "解释方法": explanation,
        "特征工程": ["LogAmount=log1p(交易金额)"] if "LogAmount" in selected_features else [],
        "数据质量": {
            "缺失值总数": sql_summary["缺失值总数"],
            "重复行数": sql_summary["重复行数"],
        },
        "SQL数据画像": sql_summary,
        "特征重要性": sorted(
            zip(selected_features, importance_values.tolist()),
            key=lambda pair: pair[1], reverse=True,
        )[:15],
        "成本假设": {
            "漏放欺诈损失": 100,
            "误拦正常成本": 6,
            "人工复核成本": 1.5,
            "人工复核召回率": 0.9,
            "人工复核误判率": 0.01,
            "最大复核比例": 0.02,
        },
        "风险分数说明": "类别加权模型输出未校准分数；策略成本是给定假设下的离线模拟。",
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame({"欺诈标签": test.Class.to_numpy(), "风险分数": test_score}).to_csv(
        report_dir / "test_scores.csv", index=False, encoding="utf-8-sig",
    )
    profile.to_csv(report_dir / "hour_profile.csv", index=False, encoding="utf-8-sig")
    return report
