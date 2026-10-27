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

ROOT = Path(__file__).resolve().parent
RAW_FEATURES = ["Amount", *[f"V{i}" for i in range(1, 29)]]
MODEL_FEATURES = ["Amount", "LogAmount", *[f"V{i}" for i in range(1, 29)]]


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


def build_features(frame):
    """保留原始特征，并对金额取对数。"""
    features = frame[RAW_FEATURES].copy()
    features["LogAmount"] = np.log1p(features["Amount"].clip(lower=0))
    return features[MODEL_FEATURES]


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
    frame[RAW_FEATURES] = frame[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    if frame[RAW_FEATURES].isna().any().any():
        raise ValueError("特征列包含缺失值或非数值，请先清洗数据")
    if frame["Class"].isna().any() or not frame["Class"].isin([0, 1]).all():
        raise ValueError("Class必须是无缺失的0/1标签")
    train, validation, test = time_split(frame)
    if any(split.Class.nunique() < 2 for split in (train, validation, test)):
        raise ValueError("训练、验证和测试集都需要同时包含正常与欺诈样本；请增加数据量")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(build_features(train))
    x_valid = scaler.transform(build_features(validation))
    x_test = scaler.transform(build_features(test))
    positive_weight = (train.Class == 0).sum() / (train.Class == 1).sum()
    models = {
        "类别加权逻辑回归": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        ),
        "类别加权XGBoost": XGBClassifier(
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
        ),
    }
    validation_scores = {}
    comparison = []
    for name, model in models.items():
        model.fit(x_train, train.Class)
        valid_score = model.predict_proba(x_valid)[:, 1]
        validation_scores[name] = valid_score
        comparison.append({
            "模型": name,
            "验证集AP": float(average_precision_score(validation.Class, valid_score)),
        })
    winner = max(comparison, key=lambda row: row["验证集AP"])["模型"]
    winning_model = models[winner]
    policy = choose_policy(validation.Class, validation_scores[winner])
    test_score = winning_model.predict_proba(x_test)[:, 1]
    test_cost = cost_of_actions(test.Class, test_score, policy["复核阈值"], policy["拦截阈值"])
    allow_all = cost_of_actions(test.Class, test_score, 2, 2)
    test_cost["全部放行模拟成本"] = allow_all["模拟成本"]
    test_cost["相对全部放行成本变化"] = (
        test_cost["模拟成本"] / allow_all["模拟成本"] - 1
        if allow_all["模拟成本"] else None
    )

    if winner == "类别加权XGBoost":
        importance_values = winning_model.feature_importances_
        explanation = "XGBoost原生特征重要性（不代表因果关系）"
    else:
        importance_values = np.abs(winning_model.coef_[0])
        explanation = "逻辑回归标准化系数绝对值（不代表因果关系）"
    report = {
        "数据来源": source,
        "总笔数": len(frame),
        "欺诈笔数": int(frame.Class.sum()),
        "时间切分": {"训练集": len(train), "验证集": len(validation), "测试集": len(test)},
        "模型比较": comparison,
        "验证集选定模型": winner,
        "测试集指标": classification_metrics(test.Class, test_score),
        "验证集选定策略": policy,
        "测试集冻结策略表现": test_cost,
        "解释方法": explanation,
        "特征工程": ["LogAmount=log1p(交易金额)"],
        "数据质量": {
            "缺失值总数": int(frame[RAW_FEATURES].isna().sum().sum()),
            "重复行数": int(frame.duplicated().sum()),
        },
        "特征重要性": sorted(
            zip(MODEL_FEATURES, importance_values.tolist()),
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
    profile = (
        frame.assign(相对小时=(frame["Time"] // 3600).astype(int))
        .groupby("相对小时")
        .agg(交易笔数=("Class", "size"), 欺诈笔数=("Class", "sum"), 平均金额=("Amount", "mean"))
    )
    profile["欺诈比例"] = profile["欺诈笔数"] / profile["交易笔数"]
    profile.reset_index().to_csv(report_dir / "hour_profile.csv", index=False, encoding="utf-8-sig")
    return report
