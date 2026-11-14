"""时间验证、等预算调参、同模型校准与报告输出。测试段不参与任何选择。"""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_model import (
    RAW_FEATURES, CostParameters, build_features, classification_metrics,
    choose_policy, cost_of_actions, make_demo_data, paired_ap_difference_interval,
    rolling_time_splits, stratified_bootstrap_intervals,
)
from sql_analysis import analyze_transactions


def training_parts(frame):
    """训练段内部依时间分为70%拟合、15%早停、15%校准；不拆并列时刻。"""
    times = np.sort(frame.Time.unique())
    if len(times) < 10:
        raise ValueError("训练内部切分至少需要10个不同时间戳")
    a, b = times[int(len(times) * .7) - 1], times[int(len(times) * .85) - 1]
    parts = (frame.loc[frame.Time <= a], frame.loc[(frame.Time > a) & (frame.Time <= b)], frame.loc[frame.Time > b])
    if any(part.Class.nunique() != 2 for part in parts):
        raise ValueError("训练、早停和校准子段均需同时包含两类；请增加样本")
    return parts


def features(frame, mode):
    value = build_features(frame)
    return value.drop(columns={"amount": ["LogAmount"], "log": ["Amount"], "both": []}[mode])


def fit_candidate(train, validation, kind, mode, config, trees, seed):
    fit, stop, calibration = training_parts(train)
    scaler = StandardScaler()
    x = scaler.fit_transform(features(fit, mode))
    if kind == "logistic":
        model = LogisticRegression(C=config["C"], max_iter=1500, class_weight="balanced", random_state=seed)
        model.fit(x, fit.Class)
    else:
        model = XGBClassifier(
            n_estimators=trees, max_depth=config["max_depth"], learning_rate=config["learning_rate"],
            subsample=.85, colsample_bytree=.85,
            scale_pos_weight=float((fit.Class == 0).sum() / (fit.Class == 1).sum()),
            eval_metric="aucpr", early_stopping_rounds=max(5, min(20, trees // 4)),
            tree_method="hist", n_jobs=2, random_state=seed,
        )
        model.fit(x, fit.Class, eval_set=[(scaler.transform(features(stop, mode)), stop.Class)], verbose=False)
    predict = lambda frame: model.predict_proba(scaler.transform(features(frame, mode)))[:, 1]
    return model, scaler, predict(validation), predict(calibration), calibration, (len(fit), len(stop), len(calibration))


def reliability_bins(y, score, bins=10):
    data = pd.DataFrame({"标签": np.asarray(y), "分数": np.asarray(score)})
    # 分位箱对极不平衡样本更可读；相同分数始终归入同一箱。
    edges = np.unique(np.quantile(data["分数"], np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return [{"平均预测风险": float(data["分数"].mean()), "实际欺诈比例": float(data["标签"].mean()), "交易笔数": len(data)}]
    data["箱"] = pd.cut(data["分数"], edges, include_lowest=True)
    return [
        {"平均预测风险": float(group["分数"].mean()), "实际欺诈比例": float(group["标签"].mean()), "交易笔数": len(group)}
        for _, group in data.groupby("箱", observed=True)
    ]


def calibration_metrics(y, score):
    bins = reliability_bins(y, score)
    return {"Brier": float(brier_score_loss(y, score)),
            "ECE": float(sum(row["交易笔数"] * abs(row["平均预测风险"] - row["实际欺诈比例"]) for row in bins) / len(y)),
            "可靠性分箱": bins}


def fit_calibrator(scores, labels, method):
    if method == "Isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        return model.predict
    # 对数几率上的 Platt 映射可直接校正类别加权导致的先验偏移。
    def logit(value):
        clipped = np.clip(value, 1e-8, 1 - 1e-8)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1500).fit(logit(np.asarray(scores)), labels)
    return lambda value: model.predict_proba(logit(np.asarray(value)))[:, 1]


def error_analysis(frame, score, threshold):
    data = frame.copy()
    data["风险分数"] = score
    groups = {"漏判": data.loc[(data.Class == 1) & (score < threshold)],
              "误报": data.loc[(data.Class == 0) & (score >= threshold)]}
    summary = []
    for name, group in groups.items():
        summary.append({"类型": name, "笔数": len(group), "金额合计": float(group.Amount.sum()),
                        "金额中位数": float(group.Amount.median()) if len(group) else 0.,
                        "金额P90": float(group.Amount.quantile(.9)) if len(group) else 0.})
    missed = groups["漏判"].sort_values("Amount", ascending=False)
    total_loss = float(missed.Amount.clip(lower=1).sum())
    return {"金额分布": summary, "漏放损失前3笔占比": float(missed.head(3).Amount.clip(lower=1).sum() / total_loss) if total_loss else 0.,
            "漏判案例": missed[["Time", "Amount", "风险分数"]].head(10).rename(columns={"Time": "相对秒", "Amount": "金额"}).to_dict("records"),
            "口径": "漏判/误报按复核告警阈值统计，区别于人工审核后预计漏损"}


def sensitivity_analysis(validation, validation_score, test, test_score, parameters):
    rows = []
    for multiplier in (.5, 1., 2.):
        for false_block_cost in (3., 6., 12.):
            changed = replace(parameters, fraud_loss_multiplier=multiplier, false_block_cost=false_block_cost)
            policy = choose_policy(validation.Class, validation_score, amounts=validation.Amount, cost_params=changed)
            evaluated = cost_of_actions(test.Class, test_score, policy["复核阈值"], policy["拦截阈值"], amounts=test.Amount, cost_params=changed)
            rows.append({"欺诈损失金额倍率": multiplier, "误拦成本": false_block_cost,
                         "复核阈值": policy["复核阈值"], "拦截阈值": policy["拦截阈值"],
                         "验证成本": policy["模拟成本"], "测试复核率": evaluated["复核率"],
                         "测试成本": evaluated["模拟成本"]})
    return rows


def run_experiment(data_path, rows, trees, seed, sql_backend, output_dir, tuning_trials, bootstrap_iterations):
    if tuning_trials not in (1, 2, 3):
        raise ValueError("调参预算支持1、2、3个候选；两类模型始终使用相同预算")
    if bootstrap_iterations < 1 or trees < 1:
        raise ValueError("树数和重采样次数必须为正整数")
    if data_path is not None:
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到数据：{path}；教学流程请运行 python run.py demo")
        frame = pd.read_csv(path)
        source = "公开信用卡CSV"
    else:
        frame, source = make_demo_data(rows, seed), "合成教学数据"
    columns = ["Time", "Class", *RAW_FEATURES]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"数据缺少字段：{', '.join(sorted(missing))}")
    frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame.to_numpy()).all() or not frame.Class.isin([0, 1]).all():
        raise ValueError("数据必须为无缺失有限数值，Class必须为0/1")
    summary, profile = analyze_transactions(frame, backend=sql_backend)
    windows, test = rolling_time_splits(frame)
    configurations = {
        "logistic": [{"C": value} for value in (.1, 1., 10.)][:tuning_trials],
        "xgboost": [{"max_depth": depth, "learning_rate": rate} for depth, rate in ((3, .05), (4, .08), (5, .12))][:tuning_trials],
    }
    trials, records = [], {}
    for kind in ("logistic", "xgboost"):
        for mode in ("amount", "log", "both"):
            for index, config in enumerate(configurations[kind]):
                print(f"调参：{kind} / {mode} / 候选{index + 1}", flush=True)
                key = (kind, mode, index)
                records[key] = []
                for window in windows:
                    result = fit_candidate(window["train"], window["validation"], kind, mode, config, trees, seed)
                    model, scaler, score, calibration_score, calibration, sizes = result
                    ap = float(average_precision_score(window["validation"].Class, score))
                    records[key].append((ap, score, calibration_score))
                    trials.append({"模型": kind, "金额特征": mode, "候选": index + 1, "参数": config,
                                   "窗口": window["窗口"], "AP": ap,
                                   "实际树数": int(model.best_iteration + 1) if kind == "xgboost" else None,
                                   "内部切分": dict(zip(["拟合", "早停", "校准"], sizes))})
    mean_ap = lambda key: float(np.mean([item[0] for item in records[key]]))
    selected_keys = {}
    for kind in ("logistic", "xgboost"):
        # LR 的两种金额表达二选一，both只作消融证据。
        selected_keys[kind] = max((key for key in records if key[0] == kind and (kind != "logistic" or key[1] != "both")), key=mean_ap)
    lr, xgb = selected_keys["logistic"], selected_keys["xgboost"]
    validation = windows[-1]["validation"]
    paired = paired_ap_difference_interval(validation.Class, records[lr][-1][1], records[xgb][-1][1], iterations=bootstrap_iterations, seed=seed)
    winner_key = lr if paired[0] <= 0 <= paired[1] else max(selected_keys.values(), key=mean_ap)
    reason = "配对AP差异区间包含0，按预设规则优先简单模型；不等同于证明两模型等效" if winner_key == lr and paired[0] <= 0 <= paired[1] else "配对AP差异区间不含0，按滚动验证平均AP选择"
    kind, mode, index = winner_key
    winner = {"logistic": "类别加权逻辑回归", "xgboost": "类别加权XGBoost"}[kind]
    model, scaler, validation_raw, calibration_score, calibration, sizes = fit_candidate(
        windows[-1]["train"], validation, kind, mode, configurations[kind][index], trees, seed,
    )
    pooled_labels = np.concatenate([window["validation"].Class for window in windows[:2]])
    pooled_scores = np.concatenate([row[1] for row in records[winner_key][:2]])
    methods = {"跨窗口Platt（仅对照）": fit_calibrator(pooled_scores, pooled_labels, "Platt"),
               "同模型Platt": fit_calibrator(calibration_score, calibration.Class, "Platt"),
               "同模型Isotonic": fit_calibrator(calibration_score, calibration.Class, "Isotonic")}
    calibration_comparison = [{"方案": "未校准", **calibration_metrics(validation.Class, validation_raw)}]
    calibration_comparison.extend({"方案": name, **calibration_metrics(validation.Class, transform(validation_raw))} for name, transform in methods.items())
    # 原跨模型方案仅用于诊断，候选必须与最终冻结基础模型一致。
    eligible = [row for row in calibration_comparison if row["方案"].startswith("同模型")]
    calibration_name = min(eligible, key=lambda row: row["Brier"])["方案"]
    transform = methods[calibration_name]
    validation_score = transform(validation_raw)
    parameters = CostParameters()
    policy = choose_policy(validation.Class, validation_score, amounts=validation.Amount, cost_params=parameters)
    # 此前所有选择完成。下面才读取测试特征预测，并使用测试标签评估。
    raw_test = model.predict_proba(scaler.transform(features(test, mode)))[:, 1]
    test_score = transform(raw_test)
    test_cost = cost_of_actions(test.Class, test_score, policy["复核阈值"], policy["拦截阈值"], amounts=test.Amount, cost_params=parameters)
    baseline_cost = cost_of_actions(test.Class, test_score, np.inf, np.inf, amounts=test.Amount, cost_params=parameters)["模拟成本"]
    test_cost.update({"全部放行模拟成本": baseline_cost, "相对全部放行成本变化": test_cost["模拟成本"] / baseline_cost - 1 if baseline_cost else None})
    metrics = classification_metrics(test.Class, test_score, policy["复核阈值"])
    constant_probability = float(calibration.Class.mean())
    baseline_brier = float(brier_score_loss(test.Class, np.full(len(test), constant_probability)))
    metrics.update({"分类阈值": policy["复核阈值"], "Brier分数": float(brier_score_loss(test.Class, test_score)),
                    "校准前Brier分数": float(brier_score_loss(test.Class, raw_test)), "常数基线Brier": baseline_brier,
                    "常数基线概率": constant_probability,
                    "Brier技能分": float(1 - brier_score_loss(test.Class, test_score) / baseline_brier),
                    "ECE": calibration_metrics(test.Class, test_score)["ECE"]})
    comparison, window_results, ablation = [], [], []
    labels = {"logistic": "类别加权逻辑回归", "xgboost": "类别加权XGBoost"}
    for candidate_kind, key in selected_keys.items():
        aps = [row[0] for row in records[key]]
        comparison.append({"模型": labels[candidate_kind], "金额特征": key[1], "参数": json.dumps(configurations[candidate_kind][key[2]]),
                           "平均AP": float(np.mean(aps)), "AP标准差": float(np.std(aps, ddof=1)), "最近窗口AP": aps[-1]})
        for window, ap in zip(windows, aps):
            window_results.append({"窗口": window["窗口"], "模型": labels[candidate_kind], "AP": ap})
        # 固定该模型选定超参数作金额消融，避免把调参效果算进特征效果。
        for candidate_mode in ("amount", "log", "both"):
            key_mode = (candidate_kind, candidate_mode, key[2])
            for window_index, (window, row) in enumerate(zip(windows, records[key_mode])):
                ablation.append({"模型": labels[candidate_kind], "金额特征": candidate_mode, "窗口": window["窗口"], "AP": row[0],
                                 "相对仅Amount的AP变化": row[0] - records[(candidate_kind, "amount", key[2])][window_index][0]})
    importance = np.abs(model.coef_[0]) if kind == "logistic" else model.feature_importances_
    unique = ~test.duplicated()
    histogram = np.histogram(test_score, bins=np.linspace(0, 1, 11))[0]
    errors = error_analysis(test, test_score, policy["复核阈值"])
    errors["验证与测试对照"] = [
        {"时段": name, "笔数": len(part), "欺诈笔数": int(part.Class.sum()), "欺诈率": float(part.Class.mean()),
         "AP": float(average_precision_score(part.Class, score)), "平均金额": float(part.Amount.mean()),
         "欺诈金额合计": float(part.loc[part.Class == 1, "Amount"].sum())}
        for name, part, score in (("验证", validation, validation_score), ("测试", test, test_score))
    ]
    report = {
        "报告版本": 2, "生成时间": datetime.now(timezone.utc).isoformat(),
        "数据来源": source, "总笔数": len(frame), "欺诈笔数": int(frame.Class.sum()),
        "输入SHA256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest() if data_path else None,
        "时间切分": {"训练集": len(windows[-1]["train"]), "验证集": len(validation), "测试集": len(test)},
        "训练内部切分": dict(zip(["拟合", "早停", "校准"], sizes)),
        "滚动验证窗口": [{"窗口": w["窗口"], "训练集笔数": len(w["train"]), "验证集笔数": len(w["validation"]),
                            "欺诈笔数": int(w["validation"].Class.sum()), "欺诈率基线AP": float(w["validation"].Class.mean())} for w in windows],
        "滚动验证结果": window_results, "模型比较": comparison, "特征消融": ablation,
        "调参记录": trials, "每类模型每种金额表达候选数": tuning_trials,
        "验证集选定模型": winner, "选定参数": configurations[kind][index], "金额特征方案": mode,
        "测试集指标": metrics, "重采样次数": bootstrap_iterations,
        "测试集指标95%区间": stratified_bootstrap_intervals(test.Class, test_score, test.Amount, policy["复核阈值"], policy["拦截阈值"], iterations=bootstrap_iterations, seed=seed),
        "模型差异检验": {"比较窗口": windows[-1]["窗口"], "AP差值定义": "逻辑回归 - XGBoost", "AP差值95%区间": paired, "选择依据": reason},
        "验证集选定策略": policy, "测试集冻结策略表现": test_cost,
        "解释方法": "逻辑回归标准化系数绝对值" if kind == "logistic" else "XGBoost原生特征重要性",
        "特征工程": {"amount": ["保留Amount，移除LogAmount"], "log": ["LogAmount=log1p(交易金额)，移除Amount"], "both": ["Amount与LogAmount"]}[mode],
        "特征重要性": sorted(zip(features(test, mode).columns, importance.tolist()), key=lambda pair: pair[1], reverse=True)[:15],
        "数据质量": {"缺失值总数": summary["缺失值总数"], "重复行数": summary["重复行数"], "重复行中的欺诈数": summary["重复中的欺诈行数"],
                    "测试集完整样本AP": metrics["平均精确率AP"], "测试集去重后AP": float(average_precision_score(test.Class.to_numpy()[unique], test_score[unique]))},
        "SQL数据画像": summary,
        "成本参数": asdict(parameters),
        "成本假设": {"漏放欺诈损失": "max(金额,1) × 金额倍率", "误拦正常成本": parameters.false_block_cost, "人工复核成本": parameters.review_cost,
                    "人工复核召回率": parameters.review_recall, "人工复核误判率": parameters.review_false_positive_rate, "最大复核比例": parameters.review_capacity},
        "校准结果": {"方法": calibration_name, "拟合数据": "冻结基础模型训练段末15%的时间外预测；模型不在校准后重训", "验证集Brier分数": float(brier_score_loss(validation.Class, validation_score)),
                    "选择规则": "仅在同模型Platt和Isotonic中按验证Brier选择；验证标签同时参与校准方案和阈值选择，测试独立"},
        "校准方案比较": calibration_comparison,
        "看板分箱": {"分数分布": [{"风险分数分箱": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "交易笔数": int(count)} for i, count in enumerate(histogram)],
                    "可靠性曲线": reliability_bins(test.Class, test_score)},
        "成本敏感性分析": sensitivity_analysis(validation, validation_score, test, test_score, parameters),
        "误差分析": errors,
        "风险分数说明": "同一冻结模型的历史留出段校准，测试标签仅供离线评估；金额成本为模拟假设。",
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"欺诈标签": test.Class.to_numpy(), "风险分数": test_score}).to_csv(directory / "test_scores.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(directory / "hour_profile.csv", index=False, encoding="utf-8-sig")
    return report
