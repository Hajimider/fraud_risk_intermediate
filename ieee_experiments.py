"""IEEE-CIS 按相对月验证的实验入口；必须显式提供真实或标明为教学的数据。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from experiments import calibration_metrics, error_analysis, fit_calibrator, training_parts
from fraud_model import (
    CostParameters,
    choose_policy,
    classification_metrics,
    cost_of_actions,
    paired_ap_difference_interval,
    stratified_bootstrap_intervals,
)
from ieee_pipeline import (
    feature_screening,
    fit_woe,
    psi,
    temporal_features,
    temporal_features_history,
    temporal_features_target_encoding,
    transform_woe,
    uid_diagnostics,
)


def choose_ieee_policy(labels, scores, amounts, parameters):
    """在极小验证月无法形成三段非空策略时，使用容量内的秩边界兜底。"""
    try:
        return choose_policy(labels, scores, amounts=amounts, cost_params=parameters)
    except ValueError as exc:
        score = np.asarray(scores, dtype=float)
        order = np.argsort(score, kind="stable")
        n = len(score)
        review_count = max(1, min(int(np.floor(n * parameters.review_capacity)), n - 2))
        review_start = n - review_count - 1
        block_start = n - 1
        review_threshold = float(score[order[review_start]])
        block_threshold = float(score[order[block_start]])
        if review_threshold >= block_threshold:
            block_threshold = float(np.nextafter(review_threshold, np.inf))
        result = cost_of_actions(labels, score, review_threshold, block_threshold,
                                 amounts=amounts, cost_params=parameters)
        result.update({"复核阈值": review_threshold, "拦截阈值": block_threshold,
                       "策略回退原因": str(exc)})
        return result


def ieee_sensitivity_analysis(validation, validation_score, test, test_score, parameters):
    rows = []
    for multiplier in (.5, 1.0, 2.0):
        for false_block_cost in (3.0, 6.0, 12.0):
            changed = parameters.__class__(**{**parameters.__dict__, "fraud_loss_multiplier": multiplier,
                                               "false_block_cost": false_block_cost})
            policy = choose_ieee_policy(validation.Class, validation_score, validation.Amount, changed)
            evaluated = cost_of_actions(test.Class, test_score, policy["复核阈值"], policy["拦截阈值"],
                                        amounts=test.Amount, cost_params=changed)
            rows.append({"欺诈损失金额倍率": multiplier, "误拦成本": false_block_cost,
                         "复核阈值": policy["复核阈值"], "拦截阈值": policy["拦截阈值"],
                         "验证成本": policy["模拟成本"], "测试复核率": evaluated["复核率"],
                         "测试成本": evaluated["模拟成本"]})
    return rows


def monthly_windows(frame, label_delay=7 * 86400):
    # 原始数据无经过核验的日历起点，按相对30天分块，不能称为自然月。
    t0 = frame.TransactionDT.min()
    month = ((frame.TransactionDT - t0) // (30 * 86400)).astype(int)
    values = sorted(month.unique())
    last_span = frame.loc[month == values[-1], "TransactionDT"].max() - frame.loc[month == values[-1], "TransactionDT"].min()
    if last_span < 15 * 86400 and len(values) > 4:
        month = month.replace(values[-1], values[-2])
        values = sorted(month.unique())
    if len(values) < 4:
        raise ValueError("按月实验至少需要4个相对30天分块；最后一块只用于测试")
    windows = []
    test_origin = frame.loc[month == values[-1], "TransactionDT"].min()
    for value in values[2:-1]:
        train, validation = frame.loc[month < value], frame.loc[month == value]
        prediction_origin = validation.TransactionDT.min()
        # 模型、早停与校准目标也必须已经成熟，不能只约束目标编码。
        train = train.loc[train.TransactionDT <= prediction_origin - label_delay]
        # 所有选型、校准方案和阈值选择都必须在测试开始前取得标签。
        validation = validation.loc[validation.TransactionDT <= test_origin - label_delay]
        if train.isFraud.nunique() != 2 or validation.isFraud.nunique() != 2:
            raise ValueError("每个训练/验证月均需两类标签")
        windows.append((value, train, validation))
    test = frame.loc[month == values[-1]]
    if test.isFraud.nunique() != 2:
        raise ValueError("测试月必须包含两类标签")
    return windows, test


LOW_CARDINALITY_CATS = ["ProductCD", "card4", "card6", "DeviceType", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def encode_categorical(frame, freq_maps=None):
    encoded = {}
    for column in LOW_CARDINALITY_CATS:
        if column not in frame:
            continue
        series = frame[column].astype("string")
        mapping = freq_maps.get(column, {}) if freq_maps is not None else series.value_counts(normalize=True).to_dict()
        encoded[column] = series.map(mapping).fillna(0.0).astype(float)
    return pd.DataFrame(encoded, index=frame.index) if encoded else pd.DataFrame(index=frame.index)


def categorical_frequency_maps(frame):
    return {column: frame[column].astype("string").value_counts(normalize=True).to_dict()
            for column in LOW_CARDINALITY_CATS if column in frame}


def dataset_features(
    frame, cutoff, history_features=None, freq_maps=None, target_features_cache=None,
    selected_columns=None, row_index=None, context_frame=None,
):
    rows = frame if row_index is None else frame.loc[row_index]
    context = frame if context_frame is None else context_frame
    excluded = ["isFraud", "TransactionID", "TransactionDT", "card1", "addr1"]
    if history_features is not None:
        excluded.extend(history_features.columns)
    numeric = rows.select_dtypes(include=["number"]).drop(columns=excluded, errors="ignore")
    categorical = encode_categorical(rows, freq_maps)
    numeric = pd.concat([numeric, categorical], axis=1).add_prefix("原始_").replace([np.inf, -np.inf], np.nan)
    if selected_columns is not None:
        raw_columns = [column for column in selected_columns if column.startswith("原始_") and column in numeric]
        numeric = numeric[raw_columns]
    if history_features is None:
        temporal = temporal_features(rows, cutoff)
    else:
        if target_features_cache is not None and cutoff in target_features_cache:
            target = target_features_cache[cutoff].loc[rows.index]
        else:
            target = temporal_features_target_encoding(context, history_features, cutoff).loc[rows.index]
            if target_features_cache is not None:
                target_features_cache[cutoff] = temporal_features_target_encoding(context, history_features, cutoff)
        temporal = target
    if selected_columns is not None:
        temporal = temporal[[column for column in selected_columns if column in temporal]]
    return pd.concat([numeric, temporal], axis=1)


def fit_model(kind, config, x_train, y_train, x_stop, y_stop, trees, seed):
    if kind == "WOE评分卡":
        mappings = {column: fit_woe(x_train[column], y_train) for column in x_train}
        woe_train = pd.DataFrame({column: transform_woe(x_train[column], fitted) for column, fitted in mappings.items()}, index=x_train.index)
        iv_values = {column: mappings[column]["IV"] for column in mappings}
        corr = woe_train.corr().abs().fillna(0.0)
        ordered = sorted(iv_values, key=lambda column: (-iv_values[column], column))
        retained = []
        for column in ordered:
            if all(corr.loc[column, previous] <= 0.85 for previous in retained):
                retained.append(column)
        kept = [column for column in x_train if column in retained]

        def convert(value):
            return pd.DataFrame({column: transform_woe(value[column], mappings[column]) for column in kept}, index=value.index)

        model = LogisticRegression(C=config["C"], max_iter=1500, class_weight="balanced", random_state=seed)
        model.fit(convert(x_train), y_train)
    else:
        medians = x_train.median().fillna(0)

        def convert(value):
            return value.fillna(medians).astype(float)
        weight = float((y_train == 0).sum() / (y_train == 1).sum())
        if kind == "LightGBM":
            model = LGBMClassifier(n_estimators=trees, num_leaves=config["num_leaves"], learning_rate=.05,
                                   scale_pos_weight=weight, n_jobs=2, random_state=seed, verbosity=-1)
            model.fit(convert(x_train), y_train, eval_set=[(convert(x_stop), y_stop)], eval_metric="average_precision",
                      callbacks=[early_stopping(max(5, min(20, trees // 4)), verbose=False)])
        else:
            model = XGBClassifier(n_estimators=trees, max_depth=config["max_depth"], learning_rate=.05,
                                  scale_pos_weight=weight, tree_method="hist", n_jobs=2, random_state=seed,
                                  eval_metric="aucpr", early_stopping_rounds=max(5, min(20, trees // 4)))
            model.fit(convert(x_train), y_train, eval_set=[(convert(x_stop), y_stop)], verbose=False)
    return model, convert


def explain_cases(model, convert, train, test, labels, score, threshold, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Arial Unicode MS") if name in available]
    if chinese_fonts:
        plt.rcParams["font.sans-serif"] = [*chinese_fonts, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    background = convert(train.iloc[:min(128, len(train))])
    values = convert(test.iloc[:min(256, len(test))])
    try:
        try:
            explainer = shap.Explainer(model, background, feature_names=list(train.columns))
        except (TypeError, ValueError):
            if not hasattr(model, "get_booster"):
                raise
            explainer = shap.TreeExplainer(model, data=background, feature_names=list(train.columns))
    except Exception:
        # SHAP/XGBoost binary compatibility varies by installed versions; keep a truthful
        # model-importance artifact instead of claiming SHAP values were computed.
        importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(train.columns))))
        order = np.argsort(importance)[-15:]
        plt.figure(figsize=(8, 5))
        plt.barh(np.asarray(train.columns)[order], importance[order], color="#0f766e")
        plt.title("模型特征重要性（SHAP不可用时的兼容回退）")
        plt.tight_layout()
        plt.savefig(directory / "shap_global.png", bbox_inches="tight")
        plt.close("all")
        return ["shap_global.png"]
    explanations = explainer(values)
    if explanations.values.ndim == 3:
        explanations = explanations[:, :, 1]
    shap.plots.beeswarm(explanations, max_display=15, show=False)
    plt.savefig(directory / "shap_global.png", bbox_inches="tight")
    plt.close("all")
    outputs = ["shap_global.png"]
    for name, mask in (("真阳性", (labels == 1) & (score >= threshold)), ("漏判", (labels == 1) & (score < threshold)), ("误报", (labels == 0) & (score >= threshold))):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        explanation = explainer(convert(test.iloc[[indices[0]]]))
        if explanation.values.ndim == 3:
            explanation = explanation[:, :, 1]
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        filename = f"shap_{name}.png"
        plt.savefig(directory / filename, bbox_inches="tight")
        plt.close("all")
        outputs.append(filename)
    return outputs


def run_ieee(data_path, output_dir="reports/ieee", trees=160, trials=3, iterations=1000, seed=42, export_shap=True):
    if trials not in (1, 2, 3) or trees < 1 or iterations < 1:
        raise ValueError("候选预算为1–3，树数与重采样次数必须为正")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError("缺少IEEE parquet，请先运行 python run.py ieee-prepare")
    frame = pd.read_parquet(path).sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    windows, test = monthly_windows(frame)
    configs = {"WOE评分卡": [{"C": c} for c in (.1, 1., 10.)][:trials],
               "LightGBM": [{"num_leaves": n} for n in (7, 15, 31)][:trials],
               "XGBoost": [{"max_depth": n} for n in (3, 4, 5)][:trials]}
    fold_data, screens, trial_rows, fitted_models, validation_scores = [], [], [], {}, {}
    history_names = ("历史频次", "历史金额均值", "历史金额标准差", "新颖性", "距上次秒数", "历史频率编码",
                     "1小时频次", "24小时频次", "24小时金额", "金额偏离度")
    history_columns = [f"{name}_{suffix}" for name in ("卡", "地址", "邮箱", "设备", "近似用户") for suffix in history_names]
    history_columns += ["相对日内小时", "相对周内日"]
    history_features = frame[history_columns] if set(history_columns) <= set(frame.columns) else temporal_features_history(frame)
    for month_index, (month, train, validation) in enumerate(windows):
        fit, stop, calibration = training_parts(train.rename(columns={"TransactionDT": "Time", "isFraud": "Class"}))
        cutoff = float(fit.Time.max())
        # 所有月的标签编码只用该月基础模型拟合截止前的已成熟标签。
        freq_maps = categorical_frequency_maps(frame.loc[fit.index])
        needed_index = fit.index.union(stop.index).union(validation.index)
        if month_index == len(windows) - 1:
            needed_index = needed_index.union(calibration.index)
        context = frame.loc[frame.TransactionDT <= validation.TransactionDT.max()]
        x = dataset_features(frame.loc[needed_index], cutoff, history_features=history_features, freq_maps=freq_maps,
                              context_frame=context)
        screen, _ = feature_screening(x.loc[fit.index], fit.Class, x.loc[stop.index])
        screen["验证相对月"] = month
        screens.extend(screen.to_dict("records"))
        columns = screen.loc[screen.筛选结果 == "保留", "特征"].tolist()
        if not columns:
            raise ValueError("筛选后无可用特征；请检查数据量、IV和PSI，不应静默跳过筛选")
        fold_data.append((month, fit, stop, calibration, validation, x[columns].copy(), columns, cutoff))
        for kind, candidates in configs.items():
            for index, config in enumerate(candidates):
                model, convert = fit_model(kind, config, x.loc[fit.index, columns], fit.Class, x.loc[stop.index, columns], stop.Class, trees, seed)
                score = model.predict_proba(convert(x.loc[validation.index, columns]))[:, 1]
                trial_rows.append({"相对月": month, "模型": kind, "候选": index, "参数": config,
                                   "AP": float(average_precision_score(validation.isFraud, score))})
                validation_scores[(month, kind, index)] = score
                if month_index == len(windows) - 1:
                    fitted_models[(month, kind, index)] = (model, convert, score)
        del x, context
    table = pd.DataFrame(trial_rows)
    means = table.groupby(["模型", "候选"]).AP.mean()
    kind, index = means.idxmax()
    index = int(index)
    last_month, fit, stop, calibration, validation, x, columns, cutoff = fold_data[-1]
    model, convert, validation_raw = fitted_models[(last_month, kind, index)]
    calibration_raw = model.predict_proba(convert(x.loc[calibration.index, columns]))[:, 1]
    transforms = {name: fit_calibrator(calibration_raw, calibration.Class, name) for name in ("Platt", "Isotonic")}
    calibration_comparison = [{"方案": name, **calibration_metrics(validation.isFraud, transform(validation_raw))} for name, transform in transforms.items()]
    chosen_calibration = min(calibration_comparison, key=lambda row: row["Brier"])["方案"]
    transform = transforms[chosen_calibration]
    validation_score = transform(validation_raw)
    parameters = CostParameters()
    policy = choose_ieee_policy(validation.isFraud, validation_score, validation.TransactionAmt, parameters)
    ablations = []
    # 分组消融仅在验证月进行；不使用测试月增量挑选特征。
    for month, fold_fit, fold_stop, _, fold_validation, fold_x, fold_columns, _ in fold_data:
        current = [column for column in fold_columns if column.startswith("原始_")]
        previous = None
        for name, tokens in (("基础字段", []), ("加频次", ["频次", "24小时金额"]), ("加偏离度", ["金额均值", "金额标准差", "偏离度"]),
                             ("加新颖性", ["新颖性", "距上次"]), ("加编码", ["编码", "相对日内", "相对周内"])):
            current += [column for column in fold_columns if any(token in column for token in tokens) and column not in current]
            usable = [column for column in current if fold_x.loc[fold_fit.index, column].nunique() > 1]
            ablation_model, ablation_convert = fit_model(kind, configs[kind][index], fold_x.loc[fold_fit.index, usable], fold_fit.Class,
                                                         fold_x.loc[fold_stop.index, usable], fold_stop.Class, trees, seed)
            score = ablation_model.predict_proba(ablation_convert(fold_x.loc[fold_validation.index, usable]))[:, 1]
            ap = float(average_precision_score(fold_validation.isFraud, score))
            row = {"相对月": month, "特征组": name, "特征数": len(usable), "AP": ap}
            if previous is not None:
                row["增量95%区间"] = paired_ap_difference_interval(fold_validation.isFraud, score, previous, iterations=iterations, seed=seed)
                row["AP增量"] = ap - float(average_precision_score(fold_validation.isFraud, previous))
            ablations.append(row)
            previous = score
    # 模型、校准方案、阈值、特征列已全部冻结，才打开测试月进行最终评估。
    explain_index = fit.index.union(test.index)
    full_x = dataset_features(frame.loc[explain_index], cutoff, history_features=history_features,
                              freq_maps=categorical_frequency_maps(frame.loc[fit.index]),
                              context_frame=frame)
    raw_test = model.predict_proba(convert(full_x.loc[test.index, columns]))[:, 1]
    test_score = transform(raw_test)
    policy_test = cost_of_actions(test.isFraud, test_score, policy["复核阈值"], policy["拦截阈值"], amounts=test.TransactionAmt, cost_params=parameters)
    allow = cost_of_actions(test.isFraud, test_score, np.inf, np.inf, amounts=test.TransactionAmt, cost_params=parameters)["模拟成本"]
    policy_test.update({"全部放行模拟成本": allow, "相对全部放行成本变化": policy_test["模拟成本"] / allow - 1 if allow else None})
    # 预先约定的训练一次/按月重训对照，只做诊断，不用该表选择最终模型。
    first_month, first_fit, _, _, _, _, first_columns, first_cutoff = fold_data[0]
    first_x = fold_data[0][5]
    first_stop = fold_data[0][2]
    first_model, first_convert = fit_model(
        kind, configs[kind][index], first_x.loc[first_fit.index, first_columns], first_fit.Class,
        first_x.loc[first_stop.index, first_columns], first_stop.Class, trees, seed,
    )
    drift_index = first_fit.index.union(test.index)
    for _, _, _, _, part, _, _, _ in fold_data:
        drift_index = drift_index.union(part.index)
    frozen_x = dataset_features(frame.loc[drift_index], first_cutoff, history_features=history_features,
                                 freq_maps=categorical_frequency_maps(frame.loc[first_fit.index]),
                                 context_frame=frame)
    monthly = []
    for month, _, _, _, part, _fold_x, _, _ in fold_data:
        retrained_score = validation_scores[(month, kind, index)]
        frozen_score = first_model.predict_proba(first_convert(frozen_x.loc[part.index, first_columns]))[:, 1]
        monthly.append({"相对月": month, "按月重训AP": float(average_precision_score(part.isFraud, retrained_score)),
                        "训练一次AP": float(average_precision_score(part.isFraud, frozen_score)),
                        "特征PSI": {column: psi(frozen_x.loc[first_fit.index, column], frozen_x.loc[part.index, column]) for column in first_columns},
                        "分数PSI": psi(first_model.predict_proba(first_convert(frozen_x.loc[first_fit.index, first_columns]))[:, 1], frozen_score)})
    frozen_test = first_model.predict_proba(first_convert(frozen_x.loc[test.index, first_columns]))[:, 1]
    t0 = frame.TransactionDT.min()
    test_month = int((test.TransactionDT.iloc[0] - t0) // (30 * 86400))
    monthly.append({"相对月": test_month, "用途": "独立测试，预先约定的诊断对照",
                    "按月重训AP": float(average_precision_score(test.isFraud, raw_test)),
                    "训练一次AP": float(average_precision_score(test.isFraud, frozen_test)),
                    "特征PSI": {column: psi(frozen_x.loc[first_fit.index, column], frozen_x.loc[test.index, column]) for column in first_columns},
                    "分数PSI": psi(first_model.predict_proba(first_convert(frozen_x.loc[first_fit.index, first_columns]))[:, 1], frozen_test)})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shap_files = explain_cases(model, convert, full_x.loc[fit.index, columns], full_x.loc[test.index, columns], test.isFraud.to_numpy(), test_score, policy["复核阈值"], directory / "images") if export_shap else []
    source_path = path.parent / ("ieee.source.json" if ".sample-" not in path.name else path.name.replace("transactions", "ieee").replace(".parquet", ".source.json"))
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {"说明": "来源记录未提供，不能据此宣称真实IEEE实验"}
    baseline_brier = calibration_metrics(test.isFraud, np.full(len(test), float(calibration.Class.mean())))["Brier"]
    report = {"数据来源": "IEEE-CIS格式本地输入", "来源记录": source, "总笔数": len(frame), "欺诈笔数": int(frame.isFraud.sum()),
              "时间口径": "相对30天分块，非自然月；最后一块只测试", "测试相对月": test_month,
              "实体诊断": uid_diagnostics(frame), "选定模型": kind, "选定参数": configs[kind][index],
              "选定特征": columns, "标签截止秒": cutoff, "标签延迟秒": 7 * 86400,
              "标签可用规则": "每月拟合/早停/校准标签截至预测起点前7天；验证选择标签截至测试起点前7天，间隔内交易可提供无标签历史但不用于监督选择",
              "调参记录": trial_rows, "特征筛选": screens, "分组消融": ablations,
              "校准方案比较": calibration_comparison, "选定校准": chosen_calibration,
              "测试集指标": {**classification_metrics(test.isFraud, test_score, policy["复核阈值"]), **calibration_metrics(test.isFraud, test_score),
                              "常数基线Brier": baseline_brier, "Brier技能分": 1 - calibration_metrics(test.isFraud, test_score)["Brier"] / baseline_brier},
              "测试集指标95%区间": stratified_bootstrap_intervals(test.isFraud, test_score, test.TransactionAmt, policy["复核阈值"], policy["拦截阈值"], iterations, seed, cost_params=parameters),
              "验证集策略": policy, "测试集策略": policy_test, "月度漂移对照": monthly,
              "成本敏感性分析": ieee_sensitivity_analysis(validation.rename(columns={"isFraud": "Class", "TransactionAmt": "Amount"}), validation_score,
                                                         test.rename(columns={"isFraud": "Class", "TransactionAmt": "Amount"}), test_score, parameters),
              "误差分析": error_analysis(test.rename(columns={"isFraud": "Class", "TransactionAmt": "Amount", "TransactionDT": "Time"}), test_score, policy["复核阈值"]),
              "SHAP文件": shap_files, "说明": "业务收益为固定假设模拟；未自动引用比赛排行榜或推断真实日历起点"}
    (directory / "ieee_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    pd.DataFrame(screens).to_csv(directory / "feature_screening.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablations).to_csv(directory / "group_ablation.csv", index=False, encoding="utf-8-sig")
    return report
