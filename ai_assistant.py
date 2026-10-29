"""读取训练报告，生成离线摘要或调用 AI 解释结果。

AI 只解释已有报告，不参与模型训练、阈值搜索或风险决策。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import request

DEFAULT_REPORT = Path(__file__).resolve().parent / "reports" / "latest.json"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
MAX_QUESTION_LENGTH = 500
DEFAULT_TIMEOUT_SECONDS = 90.0
AI_CONFIG_KEYS = ("AI_BASE_URL", "AI_MODEL", "AI_API_KEY", "AI_TIMEOUT_SECONDS")


def load_report(path: str | Path = DEFAULT_REPORT) -> dict[str, Any]:
    """读取报告，拒绝非对象格式的 JSON。"""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"找不到训练报告：{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("训练报告必须是 JSON 对象")
    return report


def load_dotenv(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """将 .env 中的 KEY=VALUE 解析为字典，不修改环境变量。"""
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_ai_config(
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """优先读取 Secrets，其次环境变量，最后读取本地 .env。"""
    values = {**load_dotenv(), **os.environ} if environ is None else dict(environ)
    if secrets is not None:
        values.update({key: secrets[key] for key in AI_CONFIG_KEYS if key in secrets})
    config = {key: str(values.get(key, "")).strip() for key in AI_CONFIG_KEYS}
    config["AI_BASE_URL"] = config["AI_BASE_URL"].rstrip("/")
    if (
        config["AI_BASE_URL"].startswith("https://your-compatible-endpoint")
        or config["AI_MODEL"] == "your-model-name"
    ):
        for key in ("AI_BASE_URL", "AI_MODEL", "AI_API_KEY"):
            config[key] = ""
    return config


def build_context(report: Mapping[str, Any], top_features: int = 15) -> dict[str, Any]:
    """只选取报告汇总字段，不传原始交易或逐行标签。"""
    metrics = report.get("测试集指标", {})
    strategy = report.get("测试集冻结策略表现", {})
    selected = report.get("验证集选定策略", {})
    return {
        "数据来源": report.get("数据来源"),
        "样本规模": {
            "总笔数": report.get("总笔数"),
            "欺诈笔数": report.get("欺诈笔数"),
        },
        "时间切分": report.get("时间切分", {}),
        "模型比较": report.get("模型比较", []),
        "验证集选定模型": report.get("验证集选定模型"),
        "测试集指标": {
            key: metrics.get(key)
            for key in (
                "平均精确率AP", "ROC-AUC", "精确率", "召回率", "F1",
                "真阴性", "假阳性", "假阴性", "真阳性",
            )
            if key in metrics
        },
        "验证集选定策略": {
            key: selected.get(key)
            for key in (
                "复核阈值", "拦截阈值", "复核笔数", "拦截笔数",
                "复核率", "是否超出复核容量", "模拟成本",
            )
            if key in selected
        },
        "测试集冻结策略表现": {
            key: strategy.get(key)
            for key in (
                "放行笔数", "复核笔数", "拦截笔数", "预计漏放欺诈", "预计误拦正常",
                "复核率", "是否超出复核容量", "模拟成本",
                "全部放行模拟成本", "相对全部放行成本变化",
            )
            if key in strategy
        },
        "成本假设": report.get("成本假设", {}),
        "特征工程": report.get("特征工程", []),
        "数据质量": report.get("数据质量", {}),
        "特征重要性": [
            {
                "特征": item[0],
                "重要性": round(float(item[1]), 6) if item[1] is not None else None,
            }
            for item in report.get("特征重要性", [])[:top_features]
            if isinstance(item, (list, tuple)) and len(item) == 2
        ],
        "解释方法": report.get("解释方法"),
        "风险分数说明": report.get("风险分数说明"),
    }


def build_messages(question: str, context: Mapping[str, Any]) -> list[dict[str, str]]:
    """构造带事实约束的 Chat Completions 消息。"""
    clean_question = " ".join(str(question).split())[:MAX_QUESTION_LENGTH]
    context_text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    system = (
        "你是数据挖掘项目中的风控分析助手。只能依据用户提供的结构化报告回答，"
        "不能编造报告没有的数字。用中文直接回答问题，引用相关数值并解释其含义，"
        "按内容需要分段或列要点，不必套用固定的章节格式。区分模型指标、成本假设和业务建议。"
        "V1-V28 是匿名 PCA 特征，不能解释成设备、商户或地域。"
        "风险分数未校准时称为风险排序分数，不称为欺诈概率。不要执行训练、改阈值、导出原始交易或调用其他工具。"
        "如需 Markdown 标题，只使用三级标题（###），避免大标题影响页面排版。"
        "不要在正文中另列‘使用的报告字段’，页面会单独展示报告依据。"
    )
    user = f"报告上下文：{context_text}\n\n用户问题：{clean_question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _trace(context: Mapping[str, Any], sources: list[str]) -> str:
    metrics = context["测试集指标"]
    model = context["验证集选定模型"]
    return (
        "\n\n报告记录："
        f"选定模型为 {model}，测试集 AP 为 {float(metrics['平均精确率AP']):.4f}，"
        f"ROC-AUC 为 {float(metrics['ROC-AUC']):.4f}。"
        f"参考字段：{'、'.join(sources)}。"
    )


def _question_topic(question: str) -> tuple[str, list[str]]:
    """按关键词确定主题和引用字段，离线摘要与 AI 回答使用同一套规则。"""
    text = " ".join(str(question).lower().split())
    if any(word in text for word in ("模型", "xgboost", "逻辑回归", "比较")):
        return "模型", ["模型比较", "验证集选定模型", "时间切分"]
    if any(word in text for word in ("策略", "阈值", "复核", "拦截", "容量")):
        return "策略", ["验证集选定策略", "测试集冻结策略表现", "成本假设"]
    if any(word in text for word in ("成本", "收益", "损失")):
        return "成本", ["测试集冻结策略表现", "成本假设"]
    if any(word in text for word in ("特征", "解释", "v14", "pca", "匿名")):
        return "特征", ["特征重要性", "解释方法", "风险分数说明"]
    if any(word in text for word in ("指标", "效果", "ap", "auc", "召回", "精确", "f1")):
        return "指标", ["测试集指标", "样本规模"]
    return "概览", ["数据来源", "样本规模", "验证集选定模型", "测试集指标"]


def offline_answer(question: str, report: Mapping[str, Any]) -> dict[str, Any]:
    """按问题主题整理报告摘要，无需调用模型服务。"""
    context = build_context(report)
    topic, sources = _question_topic(question)
    metrics = context["测试集指标"]
    strategy = context["测试集冻结策略表现"]
    if topic == "模型":
        comparison = context["模型比较"]
        answer = (
            f"验证集上，{comparison[0]['模型']} 的 AP 为 {comparison[0]['验证集AP']:.4f}，"
            f"{comparison[1]['模型']} 为 {comparison[1]['验证集AP']:.4f}，因此选定 {context['验证集选定模型']}。"
            "\n\n模型和策略阈值都在验证集上确定，测试集来自后续时间段。当前比较基于一次时间切分，还需更多时间窗口检验。"
        )
    elif topic == "策略":
        capacity = context["成本假设"]["最大复核比例"]
        answer = (
            f"测试集放行{strategy['放行笔数']:,}笔、复核{strategy['复核笔数']:,}笔、"
            f"拦截{strategy['拦截笔数']:,}笔；复核率为{float(strategy['复核率']):.2%}，容量上限为{capacity:.0%}，"
            f"状态为{'已超出' if strategy['是否超出复核容量'] else '未超出'}。"
            "\n\n阈值在验证集确定后冻结。后续时段的分数分布可能变化，不保证三类动作都会出现，也不保证始终满足容量上限。"
        )
    elif topic == "成本":
        change = strategy.get("相对全部放行成本变化")
        answer = (
            f"冻结策略模拟成本为{strategy['模拟成本']:,.2f}，全部放行为"
            f"{strategy['全部放行模拟成本']:,.2f}，相对变化为{change:.2%}。"
            "\n\n这项变化是按设定费用和复核效果计算的，比较对象为全部放行，不能据此推算线上收益。"
        )
    elif topic == "特征":
        top = context["特征重要性"][:3]
        names = "、".join(item["特征"] for item in top)
        answer = (
            f"当前模型重要性较高的特征包括 {names}。"
            "\n\nV1-V28 是 PCA 转换后的匿名特征，无法还原成设备、商户或地域。重要性反映模型对变量的使用程度，不说明欺诈原因。"
        )
    elif topic == "指标":
        answer = (
            f"测试集 AP={metrics['平均精确率AP']:.4f}、ROC-AUC={metrics['ROC-AUC']:.4f}、"
            f"精确率={metrics['精确率']:.4f}、召回率={metrics['召回率']:.4f}、F1={metrics['F1']:.4f}。"
            f"混淆矩阵为 TN={metrics['真阴性']}、FP={metrics['假阳性']}、FN={metrics['假阴性']}、TP={metrics['真阳性']}。"
            "\n\n欺诈样本很少，准确率容易掩盖漏报。AP 用于观察排序质量，精确率和召回率则反映当前分类阈值下的误报与漏报。风险分数未经概率校准。"
        )
    else:
        answer = (
            f"项目使用{context['数据来源']}，共{context['样本规模']['总笔数']:,}笔交易，"
            f"欺诈{context['样本规模']['欺诈笔数']:,}笔；验证集选定{context['验证集选定模型']}，"
            f"测试集 AP 为{metrics['平均精确率AP']:.4f}。"
            "\n\n可以继续查看模型比较、测试集指标、复核策略、模拟成本或特征重要性。"
        )
    return {"mode": "offline", "answer": answer + _trace(context, sources), "sources": sources}


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_timeout(environ: Mapping[str, str]) -> float:
    """读取请求超时，并限制在 10～300 秒。"""
    raw = str(environ.get("AI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return min(max(value, 10.0), 300.0)


def _message_content(response: Mapping[str, Any]) -> str:
    if not isinstance(response, Mapping):
        raise ValueError("模型响应必须是对象")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("模型响应缺少 choices")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise ValueError("模型响应缺少可读文本")


def normalize_answer(text: str) -> str:
    """去掉 AI 正文重复的报告字段段落，来源由页面统一呈现。"""
    lines = str(text).splitlines()
    cleaned: list[str] = []
    skip_field_list = False
    field_heading = re.compile(
        r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*)?使用的报告字段(?:\*\*)?\s*[:：]?(.*)$"
    )
    any_heading = re.compile(r"^\s{0,3}#{1,6}\s+")
    for line in lines:
        match = field_heading.match(line)
        if match:
            skip_field_list = not bool(match.group(1).strip())
            continue
        if skip_field_list:
            if any_heading.match(line):
                skip_field_list = False
            else:
                continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def answer_question(
    question: str,
    report: Mapping[str, Any] | None = None,
    report_path: str | Path = DEFAULT_REPORT,
    environ: Mapping[str, str] | None = None,
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """回答问题；没有完整 AI 配置或调用失败时自动回退离线摘要。"""
    loaded_report = dict(report) if report is not None else load_report(report_path)
    context = build_context(loaded_report)
    env = load_ai_config(environ=environ)
    base_url = env["AI_BASE_URL"]
    model = env["AI_MODEL"]
    api_key = env["AI_API_KEY"]
    if not base_url or not model:
        return offline_answer(question, loaded_report)
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": build_messages(question, context),
        "temperature": 0.1,
    }
    send = transport or _post_json
    try:
        response = send(url, payload, headers, request_timeout(env))
        answer = normalize_answer(_message_content(response))
        _, sources = _question_topic(question)
        sources = ["报告上下文", *sources]
        return {"mode": "ai", "answer": answer + _trace(context, sources), "sources": sources}
    except Exception as exc:
        fallback = offline_answer(question, loaded_report)
        fallback["mode"] = "offline-fallback"
        fallback["error"] = f"AI 服务调用失败，已使用离线摘要：{type(exc).__name__}"
        return fallback

