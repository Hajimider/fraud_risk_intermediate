"""交易反欺诈数据挖掘与风险决策工作台。"""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ai_assistant import answer_question, load_ai_config, request_timeout

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "latest.json"

st.set_page_config(
    page_title="反欺诈风险工作台",
    page_icon=":material/shield_lock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy:#102a43; --teal:#0f766e; --teal-soft:#d9f3ef; --amber:#b45309; --amber-soft:#fff3dc; --red:#b91c1c; --red-soft:#fee8e7; --ink:#172033; --muted:#64748b; --line:#d9e2ec; --canvas:#f5f8fb; }
    .stApp { background:var(--canvas); color:var(--ink); }
    /* 保留侧栏展开按钮，去掉顶部栏背景和工具区。 */
    [data-testid="stHeader"] { min-height:0; height:0 !important; background:transparent; pointer-events:none; }
    [data-testid="stToolbar"]:not(:has([data-testid="stExpandSidebarButton"])),
    [data-testid="stToolbarActions"] { display:none; }
    [data-testid="stExpandSidebarButton"] { position:fixed; top:12px; left:16px; width:32px; height:32px; pointer-events:auto; background:#e7eff7; color:var(--navy); }
    [data-testid="stSidebar"] { background:var(--navy); border-right:0; }
    [data-testid="stSidebar"] * { color:#e7eff7; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.45rem .6rem; border-radius:5px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.09); }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:.2rem; }
    .block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:3rem; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:6px; padding:1rem 1.1rem; box-shadow:0 2px 8px rgba(16,42,67,.04); }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:.78rem; }
    [data-testid="stMetricValue"] { color:var(--navy); font-variant-numeric:tabular-nums; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:6px; background:#fff; box-sizing:border-box; max-width:100%; min-width:0; overflow:hidden; }
    [data-testid="stHorizontalBlock"], [data-testid="column"] { min-width:0 !important; max-width:100% !important; }
    [data-testid="column"] { overflow:hidden; }
    [data-testid="stMarkdownContainer"] { min-width:0; max-width:100%; }
    .brand { padding:.6rem .25rem 1.6rem; }
    .brand-mark { width:54px; height:36px; display:inline-flex; align-items:center; justify-content:center; background:#e7eff7; color:var(--navy); border-radius:6px; font-weight:800; margin-bottom:.75rem; font-size:.84rem; }
    .brand-title { font-size:1.05rem; font-weight:700; line-height:1.35; }
    .brand-subtitle { color:#a9bed0; font-size:.74rem; margin-top:.25rem; line-height:1.45; }
    .eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.35rem; }
    .page-title { color:var(--navy); font-size:2rem; font-weight:750; letter-spacing:-.02em; margin:0; }
    .page-subtitle { color:var(--muted); margin:.35rem 0 1.4rem; font-size:.92rem; }
    .status-bar { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:.1rem 0 1.2rem; }
    .status { display:inline-flex; align-items:center; gap:.35rem; border-radius:99px; padding:.28rem .62rem; font-size:.76rem; font-weight:650; border:1px solid transparent; }
    .status-blue { color:var(--navy); background:#e5eef7; border-color:#c7d9e8; }
    .status-green { color:#146356; background:var(--teal-soft); border-color:#b6e3dc; }
    .status-amber { color:#8a4306; background:var(--amber-soft); border-color:#f2d39c; }
    .status-red { color:#8e1f1f; background:var(--red-soft); border-color:#f5b8b5; }
    .callout { border-left:4px solid var(--teal); background:#eaf7f5; padding:.8rem 1rem; border-radius:0 6px 6px 0; color:#164e4a; margin:.35rem 0 1.1rem; }
    .callout-warning { border-left-color:var(--amber); background:var(--amber-soft); color:#713b0b; }
    .section-kicker { color:var(--muted); font-size:.76rem; font-weight:700; letter-spacing:.04em; margin:.25rem 0 .4rem; }
    .finding { font-size:1.1rem; color:var(--navy); font-weight:700; line-height:1.45; margin:0 0 .35rem; }
    .small-note { color:var(--muted); font-size:.78rem; line-height:1.5; }
    .st-key-experiment-summary [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
    .summary-content { display:flex; flex-direction:column; justify-content:center; gap:.55rem; padding:.2rem 0; }
    .summary-content > * { margin:0 !important; }
    .table-title { color:var(--navy); font-weight:700; font-size:1rem; margin:.2rem 0 .55rem; }
    footer { visibility:hidden; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea { background:#fff !important; color:var(--ink) !important; border-color:var(--line) !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stTextArea"] textarea::placeholder { color:var(--ink) !important; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#fff; }
    [data-testid="stVegaLiteChart"] { width:100% !important; max-width:100% !important; overflow:hidden; }
    [data-testid="stVegaLiteChart"] > div, [data-testid="stVegaLiteChart"] svg { max-width:100% !important; }
    [data-testid="stMarkdownContainer"]:has(.data-table-wrap) { margin-bottom:0 !important; }
    .data-table-wrap { width:100%; max-width:100%; min-width:0; box-sizing:border-box; overflow-x:auto; overflow-y:hidden; margin:0 auto; border:1px solid #d9e2ec; border-radius:6px; background:#fff; }
    table.risk-table { width:100%; max-width:none; min-width:100%; margin:0 !important; box-sizing:border-box; table-layout:auto; border-collapse:collapse; color:#172033; background:#fff; font-size:.88rem; }
    table.risk-table thead th { background:#e7eff7; color:#102a43; font-weight:700; text-align:left; padding:.72rem .82rem; border-bottom:1px solid #c7d9e8; white-space:normal; overflow-wrap:normal; }
    table.risk-table tbody td { background:#fff; color:#172033; padding:.68rem .82rem; border-bottom:1px solid #e5edf4; overflow-wrap:normal; }
    table.risk-table tbody tr:nth-child(even) td { background:#f5f9fc; }
    table.risk-table tbody tr:last-child td { border-bottom:0; }
    table.risk-table td.numeric { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
    table.risk-table td.state-ok { color:#146356; background:#e0f2ee !important; font-weight:700; }
    table.risk-table td.state-alert { color:#8e1f1f; background:#fee8e7 !important; font-weight:700; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] { color:#172033; font-family:inherit; font-size:.94rem; line-height:1.72; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h1,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h2,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h3,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h4,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h5,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] h6 { color:#102a43; font-family:inherit; font-size:1rem !important; font-weight:700; line-height:1.5; margin:.9rem 0 .4rem; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] p,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] li,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] td,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] th { color:#26384b; font-family:inherit; font-size:.94rem !important; line-height:1.72; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] p { margin:.35rem 0 .7rem; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] ul,
    .st-key-ai-answer [data-testid="stMarkdownContainer"] ol { margin:.25rem 0 .8rem; padding-left:1.35rem; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] code { color:#146356; background:#eaf5f2; border-radius:3px; font-family:inherit; font-size:.92em; padding:.08rem .24rem; }
    .st-key-ai-answer [data-testid="stMarkdownContainer"] table { background:#fff; border-color:#d9e2ec; }
    </style>
    """,
    unsafe_allow_html=True,
)


def num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def status_html(label: str, tone: str = "blue") -> str:
    return f'<span class="status status-{tone}">{label}</span>'


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{kicker}</div>'
        f'<h1 class="page-title">{title}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def panel_title(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="table-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)


def show_table(frame: pd.DataFrame) -> None:
    """渲染浅色表格，避免系统深色主题覆盖表格配色。"""
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for column, value in zip(frame.columns, values):
            if pd.isna(value):
                display = "-"
            elif isinstance(value, float):
                display = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
            elif isinstance(value, int):
                display = f"{value:,}"
            else:
                display = str(value)
            safe_value = html.escape(display)
            css_class = "numeric" if isinstance(value, (int, float)) else ""
            if column in ("复核容量状态", "容量状态") or display in ("未超出", "已超出"):
                css_class = "state-ok" if display == "未超出" else "state-alert"
            cells.append(f'<td class="{css_class}">{safe_value}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    markup = (
        '<div class="data-table-wrap"><table class="risk-table">'
        f'<thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody>'
        '</table></div>'
    )
    st.markdown(markup, unsafe_allow_html=True)


def chart_base(chart: alt.Chart, height: int = 260) -> alt.Chart:
    return (
        chart.properties(
            height=height,
            background="#ffffff",
            padding={"left": 8, "right": 16, "top": 10, "bottom": 6},
        )
        .configure(background="#ffffff", font="Arial, Microsoft YaHei, sans-serif")
        .configure_view(stroke="#e5edf4", fill="#ffffff")
        .configure_axis(
            labelColor="#52667a", titleColor="#52667a", gridColor="#e2eaf1",
            domainColor="#c7d9e8", tickColor="#c7d9e8",
            labelFontSize=11, titleFontSize=11, gridOpacity=0.75,
        )
        .configure_title(color="#102a43", fontSize=13, anchor="start")
        .configure_legend(labelColor="#52667a", titleColor="#102a43", orient="top")
    )


def risk_distribution_chart(scores: pd.DataFrame) -> alt.Chart:
    data = (
        scores.assign(风险分数分箱=pd.cut(scores["风险分数"], bins=10).astype(str))
        .groupby("风险分数分箱", observed=True)
        .size()
        .reset_index(name="交易笔数")
    )
    return chart_base(
        alt.Chart(data).mark_bar(color="#0f766e", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("风险分数分箱:N", sort=None, axis=alt.Axis(title="风险分数区间", labelAngle=-35)),
            y=alt.Y("交易笔数:Q", axis=alt.Axis(title="交易笔数")),
            tooltip=[
                alt.Tooltip("风险分数分箱:N", title="风险区间"),
                alt.Tooltip("交易笔数:Q", title="交易笔数", format=","),
            ],
        ),
        260,
    )


def hourly_chart(profile: pd.DataFrame, field: str, color: str, title: str) -> alt.Chart:
    data = profile[["相对小时", field]].rename(columns={field: "数值"})
    return chart_base(
        alt.Chart(data).mark_line(
            color=color, strokeWidth=3,
            point=alt.OverlayMarkDef(color=color, size=45),
        ).encode(
            x=alt.X("相对小时:Q", axis=alt.Axis(title="相对小时")),
            y=alt.Y("数值:Q", axis=alt.Axis(title=title, format=".2%" if "比例" in title else ",.0f")),
            tooltip=[
                alt.Tooltip("相对小时:Q", title="相对小时"),
                alt.Tooltip("数值:Q", title=title, format=".2%" if "比例" in title else ",.2f"),
            ],
        ),
        220,
    )


def feature_chart(importance: pd.DataFrame) -> alt.Chart:
    data = importance.sort_values("重要性", ascending=True)
    return chart_base(
        alt.Chart(data).mark_bar(color="#2563eb", cornerRadiusEnd=4).encode(
            x=alt.X("重要性:Q", axis=alt.Axis(title="模型重要性", format=".3f")),
            y=alt.Y("特征:N", sort=None, axis=alt.Axis(title=None, labelLimit=150)),
            tooltip=[
                alt.Tooltip("特征:N", title="特征"),
                alt.Tooltip("重要性:Q", title="重要性", format=".4f"),
            ],
        ),
        430,
    )


def show_policy_result(policy: dict, total_count: int, capacity: float) -> None:
    """左右排列指标明细，窄屏由 Streamlit 自动堆叠。"""
    if "复核阈值" in policy:
        st.caption(f"复核阈值 {policy['复核阈值']:.7f} · 拦截阈值 {policy['拦截阈值']:.7f}")
    left, right = st.columns(2, gap="medium")
    with left:
        st.caption("交易处置（笔）")
        show_table(pd.DataFrame(
            [(key, policy[key]) for key in ("放行笔数", "复核笔数", "拦截笔数")],
            columns=["指标", "结果"],
        ))
        st.caption("误判与成本")
        show_table(pd.DataFrame(
            [(key, policy[key]) for key in ("预计漏放欺诈", "预计误拦正常", "模拟成本")],
            columns=["指标", "结果"],
        ))
    with right:
        st.caption("复核容量")
        show_table(pd.DataFrame([
            ("复核率", f"{policy['复核率']:.2%}（上限 {capacity:.0%}）"),
            ("复核容量上限", f"{int(total_count * capacity):,} 笔"),
            ("复核容量状态", "已超出" if policy["是否超出复核容量"] else "未超出"),
        ], columns=["指标", "结果"]))
        if "全部放行模拟成本" in policy:
            change = policy["相对全部放行成本变化"]
            st.caption("成本对照")
            show_table(pd.DataFrame([
                ("全部放行模拟成本", policy["全部放行模拟成本"]),
                ("相对全部放行成本变化", f"{change:.2%}" if change is not None else "无可比基线"),
            ], columns=["指标", "结果"]))


def show_overview(report: dict, metrics: dict, scores: pd.DataFrame, profile: pd.DataFrame) -> None:
    page_header("实验总览", "交易反欺诈风险工作台", "查看交易分布、模型表现和复核策略")
    source = "教学样本" if report["数据来源"] == "合成教学数据" else "公开信用卡样本"
    st.markdown('<div class="status-bar">' + status_html(f"数据源：{source}", "blue") + status_html("时间顺序切分", "green") + status_html("策略已冻结", "green") + '</div>', unsafe_allow_html=True)
    with st.container(border=True, key="experiment-summary"):
        final = report["测试集冻结策略表现"]
        state = "已超出" if final["是否超出复核容量"] else "未超出"
        st.markdown(
            '<div class="summary-content"><div class="section-kicker">本次实验结论</div>'
            f'<p class="finding">验证集选定 {html.escape(report["验证集选定模型"])}，'
            f'测试集 AP 为 {metrics["平均精确率AP"]:.4f}。'
            f'复核率 {final["复核率"]:.2%}，{state} {report["成本假设"]["最大复核比例"]:.0%} 上限。</p>'
            '<div class="small-note">测试时沿用验证集确定的模型和阈值；成本按设定的漏放、误拦和复核费用估算。</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    cols = st.columns(4)
    summaries = [
        ("交易总数", f"{report['总笔数']:,}"),
        ("欺诈笔数", f"{report['欺诈笔数']:,}"),
        ("测试集 AP", num(metrics["平均精确率AP"])),
        ("测试集 ROC-AUC", num(metrics["ROC-AUC"])),
    ]
    for col, (label, value) in zip(cols, summaries):
        with col:
            st.metric(label, value)
    left, right = st.columns([1.05, 1.35])
    with left:
        with st.container(border=True):
            panel_title("数据切分", "按交易时间顺序划分，避免未来信息进入训练")
            split = pd.DataFrame(report["时间切分"].items(), columns=["数据集", "交易笔数"])
            split["数据集"] = split["数据集"].replace({"训练集": "训练", "验证集": "验证", "测试集": "测试"})
            show_table(split)
            st.markdown('<div class="small-note">衍生特征：' + "、".join(report["特征工程"]) + '</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            panel_title("测试集风险分数分布", "分数用于风险排序，不等同于校准后的欺诈概率")
            st.altair_chart(risk_distribution_chart(scores), width="stretch")
    with st.container(border=True):
        panel_title("按相对时间统计", "相对小时从首笔交易起算，不对应真实钟点")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.caption("欺诈比例")
            st.altair_chart(hourly_chart(profile, "欺诈比例", "#b45309", "欺诈比例"), width="stretch")
        with chart_right:
            st.caption("平均交易金额")
            st.altair_chart(hourly_chart(profile, "平均金额", "#2563eb", "平均金额"), width="stretch")


def show_models(report: dict, metrics: dict) -> None:
    page_header("模型评估", "模型比较与测试结果", "按验证集 AP 选模型，再评估后续时间段的表现")
    selected = report["验证集选定模型"]
    st.markdown('<div class="status-bar">' + status_html(f"验证集选定：{selected}", "green") + status_html("关注指标：平均精确率 / 召回率 / 精确率", "blue") + '</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8])
    with left:
        with st.container(border=True):
            panel_title("验证集模型比较")
            show_table(pd.DataFrame(report["模型比较"]))
            st.caption("欺诈样本很少，准确率容易掩盖漏报，因此用 AP 比较风险排序效果。")
    with right:
        with st.container(border=True):
            panel_title("测试集指标")
            metric_rows = pd.DataFrame([
                {"指标": "平均精确率 AP", "结果": num(metrics["平均精确率AP"])},
                {"指标": "ROC-AUC", "结果": num(metrics["ROC-AUC"])},
                {"指标": "精确率", "结果": num(metrics["精确率"])},
                {"指标": "召回率", "结果": num(metrics["召回率"])},
                {"指标": "F1", "结果": num(metrics["F1"])},
            ])
            show_table(metric_rows)
    with st.container(border=True):
        panel_title("混淆矩阵", "分类阈值 0.5 仅用于展示，不是三段式策略阈值")
        matrix = pd.DataFrame([
            {"实际 / 预测": "正常", "预测正常": metrics["真阴性"], "预测欺诈": metrics["假阳性"]},
            {"实际 / 预测": "欺诈", "预测正常": metrics["假阴性"], "预测欺诈": metrics["真阳性"]},
        ])
        show_table(matrix)


def show_policy(report: dict) -> None:
    page_header("风险策略", "双阈值处置与复核容量", "在验证集确定阈值，查看测试集的处置结果与模拟成本")
    selected = report["验证集选定策略"]
    final = report["测试集冻结策略表现"]
    capacity = report["成本假设"]["最大复核比例"]
    cols = st.columns(4)
    summaries = [
        ("复核阈值", num(selected["复核阈值"], 5)),
        ("拦截阈值", num(selected["拦截阈值"], 5)),
        ("测试集复核率", f"{float(final['复核率']):.2%}"),
        ("测试集模拟成本", f"{final['模拟成本']:,.2f}"),
    ]
    for col, (label, value) in zip(cols, summaries):
        with col:
            st.metric(label, value)
    state = "已超出" if final["是否超出复核容量"] else "未超出"
    with st.container(border=True):
        st.markdown(f'<div class="section-kicker">复核容量状态</div><p class="finding">{state} · 当前复核率 {final["复核率"]:.2%} / 容量上限 {capacity:.0%}</p>', unsafe_allow_html=True)
        usage = min(final["复核率"] / capacity, 1.0) if capacity else 0.0
        st.progress(usage, text=f"容量使用率 {usage:.0%}" if capacity else "容量上限未设置")
        st.markdown(f'<div class="status-bar">{status_html("复核容量 " + state, "red" if final["是否超出复核容量"] else "green")}{status_html("测试集不重新调阈值", "blue")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        panel_title("验证集选定策略")
        show_policy_result(selected, report["时间切分"]["验证集"], capacity)
    with st.container(border=True):
        panel_title("测试集冻结策略", "沿用上述阈值，不按测试结果重新调整")
        show_policy_result(final, report["时间切分"]["测试集"], capacity)
    with st.container(border=True):
        panel_title("成本假设", "以下为设定值，未使用真实业务费用")
        show_table(pd.DataFrame(report["成本假设"].items(), columns=["成本参数", "数值"]))


def show_explain(report: dict) -> None:
    page_header("模型解释", "模型主要使用哪些特征", "查看所选模型的全局特征重要性")
    importance = pd.DataFrame(report["特征重要性"], columns=["特征", "重要性"])
    importance["特征"] = importance["特征"].replace({
        "Amount": "交易金额",
        "LogAmount": "金额对数",
        **{f"V{i}": f"匿名特征{i}" for i in range(1, 29)},
    })
    with st.container(border=True):
        panel_title("全局特征重要性")
        st.altair_chart(feature_chart(importance), width="stretch")
        show_table(importance)
    st.markdown('<div class="callout callout-warning">V1–V28 是 PCA 转换后的匿名特征，无法还原成设备、商户或地域。这里的数值反映模型对变量的使用程度，不表示欺诈原因。</div>', unsafe_allow_html=True)
    st.caption(report["解释方法"])


def show_ai(report: dict) -> None:
    page_header("智能分析", "AI 风控分析助手", "根据当前报告回答问题，模型和策略保持不变")
    try:
        env = load_ai_config(secrets=st.secrets)
    except FileNotFoundError:
        env = load_ai_config()
    configured = bool(env["AI_BASE_URL"] and env["AI_MODEL"])
    st.markdown('<div class="status-bar">' + status_html("AI 服务已配置" if configured else "使用离线摘要", "green" if configured else "amber") + status_html("仅发送报告摘要", "blue") + '</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-kicker">配置状态</div>', unsafe_allow_html=True)
        if configured:
            key_status = "API Key 已设置" if env["AI_API_KEY"] else "未设置密钥（需接口支持免密调用）"
            st.markdown(
                f'<p class="finding">{html.escape(env["AI_MODEL"])}</p>'
                f'<div class="small-note">接口：{html.escape(env["AI_BASE_URL"])} · '
                f'等待上限：{request_timeout(env):.0f} 秒 · {key_status}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="callout callout-warning">尚未配置 AI 服务，当前使用离线中文摘要。'
                '云端请在「管理应用 → Settings → Secrets」填写 AI_BASE_URL、AI_MODEL 和 AI_API_KEY；'
                '本地可继续在项目根目录 .env 中填写。</div>',
                unsafe_allow_html=True,
            )
    examples = [
        "为什么验证集选择 XGBoost？",
        "测试集指标说明了什么？",
        "复核容量是否超出？",
        "模拟成本下降代表线上收益吗？",
        "匿名特征重要性可以解释成设备或商户吗？",
    ]
    st.session_state.setdefault("ai_question", examples[0])
    st.selectbox(
        "示例问题", examples, key="ai_example", persist_state="session",
        on_change=lambda: st.session_state.update(ai_question=st.session_state.ai_example),
    )
    question = st.text_area(
        "问题", key="ai_question", persist_state="session", height=90, label_visibility="collapsed",
    )
    analysis = st.session_state.get("ai_analysis")
    if analysis and analysis["report"] != report:
        st.session_state.pop("ai_analysis")
        analysis = None
    if st.button("生成分析", type="primary", key="ai_submit"):
        with st.spinner("正在生成风控分析…"):
            result = answer_question(question, report=report, environ=env)
        analysis = {"question": question, "report": report, "result": result}
        st.session_state["ai_analysis"] = analysis
    if analysis:
        result = analysis["result"]
        if result["mode"] == "ai":
            st.success("已根据当前报告生成分析")
        elif result["mode"] == "offline-fallback":
            st.warning(result.get("error", "AI 调用失败，已使用离线摘要"))
        else:
            st.info("当前使用离线摘要")
        with st.container(key="ai-answer", border=True):
            st.markdown('<div class="section-kicker">分析结果</div>', unsafe_allow_html=True)
            st.caption(f"分析问题：{analysis['question']}")
            st.markdown(result["answer"])
        with st.container(border=True):
            panel_title("本次回答使用的报告字段")
            show_table(pd.DataFrame({"字段": result["sources"]}))


if not REPORT.exists():
    page_header("实验状态", "尚未生成训练报告", "请先运行一次 demo 或 train")
    st.stop()

report = json.loads(REPORT.read_text(encoding="utf-8"))
metrics = report["测试集指标"]
scores = pd.read_csv(ROOT / "reports" / "test_scores.csv")
profile = pd.read_csv(ROOT / "reports" / "hour_profile.csv")

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">风险分析</div><div class="brand-title">反欺诈风险工作台</div><div class="brand-subtitle">数据挖掘 · 模型评估 · 成本策略</div></div>', unsafe_allow_html=True)
    nav = st.radio(
        "工作区", ["数据概览", "模型比较", "风险策略", "特征解释", "AI 风控分析"],
        label_visibility="collapsed",
    )
    st.divider()
    source = "教学样本" if report["数据来源"] == "合成教学数据" else "公开信用卡样本"
    st.caption(f"数据源：{source}")
    st.caption(f"模型：{report['验证集选定模型']}")
    st.caption("结果：离线实验")

if report["数据来源"] == "合成教学数据":
    st.markdown('<div class="callout callout-warning">当前为教学样本，仅用于展示建模流程；简历成果应使用公开信用卡数据结果。</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="callout">当前为公开信用卡样本。模型和策略结果来自离线时间切分，成本为模拟假设。</div>', unsafe_allow_html=True)

if nav == "数据概览":
    show_overview(report, metrics, scores, profile)
elif nav == "模型比较":
    show_models(report, metrics)
elif nav == "风险策略":
    show_policy(report)
elif nav == "特征解释":
    show_explain(report)
else:
    show_ai(report)
