"""交易反欺诈数据挖掘与风险决策结果看板。"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "latest.json"

st.set_page_config(page_title="交易反欺诈数据挖掘与风险决策", page_icon=":material/shield:", layout="wide")
st.title("交易反欺诈数据挖掘与风险决策")
st.caption("公开交易数据 · 不平衡分类 · 风险排序 · 成本敏感策略")

if not REPORT.exists():
    st.info("还没有训练报告。请先运行：python run.py train --data data/creditcard.csv")
    st.stop()

report = json.loads(REPORT.read_text(encoding="utf-8"))
if report["数据来源"] == "合成教学数据":
    st.warning("当前为合成数据演示；简历成果请使用公开信用卡 CSV 的训练报告。")
else:
    st.success("当前使用公开信用卡 CSV；模型和策略结果来自离线时间切分，成本仍是假设下的模拟。")

metrics = report["测试集指标"]
with st.container(horizontal=True):
    st.metric("交易总数", f"{report['总笔数']:,}", border=True)
    st.metric("欺诈笔数", f"{report['欺诈笔数']:,}", border=True)
    st.metric("测试集 AP", f"{metrics['平均精确率AP']:.4f}", border=True)
    st.metric("测试集 ROC-AUC", f"{metrics['ROC-AUC']:.4f}", border=True)

overview, models, policy, explain = st.tabs(["数据概览", "模型比较", "风险策略", "特征解释"])
with overview:
    st.subheader("数据质量与特征工程")
    st.dataframe(pd.DataFrame({"项目": ["缺失值总数", "重复行数"],
                               "结果": [report["数据质量"]["缺失值总数"], report["数据质量"]["重复行数"]]}),
                 hide_index=True)
    st.write("使用的衍生特征：", "、".join(report["特征工程"]))
    st.subheader("时间切分")
    split = pd.DataFrame([report["时间切分"]]).T.reset_index()
    split.columns = ["数据集", "交易笔数"]
    st.dataframe(split, hide_index=True, width="stretch")
    st.caption("训练、验证、测试按交易时间顺序切分；验证集用于选模型和阈值，测试集只做最终评估。")
    scores = pd.read_csv(ROOT / "reports" / "test_scores.csv")
    st.subheader("测试集风险分数分布")
    st.bar_chart(scores.assign(风险分数分箱=pd.cut(scores["风险分数"], bins=10).astype(str))
                 .groupby("风险分数分箱", observed=True).size().rename("交易笔数"))
    st.subheader("按相对小时的数据画像")
    profile = pd.read_csv(ROOT / "reports" / "hour_profile.csv")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.caption("欺诈比例")
        st.line_chart(profile.set_index("相对小时")["欺诈比例"])
    with chart_right:
        st.caption("平均交易金额")
        st.line_chart(profile.set_index("相对小时")["平均金额"])

with models:
    st.subheader("验证集模型比较")
    st.dataframe(pd.DataFrame(report["模型比较"]), hide_index=True, width="stretch")
    st.metric("验证集选定模型", report["验证集选定模型"])
    st.subheader("测试集分类指标")
    metric_table = pd.DataFrame([metrics]).T.reset_index()
    metric_table.columns = ["指标", "数值"]
    st.dataframe(metric_table, hide_index=True, width="stretch")
    st.caption("AP 比 Accuracy 更适合极度不平衡任务；0.5 是分类指标的展示阈值，不是策略阈值。")

with policy:
    st.subheader("验证集选出的三段式策略")
    selected = report["验证集选定策略"]
    def policy_table(policy, total_count):
        row = dict(policy)
        capacity = report["成本假设"]["最大复核比例"]
        row.pop("是否超出复核容量", None)
        row["复核容量状态"] = "已超出" if policy["是否超出复核容量"] else "未超出"
        row["复核率"] = f"{policy['复核率']:.2%}（上限 {capacity:.0%}）"
        row["复核容量上限"] = f"{int(total_count * capacity):,} 笔"
        return pd.DataFrame([row])

    st.dataframe(policy_table(selected, report["时间切分"]["验证集"]), hide_index=True, width="stretch")
    st.caption("复核容量状态按该数据集的实际复核笔数与 2% 业务容量上限判断。")
    st.subheader("冻结策略在测试集上的表现")
    final = report["测试集冻结策略表现"]
    st.dataframe(policy_table(final, report["时间切分"]["测试集"]), hide_index=True, width="stretch")
    st.json(report["成本假设"], expanded=False)
    st.caption("漏放欺诈、误拦正常和复核成本都是演示假设；测试集没有重新调阈值。")

with explain:
    st.subheader("模型特征重要性")
    importance = pd.DataFrame(report["特征重要性"], columns=["特征", "重要性"])
    importance["特征"] = importance["特征"].replace({"Amount": "交易金额", "LogAmount": "金额对数",
                                                       **{f"V{i}": f"匿名特征{i}" for i in range(1, 29)}})
    st.bar_chart(importance.set_index("特征"), horizontal=True)
    st.caption(report["解释方法"])
    st.info("匿名特征 V1–V28 来自公开数据的主成分变换，不能直接解释为设备、商户或地域。特征重要性表示模型使用程度，不代表因果关系。")
