from html.parser import HTMLParser
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ai_assistant

class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers = []
        self.cells = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag in {"th", "td"}:
            self.cell = (tag, [])

    def handle_data(self, data):
        if self.cell:
            self.cell[1].append(data)

    def handle_endtag(self, tag):
        if self.cell and tag == self.cell[0]:
            target = self.headers if tag == "th" else self.cells
            target.append("".join(self.cell[1]))
            self.cell = None


def test_policy_groups_keep_results_readable_and_complete():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports" / "latest.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("风险策略").run(timeout=30)
    assert not app.exception
    tables = []
    for item in app.markdown:
        if '<table class="risk-table">' in item.value:
            table = TableReader()
            table.feed(item.value)
            tables.append(table)
    assert tables and all(table.headers == ["指标", "结果"] for table in tables[:-1])
    rows = [dict(zip(table.cells[::2], table.cells[1::2])) for table in tables[:-1]]
    validation = {key: value for row in rows[:3] for key, value in row.items()}
    test = {key: value for row in rows[3:] for key, value in row.items()}
    as_count = lambda value: int(value.replace(",", ""))
    assert sum(as_count(validation[key]) for key in ("放行笔数", "复核笔数", "拦截笔数")) == report["时间切分"]["验证集"]
    assert sum(as_count(test[key]) for key in ("放行笔数", "复核笔数", "拦截笔数")) == report["时间切分"]["测试集"]
    assert "复核率" in validation and "模拟成本" in validation
    assert "相对全部放行成本变化" in test
    captions = "\n".join(item.value for item in app.caption)
    assert f"{report['验证集选定策略']['复核阈值']:.7f}" in captions


def test_analysis_pages_show_sql_rolling_validation_and_ablation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"))
    app.run(timeout=30)
    markdown = "\n".join(item.value for item in app.markdown)
    assert not app.exception
    assert "SQL 数据画像" in markdown
    app.sidebar.radio[0].set_value("模型比较").run(timeout=30)
    markdown = "\n".join(item.value for item in app.markdown)
    assert "滚动验证 AP" in markdown
    assert "金额对数特征消融" in markdown
    assert not app.exception


@pytest.mark.parametrize("mode", ["ai", "offline", "offline-fallback"])
def test_ai_answer_survives_navigation_without_another_request(monkeypatch, mode):
    monkeypatch.setattr(ai_assistant, "load_dotenv", lambda: {})
    for key in ai_assistant.AI_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    calls = []

    def fake_answer(question, **kwargs):
        calls.append(question)
        return {
            "mode": mode, "answer": f"保留的分析结果 {len(calls)}",
            "sources": ["测试集指标", "成本假设"], "error": "模拟接口不可用",
        }

    monkeypatch.setattr(ai_assistant, "answer_question", fake_answer)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("AI 风控分析").run(timeout=30)
    app.text_area(key="ai_question").set_value("请分析当前复核率").run(timeout=30)
    app.button(key="ai_submit").click().run(timeout=30)
    for page in ("数据概览", "模型比较", "风险策略", "特征解释", "AI 风控分析"):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "保留的分析结果 1" in rendered
    assert "测试集指标" in rendered and "成本假设" in rendered
    assert app.text_area(key="ai_question").value == "请分析当前复核率"
    assert calls == ["请分析当前复核率"]

    app.selectbox(key="ai_example").set_value("测试集指标说明了什么？").run(timeout=30)
    assert app.text_area(key="ai_question").value == "测试集指标说明了什么？"
    assert calls == ["请分析当前复核率"]
    app.button(key="ai_submit").click().run(timeout=30)
    assert len(calls) == 2
    assert any("保留的分析结果 2" in item.value for item in app.markdown)
    assert not any("保留的分析结果 1" in item.value for item in app.markdown)

    # 报告换了以后不能继续显示旧结论，也不能自动发起新请求。
    app.session_state["ai_analysis"]["report"]["总笔数"] += 1
    app.run(timeout=30)
    assert not app.exception
    assert "ai_analysis" not in app.session_state
    assert len(calls) == 2
