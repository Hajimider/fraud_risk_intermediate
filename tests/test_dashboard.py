from html.parser import HTMLParser
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
    assert validation == {
        "放行笔数": "47,755", "复核笔数": "239", "拦截笔数": "1",
        "复核率": "0.50%（上限 2%）", "复核容量上限": "959 笔", "复核容量状态": "未超出",
        "预计漏放欺诈": "11.90", "预计误拦正常": "1.90", "模拟成本": "1,559.90",
    }
    assert test == {
        "放行笔数": "43,662", "复核笔数": "176", "拦截笔数": "0",
        "复核率": "0.40%（上限 2%）", "复核容量上限": "876 笔", "复核容量状态": "未超出",
        "预计漏放欺诈": "16.20", "预计误拦正常": "1.34", "模拟成本": "1,892.04",
        "全部放行模拟成本": "5,400.00", "相对全部放行成本变化": "-64.96%",
    }
    captions = "\n".join(item.value for item in app.caption)
    assert "0.1792703" in captions and "0.9998300" in captions


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
