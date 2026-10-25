import json

from ai_assistant import (answer_question, build_context, build_messages, load_dotenv,
                          load_report, normalize_answer, offline_answer, request_timeout)


def test_context_is_small_and_excludes_raw_test_rows():
    report = load_report()
    report["raw_rows"] = [{"欺诈标签": 1, "风险分数": 0.99, "卡号": "masked"}]
    report["测试集逐行标签"] = [0, 1, 0]
    context = build_context(report)
    text = json.dumps(context, ensure_ascii=False)
    assert len(context["特征重要性"]) == 15
    assert '"风险分数":' not in text
    assert "欺诈标签" not in text
    assert "raw_rows" not in text
    assert "测试集逐行标签" not in text
    assert "测试集指标" in context


def test_offline_answer_contains_report_fact_and_sources():
    report = load_report()
    result = offline_answer("测试集指标说明了什么？", report)
    assert result["mode"] == "offline"
    assert "0.7675" in result["answer"]
    assert "测试集指标" in result["sources"]
    assert "字段" in result["answer"]


def test_missing_ai_configuration_stays_offline():
    result = answer_question(
        "为什么选择当前模型？",
        report=load_report(),
        environ={"AI_BASE_URL": "", "AI_MODEL": ""},
    )
    assert result["mode"] == "offline"
    assert "XGBoost" in result["answer"]


def test_dotenv_loader_reads_quoted_values_without_overwriting_process_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nAI_BASE_URL="http://mock.local/v1"\nAI_MODEL=demo\n',
        encoding="utf-8",
    )
    values = load_dotenv(env_file)
    assert values == {"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo"}


def test_placeholder_dotenv_stays_offline():
    result = answer_question(
        "测试集指标说明了什么？",
        report=load_report(),
        environ={
            "AI_BASE_URL": "https://your-compatible-endpoint/v1",
            "AI_MODEL": "your-model-name",
            "AI_API_KEY": "your-api-key",
        },
    )
    assert result["mode"] == "offline"


def test_openai_compatible_transport_is_injected_and_context_is_bounded():
    captured = {}

    def fake_transport(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {"choices": [{"message": {"content": "报告显示当前模型排序效果较好。"}}]}

    result = answer_question(
        "为什么选择当前模型？",
        report=load_report(),
        environ={"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo", "AI_API_KEY": "secret"},
        transport=fake_transport,
    )
    assert result["mode"] == "ai"
    assert captured["url"] == "http://mock.local/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 90.0
    prompt = json.dumps(captured["payload"], ensure_ascii=False)
    assert "secret" not in prompt
    assert "creditcard.csv" not in prompt
    assert "0.7675" in result["answer"]


def test_api_error_falls_back_without_exposing_secret():
    def failing_transport(url, payload, headers, timeout):
        raise OSError("simulated network failure")

    result = answer_question(
        "测试集指标说明了什么？",
        report=load_report(),
        environ={"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo", "AI_API_KEY": "secret"},
        transport=failing_transport,
    )
    assert result["mode"] == "offline-fallback"
    assert "0.7675" in result["answer"]
    assert "secret" not in result["error"]


def test_unexpected_api_error_also_falls_back():
    def failing_transport(url, payload, headers, timeout):
        raise RuntimeError("simulated provider failure")

    result = answer_question(
        "测试集指标说明了什么？",
        report=load_report(),
        environ={"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo"},
        transport=failing_transport,
    )
    assert result["mode"] == "offline-fallback"
    assert "0.7675" in result["answer"]


def test_invalid_api_payload_falls_back():
    def invalid_transport(url, payload, headers, timeout):
        return {"unexpected": "shape"}

    result = answer_question(
        "测试集指标说明了什么？",
        report=load_report(),
        environ={"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo"},
        transport=invalid_transport,
    )
    assert result["mode"] == "offline-fallback"
    assert "0.7675" in result["answer"]


def test_ai_sources_follow_question_topic():
    def fake_transport(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": "特征解释需要谨慎。"}}]}

    result = answer_question(
        "匿名特征可以解释成设备吗？",
        report=load_report(),
        environ={"AI_BASE_URL": "http://mock.local/v1", "AI_MODEL": "demo"},
        transport=fake_transport,
    )
    assert result["mode"] == "ai"
    assert "特征重要性" in result["sources"]
    assert "解释方法" in result["sources"]
    assert "验证集选定策略" not in result["sources"]


def test_messages_truncate_long_user_question():
    messages = build_messages("问题" * 1000, build_context(load_report()))
    user_content = messages[1]["content"]
    assert len(user_content.split("用户问题：", 1)[1]) <= 500


def test_request_timeout_can_be_configured_and_is_bounded():
    assert request_timeout({}) == 90.0
    assert request_timeout({"AI_TIMEOUT_SECONDS": "120"}) == 120.0
    assert request_timeout({"AI_TIMEOUT_SECONDS": "bad"}) == 90.0
    assert request_timeout({"AI_TIMEOUT_SECONDS": "999"}) == 300.0


def test_normalize_answer_removes_repeated_report_fields_and_keeps_sections():
    text = "## 报告事实\n指标结果。\n\n## 限制或建议\n- 建议复核。\n\n**使用的报告字段**：测试集指标、成本假设"
    normalized = normalize_answer(text)
    assert "报告事实" in normalized
    assert "限制或建议" in normalized
    assert "使用的报告字段" not in normalized
    assert "成本假设" not in normalized
