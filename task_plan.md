# 交易反欺诈数据挖掘项目实施计划

- [complete] 保存独立版本设计与文件边界
- [complete] 实现训练、评估、成本策略和中文看板
- [complete] 添加关键测试并运行教学训练
- [complete] 实际切换看板区域并做最终检查
- [complete] 使用公开信用卡 CSV 生成可核实结果并更新项目文档
- [complete] 运行看板自动化验证并完成独立审查
- [complete] 规划 AI 风控分析助手
- [complete] 实现 AI 风控分析助手并完成编码测试

## AI 规划阶段交接

- 计划文件：`docs/AI功能规划.md`
- 推荐实现：结构化报告问答 + OpenAI-compatible 可选接口 + 无配置时离线摘要
- 主要新增文件：`ai_assistant.py`、`tests/test_ai_assistant.py`，并增量修改 `dashboard.py`、`README.md`
- 已新增 AI 助手、测试和看板区域；未发起真实 AI API 调用，未新增第三方依赖

## 独立审查结果

- 已检查 AI 上下文边界、错误回退、密钥读取方式、README 运行说明和页面数据展示。
- 首轮审查发现并修复 2 个重要问题：AI 字段引用固定、未覆盖 RuntimeError 回退；同时修正 README 页数描述并补强测试覆盖。
- 修复后复查未发现严重或重要缺陷；一次边界检查脚本误报已复核，原因是匹配了代码注释中的安全说明。
- 真实 API 未调用，provider 质量和费用未验证。
- `.env` 自动读取已验证；当前 `.env` 仍是占位符，因此直接运行会使用离线摘要。
