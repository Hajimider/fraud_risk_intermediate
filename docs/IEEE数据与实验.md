# IEEE-CIS数据与实验

## 当前验收边界

本机已从Kaggle下载官方`train_transaction.csv`（683,351,067字节）与`train_identity.csv`（26,529,680字节），并完成文件大小校验；原始文件不随仓库提交。全量交易表已合并为590,540行Parquet，历史特征与成熟目标编码已完成，真实全量实验报告写入`reports/ieee/ieee_report.json`。三个月验证集为92,189、98,615、64,922笔，独立测试月为92,427笔；最终选定XGBoost（`max_depth=5`），测试AP为0.4499、ROC-AUC为0.8705、Brier为0.024019、ECE为0.007573。本次运行使用160棵树、每类3个候选、1,000次Bootstrap并显式跳过SHAP；小规模防泄漏用例及合成完整实验仍通过。

原始CSV受Kaggle比赛规则约束，不随仓库分发。请先登录有权使用数据的Kaggle账号，在[比赛数据页](https://www.kaggle.com/competitions/ieee-fraud-detection/data)接受比赛规则并下载`train_transaction.csv`、`train_identity.csv`，将两个文件放入`data/ieee/`。项目不自动登录或代替用户接受协议；文件也已加入Git忽略规则，不会随代码提交。

## 运行

将`train_transaction.csv`和`train_identity.csv`放在`data/ieee/`。先用样本调通：

```bash
python run.py ieee-prepare --sample-rows 100000
```

接入全量数据后训练：

```bash
python run.py ieee-prepare
python run.py ieee-train
```

默认使用MySQL。PyCharm脚本选择`run.py`、工作目录选择项目根目录，参数填写`ieee-prepare`或`ieee-train`；解释器选择安装了项目依赖的环境。

接入输出保存在`data/ieee/`。全量运行使用`transactions.parquet`、`temporal_features.parquet`、`ieee.source.json`和`preparation.json`；带`--sample-rows N`时文件名会带`.sample-N`后缀，例如`transactions.sample-100000.parquet`和`temporal_features.sample-100000.parquet`。原始文件与Parquet被Git忽略。当前`transactions.parquet`为全量数据；样本特征文件和样本准备记录不可与全量交易表配套训练。实验输出保存在`reports/ieee/`，不覆盖ULB报告；包括JSON报告、筛选表、消融表和SHAP图片。

`ieee-prepare`生成的70%截止特征仅作接入检查。`ieee-train`不会直接拿该矩阵做所有月份验证，而是先复用不依赖标签截止的历史窗口特征，再按每个窗口的拟合截止计算轻量的成熟目标编码，避免早期验证月读到后续标签并减少重复窗口计算。

## 特征与标签口径

| 类别 | 实现 |
| --- | --- |
| 实体 | 卡、地址、邮箱、设备、近似uid |
| 频次与速度 | 过去1小时/24小时频次、24小时金额、历史频次 |
| 金额偏离 | 历史均值、总体标准差、当前金额标准分 |
| 新颖性 | 首次出现标记、距上次出现秒数 |
| 编码 | 历史频率、带历史先验的平滑目标编码 |
| 时间 | 相对日内小时、相对周内日；不声称真实钟点或星期 |

五类实体各10个历史特征，加2个相对时间字段，共52项历史特征。MySQL数值RANGE窗口以1 PRECEDING结尾，同秒交易不互相可见。空历史保留缺失，插补只从拟合段学习。

近似uid为`card1 + addr1 + floor(TransactionDT / 86400 - D1)`，参考比赛公开讨论中常见的首笔交易日构造方式。它可能将不同用户合并或将同一用户拆开，缺失字段还会混组，因此报告提供交易数与标签一致性检查，不把它当作真实用户ID。该构造尚未用真实样本验证。

目标编码的标签仅来自基础模型拟合截止之前，并额外假设7天标签成熟延迟。模型拟合、早停和校准的目标也只使用验证预测起点前至少7天的标签；用于最终选型和策略阈值的验证标签，至少在测试起点前7天产生。间隔中的交易可以提供无标签频次历史，但不能用于监督选择。验证/测试标签即使存在于本地文件也不进入特征编码。7天为演示假设，不能代替真实标签回流时延。

## 评估设计

- 原始TransactionDT未核验真实日历起点，使用相对30天分块，至少4块；最后一块只测试，不能宣传为真实自然月/星期。
- 每个滚动训练月内部依时间分为拟合、早停和校准段，特征编码截止冻结到拟合段。
- WOE分箱、平滑和映射仅从拟合段学习；评分卡用类别加权LR。IV筛选阈值0.02、缺失率95%、PSI阈值0.25都是演示启发式，PSI比较拟合与内部早停段。
- WOE评分卡、LightGBM与XGBoost各有相同的3个候选预算；树模型只在内部早停段停止，按滚动验证平均AP选择。
- 基础字段→频次→偏离→新颖性→编码逐组消融，给出验证AP增量与配对Bootstrap区间。消融为另一次固定超参数对照，不与最终筛选后的特征矩阵混称。
- 用同一冻结模型的校准段拟合Platt与Isotonic，在最近验证月按Brier选择，验证月选成本阈值；测试月只评估冻结方案。
- 输出逐验证月与测试月的特征PSI、分数PSI及“训练一次/按月重训”AP。测试对照事先约定，不用于反向选模型。
- SHAP输出一个全局图，以及实际存在的真阳性、漏判、误报各一例。若某类不存在则不伪造案例；`--skip-shap`会明确不生成图片。

比赛公开排行榜常采用与本项目不同的随机划分、完整时间范围、特征或标签使用口径。未核验可比基线前，不虚构排行榜差距数字。真实报告记录在`reports/ieee/ieee_report.json`，其中SHAP文件为空是因为本次正式运行显式使用了`--skip-shap`；报告中的策略成本仍是固定假设下的离线模拟，不能当作线上收益。

## 检查

```bash
python -m pytest tests/test_ieee_pipeline.py -q
```

完整测试需要本机MySQL。只验证WOE/PSI的无需数据库子集：

```bash
python -m pytest tests/test_ieee_pipeline.py -q -k woe
```

真实IEEE样本的来源哈希、规模、uid质量、特征筛选、消融、测试指标和月度漂移已在`reports/ieee/ieee_report.json`及同目录CSV中验收；SHAP需单独在更宽裕资源上运行，不能把本次空列表解释为已生成SHAP图。
