"""常用命令：python run.py demo 或 python run.py train。"""

import argparse
import json
import subprocess
import sys

from fraud_model import ROOT, run


def main():
    parser = argparse.ArgumentParser(description="交易反欺诈数据挖掘与风险决策")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="用合成教学数据运行")
    demo.add_argument("--rows", type=int, default=12000)
    demo.add_argument("--trees", type=int, default=80)
    train = sub.add_parser("train", help="读取 data/creditcard.csv 运行")
    train.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    train.add_argument("--trees", type=int, default=160)
    for command in (demo, train):
        command.add_argument("--output-dir", default="reports", help="实验输出目录，避免覆盖已有结果")
        command.add_argument("--tuning-trials", type=int, choices=(1, 2, 3), default=3)
        command.add_argument("--bootstrap-iterations", type=int, default=1000)
    sub.add_parser("dashboard", help="启动中文结果看板")
    ieee = sub.add_parser("ieee-prepare", help="接入已获授权的IEEE-CIS原始数据并生成MySQL窗口特征")
    ieee.add_argument("--transactions", default="data/ieee/train_transaction.csv")
    ieee.add_argument("--identity", default="data/ieee/train_identity.csv")
    ieee.add_argument("--output-dir", default="data/ieee")
    ieee.add_argument("--sample-rows", type=int)
    ieee_train = sub.add_parser("ieee-train", help="按相对月比较WOE评分卡、LightGBM、XGBoost并输出独立报告")
    ieee_train.add_argument("--data", default="data/ieee/transactions.parquet")
    ieee_train.add_argument("--output-dir", default="reports/ieee")
    ieee_train.add_argument("--trees", type=int, default=160)
    ieee_train.add_argument("--tuning-trials", type=int, choices=(1, 2, 3), default=3)
    ieee_train.add_argument("--bootstrap-iterations", type=int, default=1000)
    ieee_train.add_argument("--skip-shap", action="store_true", help="显式跳过SHAP，报告会记录为空")
    args = parser.parse_args()
    if args.command == "ieee-train":
        from ieee_experiments import run_ieee
        result = run_ieee(args.data, args.output_dir, args.trees, args.tuning_trials, args.bootstrap_iterations, export_shap=not args.skip_shap)
        print(f"IEEE实验完成，选定模型：{result['选定模型']}；报告：{args.output_dir}/ieee_report.json")
        return
    if args.command == "ieee-prepare":
        from ieee_pipeline import prepare_ieee
        result = prepare_ieee(args.transactions, args.identity, args.output_dir, args.sample_rows)
        for key, value in result.items():
            print(f"{key}：{value}")
        return
    if args.command == "dashboard":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")],
            cwd=ROOT,
            check=True,
        )
        return

    options = dict(trees=args.trees, output_dir=args.output_dir,
                   tuning_trials=args.tuning_trials, bootstrap_iterations=args.bootstrap_iterations)
    if args.command == "demo":
        report = run(rows=args.rows, **options)
    else:
        report = run(args.data, **options)
    summary = {
        "数据来源": report["数据来源"],
        "选定模型": report["验证集选定模型"],
        "测试集指标": report["测试集指标"],
        "测试集模拟成本": report["测试集冻结策略表现"]["模拟成本"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"完整报告：{args.output_dir}/latest.json")


if __name__ == "__main__":
    main()
