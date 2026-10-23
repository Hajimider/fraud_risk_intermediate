"""常用命令：python run.py demo 或 python run.py train。"""
import argparse
import json
from pathlib import Path

from fraud_model import ROOT, run


def main():
    parser = argparse.ArgumentParser(description="反欺诈进阶入门版")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="用合成教学数据运行")
    demo.add_argument("--rows", type=int, default=12000)
    demo.add_argument("--trees", type=int, default=80)
    train = sub.add_parser("train", help="读取 data/creditcard.csv 运行")
    train.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    train.add_argument("--trees", type=int, default=160)
    sub.add_parser("dashboard", help="启动中文结果看板")
    args = parser.parse_args()
    if args.command == "dashboard":
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")],
                       cwd=ROOT, check=True)
        return
    report = run(rows=args.rows, trees=args.trees) if args.command == "demo" else run(args.data, trees=args.trees)
    print(json.dumps({"数据来源": report["数据来源"], "胜出模型": report["验证集选定模型"],
                      "测试集指标": report["测试集指标"], "测试集模拟成本": report["测试集冻结策略表现"]["模拟成本"]},
                     ensure_ascii=False, indent=2))
    print(f"完整报告：{ROOT / 'reports' / 'latest.json'}")


if __name__ == "__main__":
    main()
