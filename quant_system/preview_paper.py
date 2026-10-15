"""Render a historical paper report with the new review card; no network or trading."""
import copy
import glob
import json
from pathlib import Path

from paper_research import research_review, strategy_identity
from report_mailer import generate_html_report


def main():
    root = Path(__file__).resolve().parent
    paths = sorted(glob.glob(str(root / "report_paper_[0-9]*.json")))
    if not paths:
        raise SystemExit("没有可供预览的历史模拟盘报告")
    report = json.loads(Path(paths[-1]).read_text(encoding="utf-8"))
    report = copy.deepcopy(report)
    date = report["date"]
    # Historical report has no verified unadjusted bars/actions. Do not invent them.
    portfolio = {"_available_cash": report["portfolio"].get("available_cash", 0)}
    report["research"] = research_review(
        portfolio, report["portfolio"], report.get("timing", {}), [],
        strategy_identity(str(root), []), {}, {}, date, 0., [])
    report["research"]["warnings"].insert(0, f"历史展示预览：数据日期{date}，不是今天的日报；没有重算成交、分红或策略收益。新版本观察从实际首次运行开始。")
    folder = root / "review_outputs"
    folder.mkdir(exist_ok=True)
    target = folder / f"paper_preview_{date}.html"
    target.write_text(generate_html_report(report), encoding="utf-8")
    print(f"历史展示预览已生成：{target.name}；未改账户、未联网、未发信。")


if __name__ == "__main__":
    main()
