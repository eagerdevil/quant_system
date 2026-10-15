#!/usr/bin/env python
"""
量化系统 v7.1 — 绩效追踪器
==========================
模块: 历史绩效分析
功能: 权益曲线 / 胜率 / 盈亏比 / 最大回撤 / 月度汇总

读取历史 report_*.json 文件，告诉你：这个系统帮你赚钱了吗？
"""

import json, os, sys, io, logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

TODAY = datetime.now().strftime("%Y%m%d")


def load_history(report_dir, days=90, prefix="report_"):
    """加载历史报告，返回按日期排序的列表
    prefix: 报告文件前缀（实盘"report_"，模拟盘"report_paper_"），缺省保持实盘行为
    """
    reports = []
    if not os.path.isdir(report_dir):
        return reports

    for fname in sorted(os.listdir(report_dir)):
        # 前缀后必须是8位数字日期(20260814), 排除 report_paper_* 模拟盘报告污染实盘绩效(P0修复)
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        if not fname[len(prefix):len(prefix)+8].isdigit():
            continue
        path = os.path.join(report_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            date_str = fname.replace(prefix, "").replace(".json", "")
            data["_date"] = date_str
            reports.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    # 按日期排序，取最近days天
    reports.sort(key=lambda r: r.get("_date", ""))
    return reports[-days:] if len(reports) > days else reports


def compute_equity_curve(reports):
    """
    从历史报告中提取权益曲线
    返回: [{date, total_assets, total_value, pnl, cash, cash_pct}]
    """
    curve = []
    for r in reports:
        portfolio = r.get("portfolio", {})
        if not portfolio:
            continue

        curve.append({
            "date": r.get("_date", ""),
            "total_assets": portfolio.get("total_assets", 0),
            "total_value": portfolio.get("total_value", 0),
            "total_pnl": portfolio.get("total_pnl", 0),
            "total_pnl_pct": portfolio.get("total_pnl_pct", 0),
            "cash": portfolio.get("available_cash", 0),
            "cash_pct": portfolio.get("cash_ratio", 0),
            "holdings_count": len(portfolio.get("holdings", []))
        })
    return curve


def compute_performance_metrics(equity_curve, net_outflow=0.0, per_point_adjust=None):
    """
    计算核心绩效指标
    net_outflow: v7.6 期间净出金(提现-入金)，提现不算亏损，加回后再算收益
    per_point_adjust: v7.6 逐日调整量列表(与equity_curve等长)，用于回撤等逐点指标
    """
    if len(equity_curve) < 2:
        return {"error": "数据不足，至少需要2天"}

    assets = [e["total_assets"] for e in equity_curve]
    pnls = [e["total_pnl"] for e in equity_curve]

    # v7.6: 剔除期间出入金 — 原口径把提现算成亏损(-56.2%假象)
    if per_point_adjust:
        assets = [a + adj for a, adj in zip(assets, per_point_adjust)]

    # 累计收益
    start_assets = assets[0]
    end_assets = assets[-1]
    total_return = (end_assets / start_assets - 1) if start_assets > 0 else 0
    total_pnl = end_assets - start_assets

    # 最大回撤 (v7.6: 用调整后净值序列, 剔除提现导致的假回撤)
    peak = assets[0]
    max_dd = 0
    max_dd_date = ""
    for i, e in enumerate(equity_curve):
        val = assets[i] if i < len(assets) else e["total_assets"]
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_date = e["date"]

    # 胜率（日度）
    daily_wins = 0
    daily_losses = 0
    for i in range(1, len(assets)):
        if assets[i] > assets[i-1]:
            daily_wins += 1
        elif assets[i] < assets[i-1]:
            daily_losses += 1
    total_days = daily_wins + daily_losses
    win_rate = daily_wins / total_days * 100 if total_days > 0 else 0

    # 盈亏比 — numpy diff + 布尔索引 一次完成
    arr = np.array(assets, dtype=np.float64)
    diffs = np.diff(arr)
    win_diffs = diffs[diffs > 0]
    loss_diffs = -diffs[diffs < 0]
    avg_win = float(np.mean(win_diffs)) if len(win_diffs) > 0 else 0.0
    avg_loss = float(np.mean(loss_diffs)) if len(loss_diffs) > 0 else 0.0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # 现金纪律 — 8/14修复: 原固定700元为旧账户时代(小资金)的拍脑袋阈值,
    # 改为比例口径: 现金 < max(200元, 总资产0.1%) 视为满仓违规(50万账户≈500元, 单笔最小100股成本)
    cash_days = [e for e in equity_curve if "cash" in e]
    cash_below_min = sum(1 for e in cash_days
                         if e["cash"] < max(200.0, e.get("total_assets", 0) * 0.001))
    days_with_cash_data = len(cash_days)

    return {
        "start_date": equity_curve[0]["date"],
        "end_date": equity_curve[-1]["date"],
        "trading_days": len(equity_curve),
        "start_assets": round(start_assets, 2),
        "end_assets": round(end_assets, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_date": max_dd_date,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "cash_violation_days": cash_below_min,
        "cash_violation_pct": round(cash_below_min / days_with_cash_data * 100, 1)
               if days_with_cash_data > 0 else 0,
        "latest_assets": round(assets[-1], 2),
        "latest_cash": round(equity_curve[-1]["cash"], 2)
    }


def compute_score_accuracy(reports):
    """
    v7.2: 重构信号准确性分析 — 使用预测性评估（非同日相关性）

    旧方法（已废弃）: 同日评分变化 vs 同日价格变化 → 相关性而非预测力
    新方法: T日评分变化 → T+1日价格变化 → 真实预测力评估

    同时报告:
    - 方向准确率: 评分升降 → 次日涨跌 的一致性
    - 信号质量: 大幅度评分变化(>5分)的预测准确率
    - 旧指标(参考): 同日相关性（用于对比）
    """
    if len(reports) < 4:
        return {"error": "至少需要4天数据（3个预测样本）"}

    def adjacent_report_days(date_a, date_b):
        """Conservatively accept only near-consecutive sessions (weekend included).

        The report history can have outages. Treating the next available report
        weeks later as a T+1 outcome materially overstates/understates signal
        accuracy. A four-calendar-day ceiling covers ordinary weekends while
        excluding long outages and extended exchange holidays.
        """
        try:
            a = datetime.strptime(str(date_a), "%Y%m%d").date()
            b = datetime.strptime(str(date_b), "%Y%m%d").date()
        except (TypeError, ValueError):
            return False
        gap = (b - a).days
        return 1 <= gap <= 4

    # 按ETF代码聚合
    etf_tracks = {}
    for r in reports:
        scores = r.get("scores", [])
        for s in scores:
            code = s.get("code", "")
            if code not in etf_tracks:
                etf_tracks[code] = []
            etf_tracks[code].append({
                "date": r.get("_date", ""),
                "score": s.get("score", 0),
                "grade": s.get("grade", ""),
                "price": s.get("price", 0)
            })

    results = {}
    for code, track in etf_tracks.items():
        if len(track) < 4:
            continue

        # === 新方法: 预测性评估 ===
        pred_signals = 0      # 总预测次数
        pred_correct = 0      # 方向正确次数
        strong_signals = 0    # 强信号次数 (score变化>5分)
        strong_correct = 0    # 强信号正确次数

        for i in range(1, len(track) - 1):
            if (not adjacent_report_days(track[i-1]["date"], track[i]["date"])
                    or not adjacent_report_days(track[i]["date"], track[i+1]["date"])):
                continue
            # T-1 → T 评分变化（信号）
            ds = track[i]["score"] - track[i-1]["score"]
            # T → T+1 价格变化（结果）
            dp_next = track[i+1]["price"] - track[i]["price"]

            if ds != 0 and dp_next != 0:
                pred_signals += 1
                # 评分升 → 次日涨, 评分降 → 次日跌 = 正确预测
                if (ds > 0 and dp_next > 0) or (ds < 0 and dp_next < 0):
                    pred_correct += 1

                # 强信号（评分变化超过5分）
                if abs(ds) > 5:
                    strong_signals += 1
                    if (ds > 0 and dp_next > 0) or (ds < 0 and dp_next < 0):
                        strong_correct += 1

        pred_accuracy = pred_correct / pred_signals * 100 if pred_signals > 0 else 0
        strong_accuracy = strong_correct / strong_signals * 100 if strong_signals > 0 else 0

        # === 旧方法: 同日相关性（仅供对比） ===
        same_day_signals = 0
        same_day_correct = 0
        for i in range(1, len(track)):
            if not adjacent_report_days(track[i-1]["date"], track[i]["date"]):
                continue
            ds = track[i]["score"] - track[i-1]["score"]
            dp = track[i]["price"] - track[i-1]["price"]
            if ds != 0 and dp != 0:
                same_day_signals += 1
                if (ds > 0 and dp > 0) or (ds < 0 and dp < 0):
                    same_day_correct += 1
        same_day_accuracy = same_day_correct / same_day_signals * 100 if same_day_signals > 0 else 0

        # 找ETF名称
        name = code
        for r in reports:
            for s in r.get("scores", []):
                if s["code"] == code:
                    name = s.get("name", code)
                    break
            if name != code:
                break

        # 判定有效性: 预测准确率>55%为有效, <45%为反向, 中间为随机
        if pred_signals < 5:
            effectiveness = "样本不足"
        elif pred_accuracy >= 58:
            effectiveness = "✅ 有效预测"
        elif pred_accuracy >= 52:
            effectiveness = "🟡 弱预测力"
        elif pred_accuracy >= 45:
            effectiveness = "⚪ 接近随机"
        else:
            effectiveness = "❌ 可能反向"

        results[code] = {
            "name": name,
            "data_points": len(track),
            "pred_signals": pred_signals,
            "pred_accuracy": round(pred_accuracy, 1),
            "strong_signals": strong_signals,
            "strong_accuracy": round(strong_accuracy, 1),
            "same_day_accuracy": round(same_day_accuracy, 1),  # 旧指标(对比用)
            "effectiveness": effectiveness,
            "latest_score": track[-1]["score"],
            "score_trend": "上升" if len(track) >= 3 and track[-1]["score"] > track[-3]["score"]
                          else "下降" if len(track) >= 3 and track[-1]["score"] < track[-3]["score"]
                          else "平稳"
        }

    return results


def generate_performance_summary(report_dir=None):
    """Use the same flow-aware measure as the long-term daily report."""
    from cashflow_performance import account_performance
    report_dir = report_dir or os.path.dirname(os.path.abspath(__file__))
    reports = load_history(report_dir, days=10000)
    try:
        with open(os.path.join(report_dir, "portfolio.json"), encoding="utf-8") as f:
            portfolio = json.load(f)
    except (OSError, ValueError):
        return "账户流水不可用，暂停收益率与回撤结论"
    policy = 'block'
    try:
        with open(os.path.join(report_dir, 'long_term_config.json'), encoding='utf-8') as f:
            policy = json.load(f).get('legacy_placeholder_policy', 'block')
    except (OSError,ValueError):
        pass
    result = account_performance(reports, portfolio, policy)
    if not result["available"]:
        return result["reason"]
    lines = ["账户绩效（剔除外部资金流）",
             f"跟踪区间 {result['start_date']} 至 {result['end_date']}",
             f"区间盈亏 {result['profit']:+.2f}元 | 近似时间加权收益 {result['return_pct']:+.2f}%",
             f"最大回撤 {result['max_drawdown_pct']:.2f}%", result["method_note"]]
    if result["xirr_pct"] is not None:
        lines.append(f"区间XIRR {result['xirr_pct']:+.2f}%（年化）")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print(generate_performance_summary())
