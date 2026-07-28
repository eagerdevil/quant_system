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


def load_history(report_dir, days=90):
    """加载历史报告，返回按日期排序的列表"""
    reports = []
    if not os.path.isdir(report_dir):
        return reports

    for fname in sorted(os.listdir(report_dir)):
        if not fname.startswith("report_") or not fname.endswith(".json"):
            continue
        path = os.path.join(report_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            date_str = fname.replace("report_", "").replace(".json", "")
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


def compute_performance_metrics(equity_curve):
    """计算核心绩效指标"""
    if len(equity_curve) < 2:
        return {"error": "数据不足，至少需要2天"}

    assets = [e["total_assets"] for e in equity_curve]
    pnls = [e["total_pnl"] for e in equity_curve]

    # 累计收益
    start_assets = assets[0]
    end_assets = assets[-1]
    total_return = (end_assets / start_assets - 1) if start_assets > 0 else 0
    total_pnl = end_assets - start_assets

    # 最大回撤
    peak = assets[0]
    max_dd = 0
    max_dd_date = ""
    for e in equity_curve:
        if e["total_assets"] > peak:
            peak = e["total_assets"]
        dd = (peak - e["total_assets"]) / peak if peak > 0 else 0
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

    # 现金纪律
    cash_below_700 = sum(1 for e in equity_curve if e["cash"] < 700)
    days_with_cash_data = len([e for e in equity_curve if e["cash"] > 0])

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
        "cash_violation_days": cash_below_700,
        "cash_violation_pct": round(cash_below_700 / days_with_cash_data * 100, 1)
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
    """生成绩效汇总报告"""
    if report_dir is None:
        report_dir = os.path.dirname(os.path.abspath(__file__))

    reports = load_history(report_dir)
    if not reports:
        return "暂无历史报告数据"

    equity = compute_equity_curve(reports)
    metrics = compute_performance_metrics(equity)
    accuracy = compute_score_accuracy(reports)

    lines = []
    lines.append(f"  {'─'*60}")
    lines.append(f"  [绩效追踪] 系统历史表现")
    lines.append(f"  {'─'*60}")

    if "error" in metrics:
        lines.append(f"  ⚠️ {metrics['error']}")
        return "\n".join(lines)

    # 核心指标
    lines.append(f"  📈 跟踪期间: {metrics['start_date']} → {metrics['end_date']} ({metrics['trading_days']}个交易日)")
    lines.append(f"  累计收益: {metrics['total_pnl']:+.2f}元 ({metrics['total_return_pct']:+.2f}%)")
    lines.append(f"  最大回撤: {metrics['max_drawdown_pct']:.1f}% (发生在{metrics['max_drawdown_date']})")
    lines.append(f"  日胜率: {metrics['win_rate']:.0f}% | 盈亏比: {metrics['profit_factor']:.1f}")
    lines.append(f"  期初资产: {metrics['start_assets']:.2f}元 → 期末: {metrics['end_assets']:.2f}元")
    lines.append(f"  🚨 现金纪律违反: {metrics['cash_violation_days']}天 ({metrics['cash_violation_pct']:.0f}%)")
    if metrics['cash_violation_pct'] > 50:
        lines.append(f"     ⚠️ 超过半数交易日现金低于700元！这是账户最大风险来源")

    # 信号准确性 (v7.2: 预测性评估)
    if accuracy and "error" not in accuracy:
        lines.append(f"\n  🎯 信号预测力 (T日评分变化→T+1日涨跌):")
        for code, acc in accuracy.items():
            eff = acc.get("effectiveness", "?")
            strong_str = f" | 强信号:{acc['strong_accuracy']:.0f}%" if acc.get("strong_signals", 0) >= 3 else ""
            old_str = f" | 同日相关:{acc['same_day_accuracy']:.0f}%"
            lines.append(
                f"     {eff} {acc['name']}({code}): "
                f"预测{acc['pred_accuracy']:.0f}% ({acc['pred_signals']}次)"
                f"{strong_str}{old_str}"
                f" | 最新{acc['latest_score']}分 | 趋势:{acc['score_trend']}"
            )

    # 权益曲线（简化的ASCII）
    if len(equity) >= 3:
        lines.append(f"\n  📉 权益走势 (最近{min(len(equity), 20)}天):")
        assets = [e["total_assets"] for e in equity[-20:]]
        min_a, max_a = min(assets), max(assets)
        if max_a > min_a:
            for i, e in enumerate(equity[-20:]):
                bar_len = int((e["total_assets"] - min_a) / (max_a - min_a) * 40) if max_a > min_a else 20
                bar = "█" * bar_len
                lines.append(f"     {e['date']} {e['total_assets']:>7.0f}元 {bar}")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print(generate_performance_summary())
