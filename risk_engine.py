#!/usr/bin/env python
"""
量化系统 v6.0 — 投资组合风控引擎 (pandas+numpy 向量化)
========================================================
模块: 投资组合层级风险分析
功能: 相关性矩阵 / VaR / CVaR / 集中度检测 / 风险报告

v6.0: 相关性矩阵、VaR/CVaR 计算全部迁移到 numpy/pandas 向量化
"""
import math, sys, os, logging

logger = logging.getLogger(__name__)
from datetime import datetime
import numpy as np
import pandas as pd

TODAY = datetime.now().strftime("%Y%m%d")


# ============================================================
# 1. 收益率计算 — numpy 向量化
# ============================================================
def daily_returns(closes):
    """计算日收益率序列 — numpy向量化"""
    if len(closes) < 2:
        return []
    arr = np.array(closes, dtype=np.float64)
    return ((arr[1:] - arr[:-1]) / arr[:-1]).tolist()


def percentile(data, p):
    """计算百分位数 — numpy向量化"""
    if not data:
        return 0
    return float(np.percentile(data, p))


# ============================================================
# 2. 相关性矩阵 — numpy 向量化
# ============================================================
def pearson_correlation(x, y):
    """手动计算皮尔逊相关系数 — numpy向量化"""
    n = len(x)
    if n < 3:
        return 0
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_correlation_matrix(etf_data, window=60):
    """
    计算ETF间的滚动相关性矩阵 — numpy/pandas 向量化
    输入: etf_data = {code: {"name": str, "kline": [{close,...},...]}}
    输出: {matrix, avg_correlation, max_pair, warning_pairs, n_assets}
    """
    # 提取所有ETF的收益率，构建DataFrame
    returns_map = {}
    for code, edata in etf_data.items():
        kline = edata.get("kline", [])
        if len(kline) < window + 1:
            continue
        closes = [k["close"] for k in kline[-window - 1:]]
        rets = daily_returns(closes)
        if len(rets) >= window - 5:
            returns_map[code] = rets

    codes = list(returns_map.keys())
    if len(codes) < 2:
        return {
            "matrix": {},
            "avg_correlation": 0,
            "max_pair": None,
            "warning_pairs": [],
            "n_assets": len(codes)
        }

    # 对齐到最短长度
    min_len = min(len(v) for v in returns_map.values())
    # 构建 numpy 矩阵: rows=时间, cols=ETF
    ret_matrix = np.column_stack([
        np.array(returns_map[code][-min_len:], dtype=np.float64)
        for code in codes
    ])
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(ret_matrix, rowvar=False)

    # 构建输出
    matrix = {c: {} for c in codes}
    corr_values = []
    max_pair = None
    max_corr = -1
    warning_pairs = []

    for i, c1 in enumerate(codes):
        for j, c2 in enumerate(codes):
            if i == j:
                matrix[c1][c2] = 1.0
            elif j > i:
                corr = round(float(corr_matrix[i, j]), 3)
                matrix[c1][c2] = corr
                matrix[c2][c1] = corr
                corr_values.append(corr)
                if corr > max_corr:
                    max_corr = corr
                    max_pair = (c1, c2, corr)
                if corr > 0.70:
                    name1 = etf_data.get(c1, {}).get("name", c1)
                    name2 = etf_data.get(c2, {}).get("name", c2)
                    warning_pairs.append((c1, c2, corr, name1, name2))

    avg_corr = round(float(np.mean(corr_values)), 3) if corr_values else 0

    return {
        "matrix": matrix,
        "avg_correlation": avg_corr,
        "max_pair": max_pair,
        "warning_pairs": warning_pairs,
        "n_assets": len(codes)
    }


# ============================================================
# 3. VaR / CVaR 计算 — numpy 向量化
# ============================================================
def compute_var_cvar(portfolio_value, etf_data, holdings,
                     confidence=0.95, window=60):
    """
    历史模拟法计算投资组合 VaR 和 CVaR — numpy/pandas 向量化
    """
    asset_returns = {}
    asset_weights = {}
    total_value = portfolio_value if portfolio_value > 0 else 1

    for code, pos in holdings.items():
        if code.startswith("_") or code not in etf_data:
            continue
        kline = etf_data[code].get("kline", [])
        if len(kline) < window + 1:
            continue
        closes = [k["close"] for k in kline[-window - 1:]]
        rets = daily_returns(closes)
        if len(rets) >= window - 5:
            asset_returns[code] = rets
            weight = (pos.get("shares", 0) * pos.get("current_price", 0)) / total_value
            asset_weights[code] = weight

    if not asset_returns:
        return {
            "var_95": 0, "var_95_pct": 0,
            "cvar_95": 0, "cvar_95_pct": 0,
            "worst_day": 0, "worst_day_pct": 0,
            "method": "historical", "error": "insufficient_data"
        }

    # 对齐并构建加权组合收益率
    codes = list(asset_returns.keys())
    min_len = min(len(asset_returns[code]) for code in codes)
    weights_arr = np.array([asset_weights.get(code, 0) for code in codes], dtype=np.float64)
    ret_matrix = np.column_stack([
        np.array(asset_returns[code][-min_len:], dtype=np.float64)
        for code in codes
    ])
    portfolio_rets = np.dot(ret_matrix, weights_arr)

    if len(portfolio_rets) == 0:
        return {
            "var_95": 0, "var_95_pct": 0,
            "cvar_95": 0, "cvar_95_pct": 0,
            "worst_day": 0, "worst_day_pct": 0,
            "method": "historical", "error": "empty_returns"
        }

    sorted_rets = np.sort(portfolio_rets)
    var_idx = int(len(sorted_rets) * (1 - confidence))
    var_ret = sorted_rets[var_idx] if var_idx < len(sorted_rets) else sorted_rets[-1]

    tail_rets = sorted_rets[:var_idx + 1]
    cvar_ret = np.mean(tail_rets) if len(tail_rets) > 0 else var_ret

    worst_ret = sorted_rets[0]

    return {
        "var_95": round(abs(float(var_ret)) * portfolio_value, 2),
        "var_95_pct": round(abs(float(var_ret)) * 100, 2),
        "cvar_95": round(abs(float(cvar_ret)) * portfolio_value, 2),
        "cvar_95_pct": round(abs(float(cvar_ret)) * 100, 2),
        "worst_day": round(abs(float(worst_ret)) * portfolio_value, 2),
        "worst_day_pct": round(abs(float(worst_ret)) * 100, 2),
        "method": "historical",
        "confidence": confidence,
        "lookback_days": min_len
    }


# ============================================================
# 4. 集中度风险检测
# ============================================================
def detect_concentration_risk(corr_result, holdings, etf_data):
    """
    检测投资组合集中度风险 — 量化逻辑不变，保持原有输出
    """
    issues = []
    score = 100

    # 检查1: 高相关对
    warning_pairs = corr_result.get("warning_pairs", [])
    if warning_pairs:
        for c1, c2, corr, n1, n2 in warning_pairs:
            issues.append(f"{n1}↔{n2} 相关性{corr:.2f}，同涨同跌风险高")
            score -= 15 * (corr - 0.7) / 0.3

    # 检查2: 平均相关性
    avg_corr = corr_result.get("avg_correlation", 0)
    if avg_corr > 0.6:
        issues.append(f"组合平均相关性{avg_corr:.2f}偏高，分散化不足")
        score -= 20 * (avg_corr - 0.6) / 0.4
    elif avg_corr > 0.4:
        pass
    else:
        score = min(100, score + 10)

    # 检查3: 持仓数量
    n_assets = corr_result.get("n_assets", 0)
    if n_assets <= 2:
        issues.append("仅2只持仓，系统性风险集中")
        score -= 20
    elif n_assets <= 3:
        score -= 5

    # 检查4: 仓位集中度
    total_value = sum(
        h.get("shares", 0) * h.get("current_price", h.get("cost", 0))
        for code, h in holdings.items()
        if not code.startswith("_") and isinstance(h, dict)
    )
    for code, h in holdings.items():
        if code.startswith("_") or not isinstance(h, dict):
            continue
        value = h.get("shares", 0) * h.get("current_price", h.get("cost", 0))
        weight = value / total_value * 100 if total_value > 0 else 0
        if weight > 40:
            name = etf_data.get(code, {}).get("name", code)
            issues.append(f"{name}占比{weight:.0f}%，过度集中")
            score -= 15

    score = max(0, min(100, round(score)))

    if score >= 75:
        level = "safe"
    elif score >= 50:
        level = "warning"
    else:
        level = "danger"

    return {
        "level": level,
        "score": score,
        "issues": issues,
        "advice": _diversification_advice(level, score, issues)
    }


def _diversification_advice(level, score, issues):
    if level == "safe":
        return "组合分散化良好，各持仓间风险抵消效果较好"
    elif level == "warning":
        return f"分散化评分{score}分，建议关注高相关持仓，避免同方向加仓"
    else:
        return f"分散化严重不足({score}分)，多只持仓面临同跌风险，建议减仓高相关品种"


# ============================================================
# 5. 一站式风险报告
# ============================================================
def portfolio_risk_report(portfolio, etf_data, scores=None):
    """生成投资组合风险报告 — 核心逻辑不变"""
    holdings_for_var = {}
    total_value = 0
    for code, pos in portfolio.items():
        if code.startswith("_") or not isinstance(pos, dict):
            continue
        price = pos.get("current_price", pos.get("cost", 0))
        shares = pos.get("shares", 0)
        value = shares * price
        total_value += value
        holdings_for_var[code] = {"shares": shares, "current_price": price}

    corr_result = compute_correlation_matrix(etf_data)
    var_result = compute_var_cvar(total_value, etf_data, holdings_for_var)
    conc_result = detect_concentration_risk(corr_result, holdings_for_var, etf_data)

    # 单只风险拆解 — numpy向量化
    single_risks = []
    for code, pos in portfolio.items():
        if code.startswith("_") or not isinstance(pos, dict):
            continue
        if code not in etf_data:
            continue

        kline = etf_data[code].get("kline", [])
        if len(kline) < 60:
            continue

        closes = np.array([k["close"] for k in kline[-60:]], dtype=np.float64)
        rets = np.diff(closes) / closes[:-1]
        if len(rets) == 0:
            continue

        vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
        ann_ret = float(np.mean(rets) * 252) if len(rets) > 0 else 0.0

        # max drawdown
        peak = np.maximum.accumulate(closes)
        dd = (peak - closes) / peak
        maxdd = float(np.max(dd))

        price = pos.get("current_price", pos.get("cost", 0))
        cost = pos.get("cost", price)
        pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0
        shares = pos.get("shares", 0)
        value = shares * price

        grade = None
        if scores:
            s = next((s for s in scores if s["code"] == code), None)
            if s:
                grade = s.get("grade")

        single_risks.append({
            "code": code,
            "name": pos.get("name", code),
            "value": round(value, 2),
            "weight_pct": round(value / total_value * 100, 1) if total_value > 0 else 0,
            "annual_vol": round(vol * 100, 1),
            "annual_ret": round(ann_ret * 100, 1),
            "max_drawdown_60d": round(maxdd * 100, 1),
            "pnl_pct": round(pnl_pct, 2),
            "grade": grade
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": round(total_value, 2),
        "n_holdings": len(single_risks),
        "correlation": corr_result,
        "var": var_result,
        "concentration": conc_result,
        "single_risks": single_risks
    }


def format_risk_section(report):
    """将风险报告格式化为可打印的文本段落 — 保持原样"""
    lines = []
    lines.append(f"  {'─'*60}")
    lines.append(f"  [投资组合风险管理]")
    lines.append(f"  {'─'*60}")

    var = report.get("var", {})
    if var and not var.get("error"):
        lines.append(f"  📉 风险价值 (历史模拟法, {var.get('lookback_days', '?')}日窗口):")
        lines.append(f"     VaR(95%):  {var['var_95']:.2f}元 ({var['var_95_pct']:.1f}%) — 95%概率日亏损不超过此值")
        lines.append(f"     CVaR(95%): {var['cvar_95']:.2f}元 ({var['cvar_95_pct']:.1f}%) — 超过VaR时的平均亏损")
        lines.append(f"     历史最差日: {var['worst_day']:.2f}元 ({var['worst_day_pct']:.1f}%)")

    corr = report.get("correlation", {})
    if corr and corr.get("n_assets", 0) >= 2:
        lines.append(f"  🔗 持仓相关性 (60日):")
        lines.append(f"     平均相关性: {corr['avg_correlation']:.3f}")
        if corr.get("max_pair"):
            c1, c2, maxc = corr["max_pair"]
            lines.append(f"     最相关一对: {c1}↔{c2} (r={maxc:.3f})")
        if corr.get("warning_pairs"):
            lines.append(f"     ⚠️ 高相关警报 (r>0.7):")
            for c1, c2, r, n1, n2 in corr["warning_pairs"]:
                lines.append(f"       {n1} ↔ {n2}: r={r:.3f}")

    conc = report.get("concentration", {})
    level_emoji = {"safe": "✅", "warning": "⚠️", "danger": "🚨"}
    emoji = level_emoji.get(conc.get("level", ""), "⚪")
    lines.append(f"  {emoji} 分散化评分: {conc.get('score', '?')}分 ({conc.get('level', '?')})")
    lines.append(f"     {conc.get('advice', '')}")
    if conc.get("issues"):
        for issue in conc["issues"]:
            lines.append(f"     - {issue}")

    risks = report.get("single_risks", [])
    if risks:
        lines.append(f"  📊 单只风险拆解:")
        lines.append(f"     {'名称':<14} {'权重':>5} {'年化波动':>8} {'年化收益':>8} {'60日最大回撤':>10} {'浮亏':>8} {'评分'}")
        for r in risks:
            lines.append(
                f"     {r['name']:<14} {r['weight_pct']:>4.1f}% "
                f"{r['annual_vol']:>7.1f}% {r['annual_ret']:>+7.1f}% "
                f"{r['max_drawdown_60d']:>9.1f}% {r['pnl_pct']:>+7.2f}% "
                f"{r.get('grade', '?'):>4s}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Risk Engine v6.0 — Ready")
    logger.info("  compute_correlation_matrix()")
    logger.info("  compute_var_cvar()")
    logger.info("  detect_concentration_risk()")
    logger.info("  portfolio_risk_report()")
