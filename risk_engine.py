#!/usr/bin/env python
"""
量化系统 v3.0 — 投资组合风控引擎
================================
模块: 投资组合层级风险分析
功能: 相关性矩阵 / VaR / CVaR / 集中度检测 / 风险报告

专业量化系统的核心组件——在你亏钱之前告诉你风险有多大。
"""

import math, sys, os
from datetime import datetime

TODAY = datetime.now().strftime("%Y%m%d")


# ============================================================
# 1. 收益率计算
# ============================================================
def daily_returns(closes):
    """计算日收益率序列"""
    if len(closes) < 2:
        return []
    return [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]


def percentile(data, p):
    """计算百分位数（不依赖numpy）"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


# ============================================================
# 2. 相关性矩阵
# ============================================================
def pearson_correlation(x, y):
    """手动计算皮尔逊相关系数"""
    n = len(x)
    if n < 3:
        return 0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((v - mean_x)**2 for v in x))
    std_y = math.sqrt(sum((v - mean_y)**2 for v in y))
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)


def compute_correlation_matrix(etf_data, window=60):
    """
    计算ETF间的滚动相关性矩阵
    输入: etf_data = {code: {"name": str, "kline": [{close,...},...]}}
          window = 滚动窗口天数
    输出: {
        "matrix": {code: {code: corr}},  # 两两相关性
        "avg_correlation": float,         # 平均相关性
        "max_pair": (code1, code2, corr), # 最相关的一对
        "warning_pairs": [(code1, code2, corr)]  # >0.7的高相关对
    }
    """
    # 对齐所有ETF的收益率序列（取最近window天）
    returns_map = {}
    for code, edata in etf_data.items():
        kline = edata.get("kline", [])
        if len(kline) < window + 1:
            continue
        closes = [k["close"] for k in kline[-window-1:]]
        rets = daily_returns(closes)
        if len(rets) >= window - 5:  # 允许少量缺失
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

    # 计算两两相关性
    matrix = {}
    for c in codes:
        matrix[c] = {}  # 预初始化所有inner dict
    corr_values = []
    max_pair = None
    max_corr = -1
    warning_pairs = []

    for i, c1 in enumerate(codes):
        for j, c2 in enumerate(codes):
            if i == j:
                matrix[c1][c2] = 1.0
            elif j > i:
                # 对齐长度
                min_len = min(len(returns_map.get(c1, [])), len(returns_map.get(c2, [])))
                r1 = returns_map[c1][-min_len:]
                r2 = returns_map[c2][-min_len:]
                corr = round(pearson_correlation(r1, r2), 3)
                matrix[c1][c2] = corr
                matrix[c2][c1] = corr  # 对称填充
                corr_values.append(corr)
                if corr > max_corr:
                    max_corr = corr
                    max_pair = (c1, c2, corr)
                if corr > 0.70:
                    name1 = etf_data.get(c1, {}).get("name", c1)
                    name2 = etf_data.get(c2, {}).get("name", c2)
                    warning_pairs.append((c1, c2, corr, name1, name2))

    avg_corr = round(sum(corr_values) / len(corr_values), 3) if corr_values else 0

    return {
        "matrix": matrix,
        "avg_correlation": avg_corr,
        "max_pair": max_pair,
        "warning_pairs": warning_pairs,
        "n_assets": len(codes)
    }


# ============================================================
# 3. VaR / CVaR 计算
# ============================================================
def compute_var_cvar(portfolio_value, etf_data, holdings,
                     confidence=0.95, window=60):
    """
    历史模拟法计算投资组合 VaR 和 CVaR
    输入:
        portfolio_value: 组合总市值
        etf_data: {code: {kline: [...]}}
        holdings: {code: {shares, current_price}}
        confidence: 置信水平 (0.95 = 95% VaR)
        window: 回看窗口
    输出: {
        "var_95": float,    # 95%置信度下日最大亏损(元)
        "var_95_pct": float, # 同上(百分比)
        "cvar_95": float,   # 条件VaR（超过VaR的平均亏损）
        "cvar_95_pct": float,
        "worst_day": float, # 最差单日亏损
        "worst_day_pct": float,
        "method": "historical"
    }
    """
    # 为每个持仓计算日收益率
    asset_returns = {}
    asset_weights = {}
    total_value = portfolio_value if portfolio_value > 0 else 1

    for code, pos in holdings.items():
        if code.startswith("_") or code not in etf_data:
            continue
        kline = etf_data[code].get("kline", [])
        if len(kline) < window + 1:
            continue
        closes = [k["close"] for k in kline[-window-1:]]
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

    # 计算组合历史日收益率（加权）
    min_len = min(len(r) for r in asset_returns.values())
    portfolio_rets = []
    for i in range(-min_len, 0):
        day_ret = 0
        for code, rets in asset_returns.items():
            if abs(i) <= len(rets):
                day_ret += rets[i] * asset_weights.get(code, 0)
        portfolio_rets.append(day_ret)

    if not portfolio_rets:
        return {
            "var_95": 0, "var_95_pct": 0,
            "cvar_95": 0, "cvar_95_pct": 0,
            "worst_day": 0, "worst_day_pct": 0,
            "method": "historical", "error": "empty_returns"
        }

    # 排序找VaR
    sorted_rets = sorted(portfolio_rets)
    var_idx = int(len(sorted_rets) * (1 - confidence))
    var_ret = sorted_rets[var_idx] if var_idx < len(sorted_rets) else sorted_rets[-1]

    # CVaR = VaR之外所有更差收益率的均值
    tail_rets = sorted_rets[:var_idx+1]
    cvar_ret = sum(tail_rets) / len(tail_rets) if tail_rets else var_ret

    # 最差单日
    worst_ret = sorted_rets[0]

    return {
        "var_95": round(abs(var_ret) * portfolio_value, 2),
        "var_95_pct": round(abs(var_ret) * 100, 2),
        "cvar_95": round(abs(cvar_ret) * portfolio_value, 2),
        "cvar_95_pct": round(abs(cvar_ret) * 100, 2),
        "worst_day": round(abs(worst_ret) * portfolio_value, 2),
        "worst_day_pct": round(abs(worst_ret) * 100, 2),
        "method": "historical",
        "confidence": confidence,
        "lookback_days": min_len
    }


# ============================================================
# 4. 集中度风险检测
# ============================================================
def detect_concentration_risk(corr_result, holdings, etf_data):
    """
    检测投资组合集中度风险
    返回: {
        "level": "safe" | "warning" | "danger",
        "issues": [str],
        "diversification_score": 0-100
    }
    """
    issues = []
    score = 100  # 起始满分

    # 检查1: 高相关对
    warning_pairs = corr_result.get("warning_pairs", [])
    if warning_pairs:
        for c1, c2, corr, n1, n2 in warning_pairs:
            issues.append(f"{n1}↔{n2} 相关性{corr:.2f}，同涨同跌风险高")
            score -= 15 * (corr - 0.7) / 0.3  # 0.7→扣0, 1.0→扣15

    # 检查2: 平均相关性
    avg_corr = corr_result.get("avg_correlation", 0)
    if avg_corr > 0.6:
        issues.append(f"组合平均相关性{avg_corr:.2f}偏高，分散化不足")
        score -= 20 * (avg_corr - 0.6) / 0.4
    elif avg_corr > 0.4:
        # 中等相关，正常
        pass
    else:
        # 低相关，加分
        score = min(100, score + 10)

    # 检查3: 持仓数量
    n_assets = corr_result.get("n_assets", 0)
    if n_assets <= 2:
        issues.append("仅2只持仓，系统性风险集中")
        score -= 20
    elif n_assets <= 3:
        score -= 5

    # 检查4: 仓位集中度（单只占比>40%）
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
    """
    生成投资组合风险报告
    输入:
        portfolio: {code: {shares, cost, current_price, name}}
        etf_data: {code: {name, kline, realtime}}
        scores: [{code, score, grade, ...}] 可选
    输出: 完整的风险报告dict
    """
    # 计算组合市值
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

    # 1. 相关性矩阵
    corr_result = compute_correlation_matrix(etf_data)

    # 2. VaR/CVaR
    var_result = compute_var_cvar(total_value, etf_data, holdings_for_var)

    # 3. 集中度风险
    conc_result = detect_concentration_risk(corr_result, holdings_for_var, etf_data)

    # 4. 单只风险拆解
    single_risks = []
    for code, pos in portfolio.items():
        if code.startswith("_") or not isinstance(pos, dict):
            continue
        if code not in etf_data:
            continue

        kline = etf_data[code].get("kline", [])
        if len(kline) < 60:
            continue

        closes = [k["close"] for k in kline[-60:]]
        rets = daily_returns(closes)
        if not rets:
            continue

        vol = math.sqrt(sum((r - sum(rets)/len(rets))**2 for r in rets) / (len(rets)-1)) * math.sqrt(252) if len(rets) > 1 else 0
        ann_ret = (sum(rets)/len(rets)) * 252 if rets else 0
        maxdd = 0
        peak = closes[0]
        for c in closes:
            if c > peak:
                peak = c
            dd = (peak - c) / peak
            if dd > maxdd:
                maxdd = dd

        price = pos.get("current_price", pos.get("cost", 0))
        cost = pos.get("cost", price)
        pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0
        shares = pos.get("shares", 0)
        value = shares * price

        # 找评分
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
    """
    将风险报告格式化为可打印的文本段落
    用于 daily_runner.py 和 quick_analysis.py
    """
    lines = []
    lines.append(f"  {'─'*60}")
    lines.append(f"  [投资组合风险管理]")
    lines.append(f"  {'─'*60}")

    # VaR
    var = report.get("var", {})
    if var and not var.get("error"):
        lines.append(f"  📉 风险价值 (历史模拟法, {var.get('lookback_days', '?')}日窗口):")
        lines.append(f"     VaR(95%):  {var['var_95']:.2f}元 ({var['var_95_pct']:.1f}%) — 95%概率日亏损不超过此值")
        lines.append(f"     CVaR(95%): {var['cvar_95']:.2f}元 ({var['cvar_95_pct']:.1f}%) — 超过VaR时的平均亏损")
        lines.append(f"     历史最差日: {var['worst_day']:.2f}元 ({var['worst_day_pct']:.1f}%)")

    # 相关性
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

    # 集中度
    conc = report.get("concentration", {})
    level_emoji = {"safe": "✅", "warning": "⚠️", "danger": "🚨"}
    emoji = level_emoji.get(conc.get("level", ""), "⚪")
    lines.append(f"  {emoji} 分散化评分: {conc.get('score', '?')}分 ({conc.get('level', '?')})")
    lines.append(f"     {conc.get('advice', '')}")
    if conc.get("issues"):
        for issue in conc["issues"]:
            lines.append(f"     - {issue}")

    # 单只风险拆解
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
    # 简易测试
    print("Risk Engine v3.0 — Ready", file=sys.stderr)
    print("  compute_correlation_matrix()", file=sys.stderr)
    print("  compute_var_cvar()", file=sys.stderr)
    print("  detect_concentration_risk()", file=sys.stderr)
    print("  portfolio_risk_report()", file=sys.stderr)
