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


# ============================================================
# v7.3: 历史压力测试场景
# ============================================================

# 定义5个历史极端行情场景（申万一级行业冲击因子）
STRESS_SCENARIOS = [
    {
        "name": "2015股灾 (2015.06-08)",
        "description": "A股泡沫破裂，上证从5178跌至2850，三波股灾",
        "broad_market_shock": -0.35,
        "industry_shocks": {
            "计算机": -0.55, "传媒": -0.52, "电子": -0.48,
            "国防军工": -0.50, "机械设备": -0.45, "电气设备": -0.42,
            "非银金融": -0.40, "银行": -0.20, "食品饮料": -0.25,
            "医药生物": -0.30, "有色金属": -0.30, "公用事业": -0.18,
        },
        "gold_return": 0.05,   # 黄金避险上涨
        "bond_return": 0.03,
    },
    {
        "name": "2016熔断 (2016.01)",
        "description": "熔断机制引发恐慌，全月暴跌25%",
        "broad_market_shock": -0.25,
        "industry_shocks": {
            "计算机": -0.35, "电子": -0.33, "传媒": -0.35,
            "非银金融": -0.30, "银行": -0.08, "食品饮料": -0.15,
        },
        "gold_return": 0.02,
        "bond_return": 0.01,
    },
    {
        "name": "2020疫情冲击 (2020.02.03)",
        "description": "春节后首日，全市场暴跌，3200+跌停",
        "broad_market_shock": -0.08,
        "industry_shocks": {
            "休闲服务": -0.12, "交通运输": -0.10, "传媒": -0.10,
            "房地产": -0.10, "汽车": -0.10, "医药生物": 0.05,  # 医药逆势涨
        },
        "gold_return": 0.01,
        "bond_return": 0.02,
    },
    {
        "name": "2024量化踩踏 (2024.01-02)",
        "description": "雪球敲入+DMA爆仓，小盘股流动性危机",
        "broad_market_shock": -0.12,
        "industry_shocks": {
            "综合": -0.25, "机械设备": -0.22, "计算机": -0.20,
            "电子": -0.18, "电气设备": -0.18, "传媒": -0.22,
            "银行": -0.02, "公用事业": -0.03, "食品饮料": -0.05,
        },
        "gold_return": 0.04,
        "bond_return": 0.03,
    },
    {
        "name": "2026.06近期暴跌",
        "description": "6/26上证-2.14%深证-3.04%，全市场无差别杀跌",
        "broad_market_shock": -0.06,
        "industry_shocks": {
            "电气设备": -0.07, "汽车": -0.08, "电子": -0.07,
            "有色金属": -0.06, "医药生物": -0.05, "国防军工": -0.06,
        },
        "gold_return": 0.01,
        "bond_return": 0.01,
    },
]

# v7.3: 从权威来源导入ETF→行业映射（单一数据源，避免发散）
try:
    from quant_engine import ETF_INDUSTRY_MAP as _ETF_INDUSTRY
except ImportError:
    _ETF_INDUSTRY = {}  # 导入失败时回退空映射


def stress_test_portfolio(portfolio, port_summary=None):
    """
    v7.3: 历史压力测试 — 将当前持仓放入5个历史极端行情中模拟。

    返回: {
        "scenarios": [{name, description, total_loss, loss_pct, per_holding, verdict}],
        "worst_scenario": {name, loss_pct},
        "average_loss_pct": float,
        "advice": str
    }
    """
    total_assets = port_summary.get("total_assets", 4000) if port_summary else 4000
    holdings = port_summary.get("holdings", []) if port_summary else []

    # 构建持仓快照: {code: {weight_pct, name}}
    snapshot = {}
    for h in holdings:
        code = h.get("code", "")
        if code and h.get("weight", 0) > 0:
            snapshot[code] = {"weight": h["weight"], "name": h.get("name", code)}

    if not snapshot:
        # 从portfolio原始数据构建
        for code, pos in portfolio.items():
            if code.startswith("_") or not isinstance(pos, dict):
                continue
            price = pos.get("current_price", pos.get("cost", 0))
            shares = pos.get("shares", 0)
            value = shares * price
            snapshot[code] = {
                "weight": round(value / total_assets * 100, 1) if total_assets > 0 else 0,
                "name": pos.get("name", code)
            }

    results = []
    total_losses = []

    for scenario in STRESS_SCENARIOS:
        total_shock = 0.0
        per_holding = []

        for code, info in snapshot.items():
            industries = _ETF_INDUSTRY.get(code, ["综合"])
            # 取该ETF所有行业的平均冲击
            shocks = []
            for ind in industries:
                s = scenario["industry_shocks"].get(ind, scenario["broad_market_shock"])
                shocks.append(s)
            avg_shock = sum(shocks) / len(shocks) if shocks else scenario["broad_market_shock"]

            # 黄金ETF特殊处理
            if code == "518850":
                avg_shock = scenario["gold_return"]
            # QDII ETF (15xxxx/513xxx): 冲击减半（跟踪海外市场）
            if code.startswith(("159659", "513", "15966")):
                avg_shock = avg_shock * 0.4

            holding_loss = info["weight"] * avg_shock
            total_shock += holding_loss
            per_holding.append({
                "code": code, "name": info["name"],
                "weight": info["weight"],
                "shock_pct": round(avg_shock * 100, 1),
                "loss_contribution": round(holding_loss, 2)
            })

        loss_pct = round(total_shock, 2)
        loss_amount = round(total_assets * total_shock / 100, 2)
        total_losses.append(loss_pct)

        # 判定等级
        if loss_pct > -3:
            verdict = "✅ 抗压良好"
        elif loss_pct > -7:
            verdict = "🟡 中度冲击"
        elif loss_pct > -12:
            verdict = "🟠 显著损失"
        else:
            verdict = "🔴 严重冲击"

        results.append({
            "name": scenario["name"],
            "description": scenario["description"],
            "loss_pct": loss_pct,
            "loss_amount": loss_amount,
            "per_holding": per_holding,
            "verdict": verdict
        })

    worst = min(results, key=lambda r: r["loss_pct"]) if results else None
    avg_loss = round(sum(total_losses) / len(total_losses), 2) if total_losses else 0

    if avg_loss > -4:
        advice = "✅ 组合抗压能力强，5个极端场景平均损失可控"
    elif avg_loss > -8:
        advice = "🟡 中度脆弱，建议增加现金或防御性ETF（红利低波/公用事业）比例"
    elif avg_loss > -12:
        advice = "⚠️ 较脆弱，极端行情可能损失超过10%，强烈建议增加现金比例"
    else:
        advice = "🔴 高度脆弱，当前组合在历史上多次极端行情中都会遭受重创"

    return {
        "scenarios": results,
        "worst_scenario": {"name": worst["name"], "loss_pct": worst["loss_pct"]} if worst else None,
        "average_loss_pct": avg_loss,
        "advice": advice
    }


def format_stress_test_section(stress_result):
    """格式化压力测试为文本板块"""
    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [历史压力测试] v7.3 — 5个极端场景回放")
    lines.append(f"  {'─'*60}")

    for sc in stress_result["scenarios"]:
        bar_len = max(1, int(abs(sc["loss_pct"]) * 2))
        bar = "█" * bar_len
        lines.append(f"  {sc['verdict']} {sc['name']}: {sc['loss_pct']:+.1f}% ({sc['loss_amount']:+.0f}元) {bar}")
        lines.append(f"     {sc['description']}")
        ph = sorted(sc["per_holding"], key=lambda x: x["loss_contribution"])[:3]
        ph_str = " | ".join(f"{p['name']}({p['shock_pct']:+.1f}%)" for p in ph if p["loss_contribution"] < -0.5)
        if ph_str:
            lines.append(f"     最大冲击: {ph_str}")

    lines.append(f"\n  平均损失: {stress_result['average_loss_pct']:+.1f}% | "
                f"最差场景: {stress_result['worst_scenario']['name']} ({stress_result['worst_scenario']['loss_pct']:+.1f}%)")
    lines.append(f"  {stress_result['advice']}")
    return "\n".join(lines)


# ============================================================
# v8.0 P2-9: 蒙特卡洛压力测试
# ============================================================
def monte_carlo_simulation(etf_data, portfolio, n_simulations=1000,
                            horizon_days=20, confidence_levels=(95, 99)):
    """
    蒙特卡洛模拟：从历史收益率中随机抽样，生成N条未来路径。

    与硬编码历史场景不同，MC模拟能捕捉历史中未出现的组合冲击.
    使用Bootstrap方法（有放回抽样）保持收益率之间的相关性结构。

    Args:
        etf_data: {code: {kline: [{close,...},...]}}
        portfolio: {code: {shares, cost, current_price, name}, _available_cash: ...}
        n_simulations: 模拟路径数 (默认1000)
        horizon_days: 预测周期天数 (默认20个交易日≈1个月)
        confidence_levels: VaR置信水平

    Returns: {
        "var_95": -82.5,       # 95%置信度下最差日损失(元)
        "var_99": -145.3,      # 99%置信度下最差日损失(元)
        "cvar_95": -105.2,     # 95%尾部期望损失
        "max_loss": -210.5,    # 模拟中最差情况
        "mean_return": 0.012,  # 平均收益
        "prob_profit": 0.62,   # 盈利概率
        "prob_loss_gt_5pct": 0.08,  # 损失>5%的概率
        "paths_sample": [...], # 10条典型路径(用于可视化)
        "n_simulations": 1000,
        "horizon_days": 20
    }
    """
    import numpy as np

    holdings_list = []
    for code, pos in portfolio.items():
        if code.startswith("_"):
            continue
        if not isinstance(pos, dict) or "shares" not in pos:
            continue
        price = pos.get("current_price", pos.get("cost", 0))
        if price <= 0:
            continue
        value = pos["shares"] * price
        holdings_list.append({
            "code": code, "name": pos.get("name", code),
            "shares": pos["shares"], "price": price, "value": value
        })

    if not holdings_list:
        return {"error": "无有效持仓", "n_simulations": n_simulations}

    total_value = sum(h["value"] for h in holdings_list)
    cash = portfolio.get("_available_cash", 0)
    total_assets = total_value + cash

    # 提取历史日收益率
    etf_returns = {}
    min_len = float("inf")
    for h in holdings_list:
        kline = etf_data.get(h["code"], {}).get("kline", [])
        if len(kline) < 30:
            continue
        closes = np.array([k["close"] for k in kline], dtype=np.float64)
        rets = np.diff(closes) / closes[:-1]
        etf_returns[h["code"]] = rets
        min_len = min(min_len, len(rets))

    if not etf_returns:
        return {"error": "K线数据不足", "n_simulations": n_simulations}

    # 对齐到最短长度
    min_len = min(min_len, min(len(r) for r in etf_returns.values()))
    aligned_rets = {}
    for code, rets in etf_returns.items():
        aligned_rets[code] = rets[-min_len:]

    codes = list(aligned_rets.keys())
    weights = np.array([h["value"] / total_value if total_value > 0 else 0
                        for h in holdings_list if h["code"] in codes])

    # 构建联合收益率矩阵 (n_days × n_etfs)
    ret_matrix = np.column_stack([aligned_rets[c] for c in codes])
    n_days = len(ret_matrix)

    # 蒙特卡洛模拟
    np.random.seed(42)  # 可复现
    sim_end_values = np.zeros(n_simulations)

    # 保存10条典型路径
    sample_paths = []
    sample_indices = np.linspace(0, n_simulations - 1, min(10, n_simulations), dtype=int)

    for sim in range(n_simulations):
        # Bootstrap: 有放回地随机抽取 horizon_days 个交易日
        boot_indices = np.random.randint(0, n_days, size=horizon_days)
        # 累计收益
        port_returns = np.sum(ret_matrix[boot_indices] * weights, axis=1)
        cumulative = np.prod(1 + port_returns)
        sim_end_values[sim] = total_value * cumulative

        if sim in sample_indices:
            path = [total_value]
            for r in port_returns:
                path.append(path[-1] * (1 + r))
            sample_paths.append(path)

    # 统计
    sim_returns = (sim_end_values - total_value) / total_value

    var_95 = float(np.percentile(sim_returns, 100 - confidence_levels[0])) * total_value
    var_99 = float(np.percentile(sim_returns, 100 - confidence_levels[1])) * total_value

    cvar_cutoff = np.percentile(sim_returns, 5)
    tail_returns = sim_returns[sim_returns <= cvar_cutoff]
    cvar_95 = float(np.mean(tail_returns)) * total_value if len(tail_returns) > 0 else var_95

    max_loss = float(np.min(sim_returns)) * total_value
    mean_return = float(np.mean(sim_returns))
    prob_profit = float(np.mean(sim_returns > 0))
    prob_loss_gt_5pct = float(np.mean(sim_returns < -0.05))

    return {
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "cvar_95": round(cvar_95, 2),
        "max_loss": round(max_loss, 2),
        "mean_return": round(mean_return, 4),
        "prob_profit": round(prob_profit, 3),
        "prob_loss_gt_5pct": round(prob_loss_gt_5pct, 3),
        "paths_sample": sample_paths,
        "var_95_pct": round(var_95 / total_assets * 100, 2) if total_assets > 0 else 0,
        "var_99_pct": round(var_99 / total_assets * 100, 2) if total_assets > 0 else 0,
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "total_assets": total_assets
    }


def format_monte_carlo_section(mc_result):
    """格式化蒙特卡洛结果"""
    if mc_result.get("error"):
        return f"\n  [蒙特卡洛模拟] 跳过: {mc_result['error']}"

    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [蒙特卡洛压力测试] v8.0 — {mc_result['n_simulations']}条路径×{mc_result['horizon_days']}天")
    lines.append(f"  {'─'*60}")
    lines.append(f"  VaR(95%): {mc_result['var_95']:+.0f}元 ({mc_result['var_95_pct']:+.1f}%) | "
                f"VaR(99%): {mc_result['var_99']:+.0f}元 ({mc_result['var_99_pct']:+.1f}%)")
    lines.append(f"  CVaR(95%): {mc_result['cvar_95']:+.0f}元 | 最差: {mc_result['max_loss']:+.0f}元")
    lines.append(f"  盈利概率: {mc_result['prob_profit']:.0%} | 损失>5%概率: {mc_result['prob_loss_gt_5pct']:.0%} | "
                f"平均收益: {mc_result['mean_return']:+.2%}")

    # 解读
    if mc_result.get("var_95_pct", 0) > -5:
        lines.append(f"  ✅ 风险可控: 95%概率下{mc_result['horizon_days']}天内损失不超过{abs(mc_result['var_95_pct']):.1f}%")
    elif mc_result.get("var_95_pct", 0) > -10:
        lines.append(f"  🟡 中度风险: 95%概率下损失不超过{abs(mc_result['var_95_pct']):.1f}%")
    else:
        lines.append(f"  ⚠️ 高风险: 95%概率下损失可能超过{abs(mc_result['var_95_pct']):.1f}%")

    if mc_result["prob_loss_gt_5pct"] > 0.20:
        lines.append(f"  ⚠️ 损失>5%的概率达{mc_result['prob_loss_gt_5pct']:.0%}，建议降低仓位")
    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Risk Engine v6.0 — Ready")
    logger.info("  compute_correlation_matrix()")
    logger.info("  compute_var_cvar()")
    logger.info("  detect_concentration_risk()")
    logger.info("  portfolio_risk_report()")
