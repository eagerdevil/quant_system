"""Conservative, explainable DCA timing hints for user-reviewed ETF decisions."""

from backtest_engine import COMMISSION_RATE, MIN_COMMISSION, SLIPPAGE_BPS

REFERENCE_MONTHLY_BUDGET = 1000.0


def monthly_budget_cost_note(monthly_budget=REFERENCE_MONTHLY_BUDGET):
    """Estimate configured minimum fee drag for one versus four monthly orders."""
    if monthly_budget <= 0:
        return ""
    slip = monthly_budget * SLIPPAGE_BPS / 10000
    one_cost = max(MIN_COMMISSION, monthly_budget * COMMISSION_RATE) + slip
    four_cost = 4 * max(MIN_COMMISSION, (monthly_budget / 4) * COMMISSION_RATE) + slip
    return (f"参考月投入约{monthly_budget:.0f}元（金额可变）；按系统当前最低佣金{MIN_COMMISSION:.0f}元/笔和滑点假设，"
            f"整月一笔成本约{one_cost:.1f}元（{one_cost/monthly_budget:.2%}），拆四笔约{four_cost:.1f}元"
            f"（{four_cost/monthly_budget:.2%}）。这是系统费率假设，先按你的券商实际费率核对。")


def _num(mapping, key, default=None):
    try:
        value = mapping.get(key)
        if value is None:
            return default
        value = float(value)
        return value if value == value and abs(value) != float("inf") else default
    except (AttributeError, TypeError, ValueError):
        return default


def evaluate_dca_guidance(score, data_is_current=True, history_len=0, regime=None):
    """Separate recurring-investment timing from uncalibrated technical ranking.

    This is an explanatory signal only: it never sizes or submits an order.
    It intentionally does not claim long-term fund suitability because fee,
    tracking-difference, AUM and index-composition data are not available here.
    """
    indicators = score.get("indicators") or {}
    returns = score.get("returns") or {}
    vs_ma = score.get("vs_ma") or {}
    price = _num(score, "price", 0)
    rsi = _num(indicators, "rsi")
    r5 = _num(returns, "r5d")
    r60 = _num(returns, "r60d")
    ma20 = _num(vs_ma, "pct_ma20")
    ma60 = _num(vs_ma, "pct_ma60")
    premium_info = score.get("premium_info") or {}
    premium = _num(premium_info, "premium_pct")
    history = premium_info.get("history") or {}
    premium_pctile = _num(history, "percentile") if history.get("has_history") else None
    premium_source = history.get("current_source") or premium_info.get("data_source") or premium_info.get("premium_source")
    is_qdii = bool(premium_info.get("is_qdii"))
    premium_is_live = premium_source == "iopv"

    result = {
        "action": "数据不足，暂不给时机提示",
        "reasons": [],
        "long_term_fit": "未评级：缺基金费率、跟踪误差、规模及指数成分数据",
        "score_note": "技术面相对评分，不是未来收益概率，也不代表长期质量",
        "data_confidence": "低",
        "budget_context": monthly_budget_cost_note(),
    }
    if not data_is_current:
        result["reasons"].append("行情未确认更新到最近交易日")
        return result
    if price is None or price <= 0 or rsi is None or r5 is None or r60 is None:
        result["reasons"].append("关键行情指标缺失")
        return result
    if history_len < 60:
        result["reasons"].append(f"有效日线仅{history_len}根，历史不足60根")
        return result

    result["data_confidence"] = "中（行情完整；技术时机规则未做收益概率校准）" if history_len >= 120 else "低（历史不足120根）"
    high_premium = (premium is not None and premium >= 5) or (premium_pctile is not None and premium_pctile >= 95)
    elevated_premium = (premium is not None and premium >= 3) or (premium_pctile is not None and premium_pctile >= 90)
    if high_premium and (not is_qdii or premium_is_live):
        result["action"] = "本期先跳过该ETF，等溢价回落后再看"
        if premium is not None and premium >= 5:
            result["reasons"].append(f"实时/当前估值溢价{premium:.1f}%，买入会先承担较高净值偏离成本")
        if premium_pctile is not None and premium_pctile >= 95:
            result["reasons"].append(f"当前溢价处历史{premium_pctile:.0f}%分位")
        return result

    result["action"] = "按你原定计划分批定投；不因单日评分追涨杀跌"
    premium_caution = False
    if is_qdii and premium is None:
        premium_caution = True
        result["action"] = "QDII溢价数据不足；先核对实时IOPV，暂不额外加码"
        result["reasons"].append("QDII溢价数据缺失，需另行核对IOPV/净值偏离")
    elif is_qdii and elevated_premium and not premium_is_live:
        premium_caution = True
        result["action"] = "溢价口径可能滞后；先核对实时IOPV，再决定本期是否买入"
        result["reasons"].append("当前溢价或历史分位基于滞后净值/估算值，不能视为实时拦截信号")
    elif elevated_premium:
        premium_caution = True
        result["action"] = "只考虑原定小额定投，暂不额外加码"
        result["reasons"].append("溢价偏高，避免一次性扩大买入")

    overextended = (rsi >= 70) or (r5 >= 8) or (ma20 is not None and ma20 >= 8)
    if overextended:
        if not premium_caution:
            result["action"] = "基础定投可照常；暂不追涨或额外加码"
        reasons = []
        if rsi >= 70:
            reasons.append(f"RSI {rsi:.0f}偏热")
        if r5 >= 8:
            reasons.append(f"近5日上涨{r5:.1f}%")
        if ma20 is not None and ma20 >= 8:
            reasons.append(f"高于20日均线{ma20:.1f}%")
        result["reasons"].extend(reasons)
    elif (rsi <= 38 and r60 >= 0 and ma60 is not None and ma60 <= -3
          and regime != "CRISIS"):
        if not premium_caution:
            result["action"] = "回撤分批候选；仍按预算拆分，不据此一次性抄底"
        result["reasons"].append("短期偏冷、但60日收益仍为正且价格回到60日均线下方")
    else:
        result["reasons"].append("没有触发明显溢价或短期过热条件；该规则不预测涨跌")

    if regime == "CRISIS":
        result["reasons"].append("市场处于危机状态；系统不擅自暂停基础定投，但不建议据此额外加码")
    return result
