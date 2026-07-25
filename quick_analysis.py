#!/usr/bin/env python
"""
快速持仓分析 - 今日专用
只分析用户4只持仓 + 大盘择时，不做全市场扫描
"""
import json, sys, os, io, math
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_SYSTEM_DIR = os.path.join(SCRIPT_DIR, "quant_system")
sys.path.insert(0, QUANT_SYSTEM_DIR)

from data_engine import (
    fetch_etf_kline, fetch_etf_realtime, fetch_index_daily,
    fetch_market_breadth, fetch_total_volume, fetch_north_bound_flow,
    fetch_margin_balance, fetch_bond_yield, INDEX_CODES,
    fetch_etf_fund_nav, compute_etf_premium, QDII_ETF_CODES
)
from quant_engine import score_etf_comprehensive, MarketTiming, \
    compute_atr_stop_loss, classify_market_regime, MARKET_REGIME
from risk_engine import portfolio_risk_report, format_risk_section

TODAY = datetime.now().strftime("%Y%m%d")

# ============================================================
# 用户持仓
# ============================================================
PORTFOLIO = {
    "518850": {"shares": 200, "cost": 9.091, "name": "黄金ETF华夏"},
    "159183": {"shares": 1000, "cost": 0.985, "name": "新能源车ETF招商"},
    "159659": {"shares": 300, "cost": 2.343, "name": "纳斯达克100ETF招商"},
    "562500": {"shares": 500, "cost": 1.147, "name": "机器人ETF华夏"},
}
AVAILABLE_CASH = 84.64

print("=" * 70)
print(f"  🔬 持仓量化诊断 — {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
print("=" * 70)

# ============================================================
# Step 1: 大盘数据
# ============================================================
print("\n📡 拉取大盘数据...", file=sys.stderr)

# 拉沪深300 K线（用于择时）
hs300_data = fetch_index_daily("000300", 120)
breadth = fetch_market_breadth()
total_vol = fetch_total_volume()
north_flow = fetch_north_bound_flow(10)
margin = fetch_margin_balance()
bond = fetch_bond_yield()

# 拉上证指数、深证、创业板实时行情
indices = {}
for code in ["000001", "399001", "399006", "000300", "000688"]:
    data = fetch_index_daily(code, 5)
    if data:
        indices[code] = {"name": INDEX_CODES.get(code, code), "data": data}

print("✅ 大盘数据拉取完成\n", file=sys.stderr)

# ============================================================
# Step 2: ETF K线 + 实时行情
# ============================================================
print("📡 拉取持仓ETF数据...", file=sys.stderr)

etf_data = {}
nav_data = {}  # 溢价数据
for code, info in PORTFOLIO.items():
    print(f"  -> {code} {info['name']}...", file=sys.stderr)
    kline = fetch_etf_kline(code, 250)
    rt = fetch_etf_realtime(code)
    # 获取基金净值 + 计算溢价率
    fund_nav = fetch_etf_fund_nav(code)
    premium_info = compute_etf_premium(code, rt["price"] if rt else info["cost"], fund_nav)
    if kline:
        etf_data[code] = {"name": info["name"], "kline": kline, "realtime": rt, "premium_info": premium_info}
        nav_data[code] = premium_info
        nav_str = f"溢价{premium_info['premium_pct']:+.2f}%" if premium_info.get('premium_pct') is not None else "无数据"
        qdii_flag = "🌐QDII " if premium_info.get('is_qdii') else ""
        print(f"     OK: {len(kline)}根K线, 实时价={rt['price'] if rt else 'N/A'}, {qdii_flag}{nav_str}", file=sys.stderr)
    else:
        print(f"     FAIL: 无法获取数据", file=sys.stderr)

print("✅ ETF数据拉取完成\n", file=sys.stderr)

# ============================================================
# Step 3: 大盘行情概览
# ============================================================
print()
print("─" * 60)
print("  📊 [零] 今日大盘行情")
print("─" * 60)

for code in ["000001", "399001", "399006", "000300", "000688"]:
    if code in indices:
        d = indices[code]["data"]
        if len(d) >= 2:
            today_close = d[-1]["close"]
            yesterday_close = d[-2]["close"]
            chg = (today_close / yesterday_close - 1) * 100
            print(f"  {indices[code]['name']}({code}): {today_close:.2f}  {chg:+.2f}%")
        elif d:
            print(f"  {indices[code]['name']}({code}): {d[-1]['close']:.2f}")

if total_vol:
    print(f"  全市场成交额: {total_vol:.0f}亿元")
if breadth:
    print(f"  涨停: {breadth.get('limit_up', 'N/A')}家 | 跌停: {breadth.get('limit_down', 'N/A')}家")
    print(f"  上涨: {breadth.get('up_count', 'N/A')}家 | 下跌: {breadth.get('down_count', 'N/A')}家")
if bond:
    print(f"  10年国债收益率: {bond.get('yield', 0)*100:.2f}%")
if north_flow:
    nf_today = north_flow[-1] if north_flow else {}
    print(f"  北向资金今日净流入: {nf_today.get('net_flow', 0):+.1f}亿元")

# ============================================================
# Step 4: 量化评分
# ============================================================
print()
print("─" * 60)
print("  🔬 [一] 持仓ETF 15因子量化评分")
print("─" * 60)

scores = []
for code, edata in etf_data.items():
    kline = edata["kline"]
    closes = [k["close"] for k in kline]
    highs = [k["high"] for k in kline]
    lows = [k["low"] for k in kline]
    volumes = [k["volume"] for k in kline]

    # 传入溢价率（v2.1 新增）
    premium_pct = edata.get("premium_info", {}).get("premium_pct")
    result = score_etf_comprehensive(code, edata["name"], closes, highs, lows, volumes,
                                     premium_pct=premium_pct)
    scores.append(result)

    # 打印单只详情
    s = result
    ind = s["indicators"]
    ret = s["returns"]
    vs = s["vs_ma"]
    f = s["factors"]

    grade_emoji = {"A_强烈买入": "🟢", "B_买入": "🔵", "C_观察": "🟡", "D_谨慎": "🟠", "E_回避": "🔴"}
    emoji = grade_emoji.get(s["grade"], "⚪")

    pos = PORTFOLIO.get(code, {})
    cost = pos.get("cost", s["price"])
    shares = pos.get("shares", 0)
    pnl = (s["price"] / cost - 1) * 100 if cost > 0 else 0
    pnl_amount = (s["price"] - cost) * shares

    # 溢价调整信息（v2.1 新增）
    premium_info = s.get("premium_info", {})
    premium_tag = ""
    if premium_info.get("premium_pct") is not None and premium_info["premium_pct"] > 2:
        penalty = premium_info.get("score_penalty", 0)
        tech = s.get("technical_score", s["score"])
        premium_tag = f" | 原始技术分:{tech} 溢价扣{penalty}分→{s['score']}"

    print(f"  {emoji} {s['name']}({code}) | {s['score']}分 | {s['grade']}{premium_tag}")
    # 溢价警告（v2.1 新增）
    if premium_info.get("warning"):
        print(f"     {premium_info['warning']}")
    print(f"     现价: {s['price']:.4f} | 成本: {cost:.4f} | 浮亏: {pnl_amount:+.2f}元 ({pnl:+.2f}%)")
    print(f"     RSI(14): {ind['rsi']:.0f} | 波动率: {ind['volatility_pct']:.1f}% | 夏普: {ind['sharpe']:.2f} | Sortino: {ind['sortino']:.2f}")
    print(f"     最大回撤: {ind['max_dd_pct']:.1f}% | 连涨: {ind['consecutive_up']}天 | 连跌: {ind['consecutive_down']}天")
    print(f"     5日: {ret['r5d']:+.1f}% | 10日: {ret['r10d']:+.1f}% | 20日: {ret['r20d']:+.1f}% | 60日: {ret['r60d']:+.1f}%")
    print(f"     均线: MA5={vs['pct_ma5']:+.1f}% MA10={vs['pct_ma10']:+.1f}% MA20={vs['pct_ma20']:+.1f}% MA60={vs['pct_ma60']:+.1f}%")
    print(f"     MACD: DIF={ind['macd_dif']:.4f} Hist={ind['macd_hist']:.4f}")
    print(f"     布林位置: {ind['bb_position']:.0f}% | 量比(5日): {ind['vol_ratio_5d']:.2f} | ATR: {ind['atr_pct']:.2f}%")

    # 关键因子
    key_factors = []
    if f.get("F1_趋势强度", 0) >= 7: key_factors.append(f"趋势强(F1={f['F1_趋势强度']})")
    elif f.get("F1_趋势强度", 0) <= 3: key_factors.append(f"趋势弱(F1={f['F1_趋势强度']})")
    if f.get("F8_回调", 0) >= 8: key_factors.append(f"回调充分(F8={f['F8_回调']})")
    if f.get("F3_反转", 0) >= 7: key_factors.append(f"反转信号(F3={f['F3_反转']})")
    if f.get("F12_多周期", 0) <= 3: key_factors.append(f"多周期弱(F12={f['F12_多周期']})")
    # 量价背离（v3.0 新增）
    vol_div = s.get("volume_divergence", {})
    if vol_div.get("divergence_type"):
        div_emoji = {"bearish": "🔴", "bullish": "🟢", "weak_rally": "🟡", "panic_sell": "🟠"}
        print(f"     量价信号: {div_emoji.get(vol_div['divergence_type'], '')} {vol_div.get('description', '')}")

    if key_factors:
        print(f"     关键信号: {' | '.join(key_factors)}")

# ============================================================
# Step 5: 大盘择时 + 市场状态
# ============================================================
print()
print("─" * 60)
print("  🎯 [二] 大盘择时 + 市场状态 (v3.0)")
print("─" * 60)

# Build index_data dict for MarketTiming
timing_index_data = {}
if hs300_data:
    timing_index_data["000300"] = {"name": "沪深300", "data": hs300_data}

timing_engine = MarketTiming(
    timing_index_data,
    north_flow or [],
    total_vol or 0,
    breadth or {},
    margin or {}
)
timing_result = timing_engine.position_advice()

# 市场状态分类
regime = timing_result.get("regime", "CHOPPY")
regime_emoji = {"TREND_UP": "🟢", "CHOPPY": "🟡", "TREND_DOWN": "🟠", "CRISIS": "🔴"}
print(f"  市场状态: {regime_emoji.get(regime, '⚪')} {timing_result.get('regime_name', regime)} (置信度{timing_result.get('regime_confidence', 0):.0%})")
regime_signals = timing_result.get('regime_signals', [])
if regime_signals:
    print(f"    判定: {' | '.join(regime_signals)}")
print(f"    策略: {timing_result.get('regime_description', '')}")
print(f"    状态止损: {timing_result.get('regime_stop_loss', -0.08)*100:.0f}% | 最低买入: {timing_result.get('regime_buy_grade_min', 'B_买入')}")
print()

print(f"  传统择时信号: {timing_result['bull_signals']}/{timing_result['total_signals']}看多 -> 建议仓位 {timing_result['base_position']*100:.0f}%")
for name, value in timing_result['signal_detail'].items():
    icon = "✅" if value else "❌"
    label = name.replace("S1_HS300_above_MA20","沪深300在20日线上").replace("S2_HS300_MA60_up","沪深300的60日线向上").replace("S3_NorthFlow_5d_positive","北向资金5日净流入").replace("S4_Volume_active","成交额>2万亿").replace("S5_LimitDown_low","跌停<20家").replace("S6_Margin_increasing","融资余额增加")
    print(f"    {icon} {label}")
if timing_result['force_capped']:
    print(f"  ⚠️ 强制限制生效: 仓位上限30%")
print(f"  北向5日: {timing_result.get('north_flow_5d', 0):+.1f}亿元")
print(f"  判断: {timing_result['advice']}")

# ============================================================
# Step 6: 持仓诊断 + 对策
# ============================================================
print()
print("─" * 60)
print("  💊 [三] 持仓诊断与对策")
print("─" * 60)

total_value = 0
total_cost = 0
total_pnl = 0

for code, pos in PORTFOLIO.items():
    score_data = next((s for s in scores if s["code"] == code), None)
    if not score_data:
        continue

    shares = pos["shares"]
    cost = pos["cost"]
    price = score_data["price"]
    name = pos["name"]
    value = shares * price
    cost_value = shares * cost
    pnl = value - cost_value
    pnl_pct = (price / cost - 1) * 100

    total_value += value
    total_cost += cost_value
    total_pnl += pnl

    ind = score_data["indicators"]
    ret = score_data["returns"]
    f = score_data["factors"]
    grade = score_data["grade"]
    score = score_data["score"]

    print(f"\n  ▸ {name}({code})")
    print(f"    持仓: {shares}股 × {price:.4f} = {value:.2f}元 | 浮亏: {pnl:+.2f}元 ({pnl_pct:+.2f}%)")
    print(f"    评分: {score}分 ({grade})")

    # 诊断
    issues = []
    actions = []

    # 溢价风险诊断（v2.1 新增 — 放在最前面因为最重要）
    p_info = score_data.get("premium_info", {})
    if p_info.get("premium_pct") is not None and p_info["premium_pct"] > 5:
        issues.append(f"🚨 ETF溢价{p_info['premium_pct']:.1f}%！买入即多付{p_info['premium_pct']:.1f}%成本，溢价回归将直接亏损")
    elif p_info.get("premium_pct") is not None and p_info["premium_pct"] > 3:
        issues.append(f"⚠️ ETF溢价{p_info['premium_pct']:.1f}%，偏高，需关注溢价回落进度")
    if p_info.get("premium_pct") is None and p_info.get("is_qdii"):
        issues.append("⚠️ QDII ETF但无法获取净值数据，溢价风险未知")

    # 趋势诊断
    if f.get("F1_趋势强度", 0) <= 3:
        issues.append("⚠️ 趋势信号偏弱，MACD可能死叉或弱势")
    if ind["rsi"] > 70:
        issues.append("⚠️ RSI超买，短期回调风险大")
    elif ind["rsi"] < 30:
        issues.append("⚠️ RSI超卖，但接飞刀风险也存在")
    if ind["consecutive_down"] >= 3:
        issues.append(f"🔴 连跌{ind['consecutive_down']}天，卖压在持续")
    if ind["consecutive_up"] >= 5:
        issues.append(f"⚠️ 连涨{ind['consecutive_up']}天，获利盘压力大")
    if ret["r5d"] < -8:
        issues.append(f"🔴 5日暴跌{ret['r5d']:.1f}%，短期严重超跌")
    if ret["r20d"] < -10:
        issues.append(f"🔴 20日跌{ret['r20d']:.1f}%，中期趋势恶化")
    if pnl_pct <= -8:
        issues.append(f"🚨 浮亏{pnl_pct:.1f}%已达止损线！")

    # 积极信号
    if f.get("F8_回调", 0) >= 8:
        issues.append("✅ 回调充分，技术反弹概率大")
    if f.get("F3_反转", 0) >= 7:
        issues.append("✅ 反转信号出现，短期可能见底")
    if 35 <= ind["rsi"] <= 50 and ind["consecutive_down"] >= 2:
        issues.append("💡 RSI中性偏低+连跌，接近超卖反弹区")

    for issue in issues:
        print(f"    {issue}")

    # 具体对策
    print(f"    ┌─ 操作建议 ─────────────────────")

    # 溢价风险优先判断（v2.1 新增）
    p_info_full = score_data.get("premium_info", {})
    if p_info_full.get("premium_pct") is not None and p_info_full["premium_pct"] > 5:
        print(f"    │ 🚨 溢价{p_info_full['premium_pct']:.1f}%！不建议加仓，溢价回归即亏损")
        print(f"    │    等溢价回落至3%以内再考虑操作")
    elif p_info_full.get("premium_pct") is not None and p_info_full["premium_pct"] > 3:
        print(f"    │ ⚠️ 溢价{p_info_full['premium_pct']:.1f}%，偏高，减少操作频率")

    if pnl_pct <= -8:
        # 止损触发
        stop_price = round(cost * 0.92, 4)
        print(f"    │ 🚨 止损触发！建议止损价: {stop_price:.4f} (-8%)")
        print(f"    │    立即卖出{shares}股，亏损约{shares*(stop_price-cost):.0f}元")
        print(f"    │    不止损的最大风险: 继续下跌到-12%~-15%")
    elif grade in ["D_谨慎", "E_回避"] and pnl_pct > 3:
        print(f"    │ 💰 浮盈{pnl_pct:.1f}%但评分走弱，建议止盈")
        print(f"    │    卖出一半锁定利润，剩余观察")
    elif grade in ["D_谨慎", "E_回避"] and pnl_pct <= 0:
        print(f"    │ ⚠️ 浮亏+评分走弱，但尚未触发止损")
        print(f"    │    暂持观察，密切关注止损线")
        stop_price = round(cost * 0.92, 4)
        print(f"    │    止损价: {stop_price:.4f} (距当前{((price-stop_price)/price*100):.1f}%)")
    elif grade in ["A_强烈买入", "B_买入"] and pnl_pct < 0:
        print(f"    │ 📈 评分良好但浮亏中，可考虑逢低加仓")
        print(f"    │    但现金仅{AVAILABLE_CASH}元，加仓需等7月资金到账")
    elif grade in ["A_强烈买入", "B_买入"] and pnl_pct >= 0:
        print(f"    │ ✅ 评分好+盈利，继续持有")
        print(f"    │    止盈线: {round(cost*1.08,4):.4f} (+8%)")
    else:
        print(f"    │ ⏸️ 观望为主，不做操作")

    # 止损止盈价
    stop_loss = round(cost * 0.92, 4)
    take_profit = round(cost * 1.08, 4)
    print(f"    │ 止损: {stop_loss:.4f} | 止盈: {take_profit:.4f}")
    print(f"    └──────────────────────────────────")

# ============================================================
# Step 7: 投资组合风险 (v3.0 新增)
# ============================================================
print()
print("─" * 60)
print("  🛡️ [四] 投资组合风险管理 (v3.0)")
print("─" * 60)

# 构建portfolio dict传给risk_engine
portfolio_for_risk = {}
for code, pos in PORTFOLIO.items():
    score_data = next((s for s in scores if s["code"] == code), None)
    price = score_data["price"] if score_data else pos["cost"]
    portfolio_for_risk[code] = {
        "shares": pos["shares"],
        "cost": pos["cost"],
        "current_price": price,
        "name": pos["name"]
    }

risk_report = portfolio_risk_report(portfolio_for_risk, etf_data, scores)
# 出相关性矩阵
corr = risk_report.get("correlation", {})
if corr.get("n_assets", 0) >= 2:
    print(f"  相关性矩阵 (60日):")
    codes = list(corr.get("matrix", {}).keys())
    if codes:
        header = "         " + "".join(f"{c:<8}" for c in codes)
        print(header)
        for c1 in codes:
            row = f"  {c1:<7}"
            for c2 in codes:
                r_val = corr["matrix"].get(c1, {}).get(c2, 0)
                row += f"{r_val:>7.3f} "
            print(row)
    print(f"  平均相关性: {corr.get('avg_correlation', 0):.3f}")
    if corr.get("warning_pairs"):
        print(f"  ⚠️ 高相关警报 (r>0.7):")
        for c1, c2, r, n1, n2 in corr["warning_pairs"]:
            print(f"    {n1} ↔ {n2}: r={r:.3f} (同涨同跌风险)")
else:
    print(f"  相关性: 数据不足")

# VaR
var = risk_report.get("var", {})
if var and not var.get("error"):
    print(f"  VaR(95%): {var['var_95']:.2f}元 ({var['var_95_pct']:.1f}%) — 95%概率日亏损")
    print(f"  CVaR(95%): {var['cvar_95']:.2f}元 ({var['cvar_95_pct']:.1f}%) — 极端情况平均亏损")
    print(f"  历史最差日: {var['worst_day']:.2f}元 ({var['worst_day_pct']:.1f}%)")

# 集中度
conc = risk_report.get("concentration", {})
level_emoji = {"safe": "✅", "warning": "⚠️", "danger": "🚨"}
print(f"  分散化评分: {level_emoji.get(conc.get('level', ''), '⚪')} {conc.get('score', '?')}分 ({conc.get('level', '?')})")
print(f"  {conc.get('advice', '')}")

# ============================================================
# Step 8: 账户汇总
# ============================================================
print()
print("─" * 60)
print("  📋 [五] 账户汇总")
print("─" * 60)

total_assets = total_value + AVAILABLE_CASH
cash_pct = AVAILABLE_CASH / total_assets * 100 if total_assets > 0 else 0

print(f"  持仓总市值: {total_value:.2f}元")
print(f"  持仓总成本: {total_cost:.2f}元")
print(f"  浮动盈亏: {total_pnl:+.2f}元 ({total_pnl/total_cost*100:+.2f}%)")
print(f"  可用资金: {AVAILABLE_CASH:.2f}元")
print(f"  总资产: {total_assets:.2f}元")
print(f"  现金占比: {cash_pct:.1f}% {'🚨 严重不足!' if cash_pct < 5 else '⚠️ 偏低' if cash_pct < 15 else '✅ 健康'}")

# 择时仓位建议
target_pos = timing_result["base_position"]
target_amount = total_assets * target_pos
print(f"\n  大盘择时建议仓位: {target_pos*100:.0f}% (≈{target_amount:.0f}元)")
print(f"  当前实际仓位: {(total_value/total_assets*100):.0f}% (≈{total_value:.0f}元)")

if total_value > target_amount:
    print(f"  ⚠️ 当前仓位高于择时建议，注意风险控制")
else:
    print(f"  ✅ 当前仓位在择时建议范围内")

# 优先级建议
print(f"\n  ⚡ 优先级行动清单:")
worst = sorted(scores, key=lambda x: x["score"])
for i, s in enumerate(worst):
    pos = PORTFOLIO.get(s["code"], {})
    cost = pos.get("cost", s["price"])
    pnl_pct = (s["price"]/cost - 1)*100
    p_info = s.get("premium_info", {})
    if p_info.get("premium_pct") is not None and p_info["premium_pct"] > 5:
        print(f"    {i+1}. 🚨 {s['name']}: 溢价{p_info['premium_pct']:.1f}%! 评分因溢价从{s.get('technical_score','?')}降至{s['score']}分")
    elif pnl_pct <= -5:
        print(f"    {i+1}. 🔴 {s['name']}: 浮亏{pnl_pct:.1f}%+评分{s['score']}分 — 优先关注止损")
    elif s["grade"] in ["D_谨慎", "E_回避"]:
        print(f"    {i+1}. 🟠 {s['name']}: 评分{s['score']}分({s['grade']}) — 评分走弱需警惕")

print()
print("=" * 70)
print("  ⚠️ 免责声明: 量化模型仅供辅助决策，不构成投资建议。")
print("  投资有风险，入市需谨慎。请结合自身情况独立判断。")
print("=" * 70)
