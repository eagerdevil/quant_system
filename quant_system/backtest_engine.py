#!/usr/bin/env python
"""
回测引擎 v6.0 (pandas+numpy 向量化)
=====================================
基于历史数据模拟交易，评估16因子策略表现
v6.0: 绩效指标计算迁移到 numpy 向量化
"""
import json, sys, os, io, math, time, logging
from datetime import datetime, timedelta
import numpy as np

# Only wrap stdout/stderr when running as script (avoids subprocess import conflicts)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_etf_kline, fetch_index_daily, KEY_ETFS
from quant_engine import (
    compute_indicators, score_factors, sma, FACTOR_MAX,
    CURRENT_WEIGHTS, FACTOR_NAMES, SYSTEM_CONFIG
)

TODAY = datetime.now().strftime("%Y%m%d")
COMMISSION_RATE = SYSTEM_CONFIG['commission_rate']  # 万5 (0.0005)
MIN_COMMISSION = 5.0       # A股最低佣金5元/笔
SLIPPAGE_BPS = 1.0         # 滑点1bp (0.01%)，ETF流动性好取低值
STAMP_DUTY_SELL = 0.0      # ETF免印花税（个股为0.05%）

def trade_cost(shares, price, is_buy=True):
    """计算真实交易成本（含最低佣金+滑点）

    A股ETF交易成本:
    - 佣金: 万2.5(双向), 最低5元/笔
    - 印花税: 0 (ETF免)
    - 滑点: 约1bp

    返回: (成交金额, 费用)
    """
    trade_amount = shares * price
    # 佣金: max(万2.5, 5元)
    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    # 滑点: 买卖各1bp
    slippage = trade_amount * SLIPPAGE_BPS / 10000
    cost = commission + slippage
    # 买入: 实际花费 = 成交金额 + 费用
    # 卖出: 实际到账 = 成交金额 - 费用
    if is_buy:
        return trade_amount + cost, cost
    else:
        return trade_amount - cost, cost

# v5.0: cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")
CACHE_TTL_HOURS = 6


# ============================================================
# 回测配置
# ============================================================
# v7.6: 池子从手工17只 → 全量KEY_ETFS(60只, 覆盖宽基/行业/QDII/策略)
# 旧版手工选池存在幸存者偏差: 只含当前存活且被关注的ETF, 历史表现差/已退市的
# 不在池中, 回测收益虚高。全量池让回测覆盖更真实的横截面(上市日由数据自动处理)。
BACKTEST_POOL = dict(KEY_ETFS)

DEFAULT_START = "2024-01-01"
DEFAULT_CAPITAL = 100000
MAX_HOLDINGS = SYSTEM_CONFIG['max_total_holdings']
SINGLE_WEIGHT = SYSTEM_CONFIG['max_single_weight']

# ============================================================
# 数据准备 (v5.0: API缓存)
# ============================================================
def _cache_key(start_date, end_date, pool_keys):
    # 用全部代码的排序+hash避免碰撞（旧版只取前10个 → 不同子集可能碰撞）
    # 注意: 必须用 hashlib 而非内置 hash() —— Python 内置 hash 按进程随机(PYTHONHASHSEED)，
    #       跨进程运行永远算不出相同键，缓存形同虚设
    import hashlib
    codes_str = "-".join(sorted(pool_keys))
    codes_hash = hashlib.md5(codes_str.encode("utf-8")).hexdigest()[:8]
    return f"{start_date}_{end_date or 'now'}_{len(pool_keys)}etf_{codes_hash}"

def _load_cache(cache_key):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.info(f"[backtest] cache expired ({age_hours:.1f}h), refetching")
            return None
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[backtest] using cache ({age_hours:.1f}h ago, {len(data.get('trading_days',[]))} days)")
        return data
    except:
        return None

def _save_cache(cache_key, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    slim = {"etfs": {}, "benchmark": data.get("benchmark", []), "trading_days": data.get("trading_days", [])}
    for code, edata in data.get("etfs", {}).items():
        slim["etfs"][code] = {"name": edata["name"], "klines": edata.get("klines", [])}
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(slim, f, ensure_ascii=False)
        logger.info(f"[backtest] cache saved: {cache_file}")
    except Exception as e:
        logger.info(f"[backtest] cache write failed: {e}")

def prepare_backtest_data(start_date, end_date=None, pool=None):
    """批量拉取回测所需全部数据"""
    if pool is None:
        pool = BACKTEST_POOL
    if end_date is None:
        end_date = TODAY

    cache_key = _cache_key(start_date, end_date, list(pool.keys()))
    cached = _load_cache(cache_key)
    if cached:
        return cached

    logger.info(f"[回测] 拉取数据: {start_date} ~ {end_date}")
    try:
        # 兼容 YYYYMMDD 与 YYYY-MM-DD 两种格式（默认配置为 "2024-01-01"）
        start_dt = datetime.strptime(str(start_date).replace("-", ""), "%Y%m%d")
        lookback_dt = start_dt - timedelta(days=400)
        total_days = (datetime.now() - lookback_dt).days + 50
    except (ValueError, TypeError):
        total_days = 600

    raw_etfs = {}
    codes = list(pool.keys())
    logger.info(f"[回测] 拉取 {len(codes)} 只ETF K线 (每只{total_days}天)...")
    for i, code in enumerate(codes):
        name = pool.get(code, code)
        klines = None
        for attempt in range(3):
            klines = fetch_etf_kline(code, days=total_days)
            if klines:
                break
            time.sleep(1.0)
        if klines:
            raw_etfs[code] = {"name": name, "klines": klines}
            if (i + 1) % 5 == 0:
                logger.info(f"  [{i+1}/{len(codes)}] {code} {name} OK ({len(klines)}条)")
        else:
            logger.info(f"  [{i+1}/{len(codes)}] {code} {name} FAIL")

    min_klines = SYSTEM_CONFIG['min_klines_for_backtest']
    valid_etfs = {}
    dropped = []
    for code, data in raw_etfs.items():
        if len(data["klines"]) >= min_klines:
            valid_etfs[code] = data
        else:
            dropped.append(f"{code}({data['name']}:{len(data['klines'])}条)")
    if dropped:
        logger.info(f"[回测] 剔除数据不足的ETF: {', '.join(dropped)}")
    logger.info(f"[回测] 有效ETF: {len(valid_etfs)}只 (要求>={min_klines}条K线)")
    logger.info(f"[回测] 拉取沪深300指数 ({total_days}天)...")
    benchmark_klines = fetch_index_daily("000300", days=total_days)
    if benchmark_klines:
        benchmark = [{"date": k["date"], "close": k["close"]} for k in benchmark_klines]
    else:
        benchmark = []
    trading_days = _get_common_trading_days(valid_etfs, start_date)
    logger.info(f"[回测] 数据就绪: {len(valid_etfs)}只ETF, {len(trading_days)}个交易日")
    result = {"etfs": valid_etfs, "benchmark": benchmark, "trading_days": trading_days}
    _save_cache(cache_key, result)
    return result

def _get_common_trading_days(valid_etfs, start_date, min_etfs=5):
    if not valid_etfs:
        return []
    date_count = {}
    for code, data in valid_etfs.items():
        for k in data["klines"]:
            d = k["date"]
            date_count[d] = date_count.get(d, 0) + 1
    result = sorted(d for d, cnt in date_count.items() if cnt >= min_etfs and d >= start_date)
    return result


# ============================================================
# 回测核心
# ============================================================
def run_backtest(data, initial_capital=DEFAULT_CAPITAL):
    """执行回测"""
    valid_etfs = data["etfs"]
    trading_days = data["trading_days"]
    benchmark_data = data.get("benchmark", [])

    bench_map = {}
    for b in benchmark_data:
        bench_map[b["date"]] = b["close"]

    if len(trading_days) < 251:
        logger.info("[回测] 交易日不足250天，无法回测")
        return None

    equity_days = trading_days[250:]

    cash = initial_capital
    holdings = {}
    equity_curve = []
    trades_log = []

    initial_bench = None
    for d in equity_days:
        if d in bench_map:
            initial_bench = bench_map[d]
            break

    if initial_bench is None:
        initial_bench = initial_capital  # 无基准数据时回退

    logger.info(f"[回测] 开始: {equity_days[0]} ~ {equity_days[-1]} ({len(equity_days)}天)")
    logger.info(f"[回测] 初始资金: ¥{initial_capital:,.0f}")

    # v7.6 (8/5): 消除前视偏差 — T日收盘计算信号 → T+1日开盘价执行
    # pending: [{code, name, action, shares, reason}]，每个T日信号层生成，次日执行层用开盘价成交
    pending = []

    def _bar_on(code, date):
        """取某ETF在指定日期的K线（无数据返回None，停牌跳过）"""
        edata = valid_etfs.get(code)
        if not edata:
            return None
        for k in edata["klines"]:
            if k["date"] == date:
                return k
        return None

    def _execute_pending(today):
        """执行层（T+1）：按昨日信号用今日开盘价成交；先卖后买"""
        nonlocal cash
        if not pending:
            return
        # 先执行全部卖出（止损/排名），再执行买入，保证卖出回笼资金可用于当日买入
        # 8/12修复: 原排序 BUY=0/SELL=1 是先买后卖，与注释矛盾（模拟盘 paper_trading.py 已按 SELL-first 实现）
        for p in sorted(pending, key=lambda x: 0 if x["action"] != "BUY" else 1):
            code = p["code"]
            bar = _bar_on(code, today)
            if bar is None or bar.get("open", 0) <= 0:
                logger.info(f"[回测] {today} {code} 无有效开盘价，跳过{p['action']}({p['reason']})")
                continue
            exec_price = bar["open"]

            if p["action"] != "BUY":
                pos = holdings.get(code)
                if pos is None:
                    continue
                sell_shares = min(p["shares"], pos["shares"])
                sell_amount, _ = trade_cost(sell_shares, exec_price, is_buy=False)
                cash += sell_amount
                pnl_pct = (exec_price / pos["cost"] - 1) * 100
                trades_log.append({
                    "date": today, "code": code, "name": pos["name"],
                    "action": p["action"], "price": exec_price,
                    "shares": sell_shares, "amount": round(sell_amount, 2),
                    "pnl_pct": round(pnl_pct, 1), "reason": p["reason"]
                })
                if sell_shares >= pos["shares"]:
                    del holdings[code]
                else:
                    pos["shares"] -= sell_shares
            else:
                # BUY: 防止止损当日重买回（P0修复: 已计划卖出的代码不进入买入候选，此处再兜底一次）
                if code in holdings:
                    continue
                buy_shares = p["shares"]
                cost_with_fee, _ = trade_cost(buy_shares, exec_price, is_buy=True)
                if cost_with_fee > cash:
                    affordable = int(cash / (exec_price * (1 + COMMISSION_RATE) + exec_price * SLIPPAGE_BPS / 10000) / 100) * 100
                    if affordable < 100:
                        continue
                    buy_shares = affordable
                    cost_with_fee, _ = trade_cost(buy_shares, exec_price, is_buy=True)
                    if cost_with_fee > cash:  # P0修复: 最低5元佣金使实际成本可超估算, 现金永不为负
                        continue
                if buy_shares < 100:
                    continue
                cash -= cost_with_fee
                holdings[code] = {"shares": buy_shares, "cost": exec_price, "name": p["name"], "price": exec_price}
                trades_log.append({
                    "date": today, "code": code, "name": p["name"],
                    "action": "BUY", "price": exec_price, "shares": buy_shares,
                    "amount": round(cost_with_fee, 2), "reason": p["reason"]
                })
        pending.clear()

    for day_idx, today in enumerate(equity_days):
        # ===== 执行层: 先执行昨日生成的计划（T+1开盘价成交）=====
        _execute_pending(today)

        # ===== 信号层: 用今日收盘数据计算信号 =====
        etf_scores = []
        for code, edata in valid_etfs.items():
            all_dates = [k["date"] for k in edata["klines"]]
            if today not in all_dates:
                continue
            pos = all_dates.index(today)
            kline_slice = edata["klines"][:pos + 1]
            if len(kline_slice) < 30:
                continue

            closes = [k["close"] for k in kline_slice]
            highs = [k["high"] for k in kline_slice]
            lows = [k["low"] for k in kline_slice]
            volumes = [k["volume"] for k in kline_slice]

            try:
                indicators = compute_indicators(closes, highs, lows, volumes)
                factors = score_factors(indicators)
                weights = CURRENT_WEIGHTS
                weighted_sum = sum(factors[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
                max_weighted = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
                score = round(weighted_sum / max_weighted * 100) if max_weighted > 0 else 50
                etf_scores.append({
                    "code": code, "name": edata["name"], "score": score,
                    "price": closes[-1],
                    "rsi": round(indicators["rsi"], 1),
                    "volatility": round(indicators.get("volatility", 0.25), 4),
                })
            except Exception:
                continue

        if not etf_scores:
            if equity_curve:
                prev = equity_curve[-1]
                equity_curve.append(dict(prev, date=today))
            continue

        # 简化大盘择时
        hs300_etf = valid_etfs.get("510300")
        market_bullish = 1.0  # 默认满仓（无指数数据时）
        if hs300_etf:
            hs300_closes = [k["close"] for k in hs300_etf["klines"]]
            all_dates = [k["date"] for k in hs300_etf["klines"]]
            if today in all_dates:
                pos = all_dates.index(today)
                hs300_slice = hs300_closes[:pos + 1]
                if len(hs300_slice) >= 60:
                    ma20 = sma(hs300_slice, 20)
                    ma60 = sma(hs300_slice, 60)
                    ma60_prev = sma(hs300_slice[:-1], 60) if len(hs300_slice) > 60 else ma60
                    s1 = hs300_slice[-1] > ma20
                    s2 = ma60 > ma60_prev
                    bull_signals = sum([s1, s2])
                    if bull_signals == 0:
                        market_bullish = 0.20
                    elif bull_signals == 1:
                        market_bullish = 0.50
                    else:
                        market_bullish = 1.0

        target_invested = initial_capital * market_bullish
        etf_scores.sort(key=lambda x: x["score"], reverse=True)
        top_codes = set(s["code"] for s in etf_scores[:MAX_HOLDINGS])

        # === 信号层-止损检查（v7.2）: 生成T+1卖出计划 ===
        # 已计划卖出的代码集合：防止同一持仓同日生成重复卖出计划、以及止损后当日重买回
        sold_codes = set()
        for code in list(holdings.keys()):
            if code in sold_codes:
                continue
            pos = holdings[code]
            current_price = next((s["price"] for s in etf_scores if s["code"] == code), pos.get("price", pos["cost"]))
            pnl_pct = (current_price / pos["cost"] - 1) * 100
            # 更新持仓中的最高价（用于追踪止盈）
            high_water = pos.get("high_water", pos["cost"])
            if current_price > high_water:
                pos["high_water"] = current_price
                high_water = current_price
            pullback_from_high = (current_price / high_water - 1) * 100  # 从高点回撤

            should_stop = False
            stop_reason = ""

            # 条件1: 硬止损 -8%
            if pnl_pct <= -8.0:
                should_stop = True
                stop_reason = f"硬止损触发 (浮亏{pnl_pct:.1f}% ≤ -8%)"
            # 条件2: 追踪止盈 - 从高点回撤>6%且曾经浮盈>10%
            elif pnl_pct > 10 and pullback_from_high <= -6.0:
                should_stop = True
                stop_reason = f"追踪止盈 (高点+{(high_water/pos['cost']-1)*100:.1f}%→回撤{pullback_from_high:.1f}%)"
            # 条件3: 评分E级+亏损中
            elif pnl_pct < 0:
                score_data = next((s for s in etf_scores if s["code"] == code), None)
                if score_data and score_data["score"] < 42:
                    should_stop = True
                    stop_reason = f"评分E级({score_data['score']}分)+浮亏{pnl_pct:.1f}%"

            if should_stop:
                sold_codes.add(code)
                pending.append({
                    "code": code, "name": pos["name"], "action": "STOP_LOSS",
                    "shares": pos["shares"], "reason": stop_reason
                })

        # 卖出（评分排名下滑）→ 生成T+1卖出计划
        for code in list(holdings.keys()):
            if code in sold_codes:
                continue
            if code not in top_codes:
                pos = holdings[code]
                pnl_pct = (pos["price"] / pos["cost"] - 1) * 100
                sold_codes.add(code)
                pending.append({
                    "code": code, "name": pos["name"], "action": "SELL",
                    "shares": pos["shares"], "reason": f"得分排名下滑 (浮盈{pnl_pct:+.1f}%)"
                })

        # 买入 → 生成T+1买入计划（排除已计划卖出的代码，修复止损当日重买回）
        needed_holdings = MAX_HOLDINGS - len(holdings) + len(sold_codes)
        if needed_holdings > 0:
            buy_candidates = [s for s in etf_scores[:MAX_HOLDINGS]
                              if s["code"] not in holdings and s["code"] not in sold_codes]
            if buy_candidates:
                inv_vols = [1.0 / max(s.get("volatility", 0.20), 0.05) for s in buy_candidates]
                total_inv_vol = sum(inv_vols)
                weights_list = [iv / total_inv_vol for iv in inv_vols] if total_inv_vol > 0 else [1.0 / len(buy_candidates)] * len(buy_candidates)
            else:
                weights_list = []

            for idx, candidate in enumerate(buy_candidates[:needed_holdings]):
                vol_weight = weights_list[idx] if idx < len(weights_list) else 1.0 / needed_holdings
                budget = min(cash * vol_weight * 1.2, initial_capital * SINGLE_WEIGHT)
                current_invested = sum(h["shares"] * h.get("price", h["cost"]) for h in holdings.values())
                if current_invested >= target_invested:
                    break
                price = candidate["price"]
                if price <= 0:
                    continue
                shares = int(budget / price / 100) * 100
                if shares < 100:
                    continue
                pending.append({
                    "code": candidate["code"], "name": candidate["name"],
                    "action": "BUY", "shares": shares,
                    "reason": f"评分{candidate['score']}分 排名TOP{MAX_HOLDINGS}"
                })

        # 更新持仓市值
        holdings_value = 0
        for code, pos in holdings.items():
            current_price = next((s["price"] for s in etf_scores if s["code"] == code), pos.get("price", pos["cost"]))
            pos["price"] = current_price
            holdings_value += pos["shares"] * current_price

        nav = cash + holdings_value

        bench_close = bench_map.get(today)
        if bench_close:
            if initial_bench is None:
                initial_bench = bench_close
            bench_nav = bench_close / initial_bench * initial_capital if initial_bench else initial_capital
        else:
            bench_nav = equity_curve[-1]["benchmark_nav"] if equity_curve else initial_capital

        equity_curve.append({
            "date": today, "nav": round(nav, 2), "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2), "holdings_count": len(holdings),
            "benchmark_nav": round(bench_nav, 2),
        })

        if (day_idx + 1) % 50 == 0:
            ret = (nav / initial_capital - 1) * 100
            logger.info(f"  [{day_idx+1}/{len(equity_days)}] {today} | 净值: ¥{nav:,.0f} | 收益: {ret:+.1f}%")

    metrics = _calc_metrics(equity_curve, initial_capital, trades_log)
    return {
        "equity_curve": equity_curve, "trades": trades_log, "metrics": metrics,
        "config": {
            "start_date": equity_days[0], "end_date": equity_days[-1],
            "initial_capital": initial_capital,
            "commission_rate": COMMISSION_RATE,
            "min_commission": MIN_COMMISSION,
            "slippage_bps": SLIPPAGE_BPS,
            "pool_size": len(valid_etfs), "trading_days": len(equity_days),
        }
    }


# ============================================================
# 绩效计算 — numpy 向量化
# ============================================================
def _calc_metrics(equity_curve, initial_capital, trades):
    """计算全套绩效指标 — numpy 向量化"""
    if len(equity_curve) < 2:
        return {}

    navs = np.array([e["nav"] for e in equity_curve], dtype=np.float64)
    bench_navs = np.array([e["benchmark_nav"] for e in equity_curve], dtype=np.float64)
    dates = [e["date"] for e in equity_curve]

    final_nav = navs[-1]
    total_return = float((final_nav / initial_capital - 1) * 100)

    # 日收益率
    daily_returns_arr = np.diff(navs) / navs[:-1]
    n_days = len(daily_returns_arr)
    n_years = n_days / 252
    annual_return = float(((1 + total_return / 100) ** (1 / n_years) - 1) * 100) if n_years > 0 else 0

    # 夏普比率
    if len(daily_returns_arr) > 1:
        avg_daily = float(np.mean(daily_returns_arr))
        std_daily = float(np.std(daily_returns_arr, ddof=1))
        sharpe = float((avg_daily / std_daily) * np.sqrt(252)) if std_daily > 0 else 0.0
    else:
        sharpe = 0.0

    # 最大回撤 & 回撤持续天数 — numpy
    peak = np.maximum.accumulate(navs)
    drawdowns = (peak - navs) / peak
    max_dd = float(np.max(drawdowns) * 100)
    max_dd_idx = int(np.argmax(drawdowns))
    max_dd_date = dates[max_dd_idx] if max_dd_idx < len(dates) else ""

    # 回撤持续时间
    peak_idx = int(np.argmax(peak[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    dd_days = max_dd_idx - peak_idx

    # Calmar比率
    calmar = float(annual_return / max_dd) if max_dd > 0 else 999

    # 胜率
    wins = float(np.sum(daily_returns_arr > 0))
    total_trades = len(daily_returns_arr)
    win_rate = float(wins / total_trades * 100) if total_trades > 0 else 0

    # 盈亏比
    winning_rets = daily_returns_arr[daily_returns_arr > 0]
    losing_rets = daily_returns_arr[daily_returns_arr < 0]
    avg_win = float(np.mean(winning_rets)) if len(winning_rets) > 0 else 0
    avg_loss = float(np.mean(np.abs(losing_rets))) if len(losing_rets) > 0 else 0
    profit_factor = float(avg_win / avg_loss) if avg_loss > 0 else 999

    # 超额收益 vs 基准
    excess_return = total_return - float((bench_navs[-1] / bench_navs[0] - 1) * 100)

    # 交易统计
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]
    sell_pnls = []
    for t in sells:
        matching_buys = [b for b in buys if b["code"] == t["code"] and b["date"] <= t["date"]]
        if matching_buys:
            buy_price = matching_buys[-1]["price"]
            sell_pnls.append((t["price"] / buy_price - 1) * 100)

    winning_trades = len([p for p in sell_pnls if p > 0])
    trade_win_rate = winning_trades / len(sell_pnls) * 100 if sell_pnls else 0

    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_date": max_dd_date,
        "max_drawdown_days": dd_days,
        "calmar_ratio": round(calmar, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != 999 else 999,
        "total_trades": len(trades),
        "winning_trades": winning_trades,
        "trade_win_rate": round(trade_win_rate, 1) if sell_pnls else 0,
        "excess_return_pct": round(excess_return, 2),
        "final_nav": round(float(final_nav), 2),
        "initial_capital": initial_capital,
    }


# ============================================================
# v8.0 P2-10: 回测报告一键生成
# ============================================================
def generate_full_report(start_date=None, end_date=None, output_path=None):
    """
    运行完整回测并生成自包含HTML报告。

    报告包含:
    1. 权益曲线 (组合 vs 基准)
    2. 年度收益表
    3. 最大回撤区间图
    4. 月度收益热力图数据
    5. 交易记录
    6. 绩效指标面板

    Args:
        start_date: 回测起始日期 (默认 2024-01-01)
        end_date: 回测结束日期 (默认今天)
        output_path: 输出HTML路径 (默认 backtest_report_{date}.html)

    返回: output_path
    """
    import numpy as np

    if start_date is None:
        start_date = DEFAULT_START
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"backtest_report_{datetime.now().strftime('%Y%m%d')}.html")

    logger.info(f"回测报告: {start_date} → {end_date}")

    # 运行回测
    data = prepare_backtest_data(start_date, end_date)
    if data is None:
        raise RuntimeError("回测数据准备失败")

    result = run_backtest(data)
    if result is None:
        raise RuntimeError("回测执行失败")

    metrics = result["metrics"]
    equity = result["equity_curve"]
    trades = result["trades"]
    # 基准净值在 equity_curve 每条记录的 benchmark_nav 字段中（run_backtest 无独立键）
    benchmark = result.get("benchmark_navs") or [
        {"date": e["date"], "nav": e["benchmark_nav"]}
        for e in equity if e.get("benchmark_nav") is not None
    ]

    # ===== 计算报表数据 =====
    # 年度收益
    annual_returns = {}
    for i in range(1, len(equity)):
        prev = datetime.strptime(equity[i - 1]["date"], "%Y-%m-%d")
        curr = datetime.strptime(equity[i]["date"], "%Y-%m-%d")
        year = curr.year
        if year not in annual_returns:
            annual_returns[year] = equity[i]["nav"] / equity[i - 1]["nav"] - 1
        else:
            annual_returns[year] = (1 + annual_returns[year]) * (equity[i]["nav"] / equity[i - 1]["nav"]) - 1

    # 月度收益
    monthly_returns = {}
    for i in range(1, len(equity)):
        curr = datetime.strptime(equity[i]["date"], "%Y-%m-%d")
        key = curr.strftime("%Y-%m")
        if key not in monthly_returns:
            monthly_returns[key] = equity[i]["nav"] / equity[i - 1]["nav"] - 1
        else:
            monthly_returns[key] = (1 + monthly_returns[key]) * (equity[i]["nav"] / equity[i - 1]["nav"]) - 1

    # 权益曲线JSON (for Chart.js)
    navs = [{"date": e["date"], "nav": round(e["nav"], 2)} for e in equity]
    bench_navs = [{"date": b["date"], "nav": round(b["nav"], 2)} for b in benchmark] if benchmark else []
    # 基准 dataset 预生成（为空时省略该条，避免JS语法错误导致整页图表不渲染）
    bench_dataset_html = (
        "{ label: '基准(沪深300)', data: bench.map(e => e.nav), "
        "borderColor: '#8b949e', borderDash: [5,5], tension: 0.3, pointRadius: 0 }"
    ) if bench_navs else ""

    # 交易汇总
    buy_count = len([t for t in trades if t["action"] == "BUY"])
    sell_count = len([t for t in trades if t["action"] == "SELL"])

    # ===== 生成HTML =====
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略回测报告 | {start_date} → {end_date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{ --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a; --text: #e1e4e8; --text2: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff; --gold: #d2991d; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}
  h1 {{ font-size: 28px; font-weight: 700; background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ color: var(--text2); font-size: 14px; margin-bottom: 28px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }}
  .metric .label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }}
  .metric .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .pos {{ color: var(--green); }} .neg {{ color: var(--red); }} .neutral {{ color: var(--blue); }}
  .chart-box {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 16px; }}
  .chart-box h2 {{ color: var(--blue); font-size: 16px; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--border); color: var(--text2); font-weight: 600; }}
  .footer {{ text-align: center; color: var(--text2); margin-top: 40px; font-size: 11px; }}
</style></head><body><div class="container">
<h1>📊 量化策略回测报告</h1>
<div class="subtitle">回测区间: {start_date} → {end_date} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 初始资金: {metrics['initial_capital']:,.0f}元</div>

<!-- 绩效指标 -->
<div class="metrics">
<div class="metric"><div class="label">累计收益</div><div class="value {'pos' if metrics['total_return_pct']>=0 else 'neg'}">{metrics['total_return_pct']:+.1f}%</div></div>
<div class="metric"><div class="label">年化收益</div><div class="value {'pos' if metrics['annual_return_pct']>=0 else 'neg'}">{metrics['annual_return_pct']:+.1f}%</div></div>
<div class="metric"><div class="label">夏普比率</div><div class="value neutral">{metrics['sharpe_ratio']:.2f}</div></div>
<div class="metric"><div class="label">最大回撤</div><div class="value neg">{metrics['max_drawdown_pct']:.1f}%</div></div>
<div class="metric"><div class="label">Calmar</div><div class="value neutral">{metrics['calmar_ratio']:.2f}</div></div>
<div class="metric"><div class="label">胜率</div><div class="value {'pos' if metrics['trade_win_rate']>=50 else 'neg'}">{metrics['trade_win_rate']:.0f}%</div></div>
<div class="metric"><div class="label">盈亏比</div><div class="value neutral">{metrics.get('profit_factor', 0):.2f}</div></div>
<div class="metric"><div class="label">超额收益</div><div class="value {'pos' if metrics['excess_return_pct']>=0 else 'neg'}">{metrics['excess_return_pct']:+.1f}%</div></div>
<div class="metric"><div class="label">总交易</div><div class="value neutral">{metrics['total_trades']}</div></div>
<div class="metric"><div class="label">最终净值</div><div class="value neutral">{metrics['final_nav']:.0f}元</div></div>
</div>

<!-- 权益曲线 -->
<div class="chart-box"><h2>权益曲线 (组合 vs 基准)</h2>
<canvas id="equityChart" height="300"></canvas></div>

<!-- 回撤 -->
<div class="chart-box"><h2>回撤曲线</h2>
<canvas id="drawdownChart" height="200"></canvas></div>
"""

    # 年度收益表
    html += '<div class="chart-box"><h2>年度收益</h2><table><tr><th>年份</th><th>收益率</th></tr>'
    for year in sorted(annual_returns.keys()):
        ret = annual_returns[year] * 100
        html += f'<tr><td>{year}</td><td class="{"pos" if ret>=0 else "neg"}">{ret:+.1f}%</td></tr>'
    html += '</table></div>'

    # 交易记录 (最近20条)
    html += '<div class="chart-box"><h2>交易记录 (最近20笔)</h2><table>'
    html += '<tr><th>日期</th><th>操作</th><th>代码</th><th>名称</th><th>价格</th><th>数量</th><th>金额</th></tr>'
    for t in trades[-20:]:
        act_color = "pos" if t["action"] == "BUY" else "neg"
        html += f'<tr><td>{t["date"]}</td><td class="{act_color}">{t["action"]}</td><td>{t["code"]}</td><td>{t["name"]}</td><td>{t["price"]:.4f}</td><td>{t["shares"]}</td><td>{t["amount"]:.0f}</td></tr>'
    html += '</table></div>'

    # 权益曲线 JSON
    navs_json = json.dumps(navs, ensure_ascii=False)
    bench_json = json.dumps(bench_navs, ensure_ascii=False)

    # 计算回撤序列
    max_so_far = 0
    dd_series = []
    for e in navs:
        max_so_far = max(max_so_far, e["nav"])
        dd = (e["nav"] - max_so_far) / max_so_far * 100 if max_so_far > 0 else 0
        dd_series.append({"date": e["date"], "dd": round(dd, 2)})
    dd_json = json.dumps(dd_series, ensure_ascii=False)

    html += f"""
<div class="footer">
  <p>🤖 量化策略回测报告 | 由 backtest_engine.py v8.0 自动生成</p>
  <p>⚠️ 过去表现不代表未来收益 | 仅供学习参考，不构成投资建议</p>
</div></div>

<script>
const navs = {navs_json};
const bench = {bench_json};
const dd = {dd_json};

// 权益曲线
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: navs.map(e => e.date.slice(5)),
    datasets: [
      {{ label: '策略组合', data: navs.map(e => e.nav), borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3, pointRadius: 0 }},
      {bench_dataset_html}
    ].filter(Boolean)
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e1e4e8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 20 }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v.toFixed(0) + '元' }} }}
    }}
  }}
}});

// 回撤曲线
new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{
    labels: dd.map(e => e.date.slice(5)),
    datasets: [{{ label: '回撤%', data: dd.map(e => e.dd), borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.2)', fill: true, tension: 0.3, pointRadius: 0 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e1e4e8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 20 }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v.toFixed(1) + '%' }}, max: 0 }}
    }}
  }}
}});
</script>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"回测报告已生成: {output_path}")
    logger.info(f"  累计收益: {metrics['total_return_pct']:+.1f}%")
    logger.info(f"  年化收益: {metrics['annual_return_pct']:+.1f}%")
    logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown_pct']:.1f}%")
    logger.info(f"  超额收益: {metrics['excess_return_pct']:+.1f}%")

    return output_path


if __name__ == "__main__":
    import sys, logging
    if "--report" in sys.argv:
        # v8.0: 一键回测报告
        start = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[1].startswith("--") else DEFAULT_START
        output = sys.argv[3] if len(sys.argv) > 3 else None
        path = generate_full_report(start_date=start, output_path=output)
        print(f"Report generated: {path}")
    else:
        # 简易测试
        start = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--start" else DEFAULT_START
        data = prepare_backtest_data(start)
        result = run_backtest(data)
        if result:
            print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
