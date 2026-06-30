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
COMMISSION = SYSTEM_CONFIG['commission_rate']

# v5.0: cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")
CACHE_TTL_HOURS = 6


# ============================================================
# 回测配置
# ============================================================
BACKTEST_POOL = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "512880": "证券ETF",
    "512760": "芯片ETF国泰",
    "515070": "人工智能ETF华夏",
    "512670": "国防ETF",
    "512400": "有色ETF",
    "512170": "医疗ETF",
    "512890": "红利低波ETF",
    "159928": "消费ETF",
    "513100": "纳指ETF国泰",
    "518850": "黄金ETF华夏",
    "159183": "新能源车ETF招商",
    "159659": "纳斯达克100ETF招商",
    "562500": "机器人ETF华夏",
}

DEFAULT_START = "2024-01-01"
DEFAULT_CAPITAL = 100000
MAX_HOLDINGS = SYSTEM_CONFIG['max_total_holdings']
SINGLE_WEIGHT = SYSTEM_CONFIG['max_single_weight']

# ============================================================
# 数据准备 (v5.0: API缓存)
# ============================================================
def _cache_key(start_date, end_date, pool_keys):
    codes = "-".join(sorted(pool_keys)[:10])
    return f"{start_date}_{end_date or 'now'}_{codes}"

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
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        lookback_dt = start_dt - timedelta(days=400)
        total_days = (datetime.now() - lookback_dt).days + 50
    except:
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

    logger.info(f"[回测] 开始: {equity_days[0]} ~ {equity_days[-1]} ({len(equity_days)}天)")
    logger.info(f"[回测] 初始资金: ¥{initial_capital:,.0f}")

    for day_idx, today in enumerate(equity_days):
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
        market_bullish = True
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

        # 卖出
        for code in list(holdings.keys()):
            if code not in top_codes:
                pos = holdings[code]
                sell_price = next((s["price"] for s in etf_scores if s["code"] == code), pos["cost"])
                sell_amount = pos["shares"] * sell_price * (1 - COMMISSION)
                cash += sell_amount
                pnl_pct = (sell_price / pos["cost"] - 1) * 100
                trades_log.append({
                    "date": today, "code": code, "name": pos["name"],
                    "action": "SELL", "price": sell_price,
                    "shares": pos["shares"], "amount": round(sell_amount, 2),
                    "reason": f"得分排名下滑 (浮盈{pnl_pct:+.1f}%)"
                })
                del holdings[code]

        # 买入
        needed_holdings = MAX_HOLDINGS - len(holdings)
        if needed_holdings > 0:
            buy_candidates = [s for s in etf_scores[:MAX_HOLDINGS] if s["code"] not in holdings]
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
                cost = shares * price * (1 + COMMISSION)
                if cost > cash:
                    affordable = int(cash / (price * (1 + COMMISSION)) / 100) * 100
                    if affordable < 100:
                        continue
                    shares = affordable
                    cost = shares * price * (1 + COMMISSION)
                cash -= cost
                holdings[candidate["code"]] = {"shares": shares, "cost": price, "name": candidate["name"], "price": price}
                trades_log.append({
                    "date": today, "code": candidate["code"], "name": candidate["name"],
                    "action": "BUY", "price": price, "shares": shares,
                    "amount": round(cost, 2),
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
            "initial_capital": initial_capital, "commission": COMMISSION,
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


if __name__ == "__main__":
    # 简易测试
    import sys, logging
    start = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--start" else DEFAULT_START
    data = prepare_backtest_data(start)
    result = run_backtest(data)
    if result:
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
