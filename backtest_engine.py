#!/usr/bin/env python
"""
回测引擎 v1.0
============
基于历史数据模拟交易，评估15因子策略表现

核心原则：
  - 逐日推进，只用当天之前的数据（杜绝未来函数）
  - 按收盘价成交，扣除 0.05% 手续费
  - 基准对比：沪深300买入持有

用法：
  python backtest_engine.py                    # 默认参数运行
  python backtest_engine.py --start 20240101   # 指定起始日
  python backtest_engine.py --cap 100000       # 指定初始资金
"""
import json, sys, os, io, math, time
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_etf_kline, fetch_index_daily, KEY_ETFS
from quant_engine import (
    compute_indicators, score_factors, sma, FACTOR_MAX,
    CURRENT_WEIGHTS, FACTOR_NAMES
)

TODAY = datetime.now().strftime("%Y%m%d")
COMMISSION = 0.0005  # 万5手续费

# ============================================================
# 回测配置
# ============================================================
# 回测 ETF 池（选流动性好的代表性ETF）
BACKTEST_POOL = {
    "510300": "沪深300ETF",       # 宽基
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
}

# 默认回测参数
DEFAULT_START = "2024-01-01"
DEFAULT_CAPITAL = 100000  # 初始资金 10万
MAX_HOLDINGS = 5           # 最多持仓数
SINGLE_WEIGHT = 0.20       # 单只仓位上限 20%

# ============================================================
# 数据准备
# ============================================================
def prepare_backtest_data(start_date, end_date=None, pool=None):
    """
    批量拉取回测所需全部数据。
    返回: {
        etfs: {code: {name, klines: [{date, close, high, low, volume}...]}},
        benchmark: [{date, close}...],
        trading_days: [date_str...]
    }
    """
    if pool is None:
        pool = BACKTEST_POOL
    if end_date is None:
        end_date = TODAY

    print(f"[回测] 拉取数据: {start_date} ~ {end_date}", file=sys.stderr)

    # 计算需要多少天数据（回测起始日 + 250天lookback）
    try:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        lookback_dt = start_dt - timedelta(days=400)  # 多拉一些保底
        total_days = (datetime.now() - lookback_dt).days + 50
    except:
        total_days = 600

    # 拉取 ETF K 线
    raw_etfs = {}
    codes = list(pool.keys())
    print(f"[回测] 拉取 {len(codes)} 只ETF K线 (每只{total_days}天)...", file=sys.stderr)

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
            if (i+1) % 5 == 0:
                print(f"  [{i+1}/{len(codes)}] {code} {name} OK ({len(klines)}条)", file=sys.stderr)
        else:
            print(f"  [{i+1}/{len(codes)}] {code} {name} FAIL", file=sys.stderr)

    # 过滤数据不足的ETF（至少300条K线才参与回测）
    min_klines = 300
    valid_etfs = {}
    dropped = []
    for code, data in raw_etfs.items():
        if len(data["klines"]) >= min_klines:
            valid_etfs[code] = data
        else:
            dropped.append(f"{code}({data['name']}:{len(data['klines'])}条)")

    if dropped:
        print(f"[回测] 剔除数据不足的ETF: {', '.join(dropped)}", file=sys.stderr)
    print(f"[回测] 有效ETF: {len(valid_etfs)}只 (要求≥{min_klines}条K线)", file=sys.stderr)
    print(f"[回测] 拉取沪深300指数 ({total_days}天)...", file=sys.stderr)
    benchmark_klines = fetch_index_daily("000300", days=total_days)
    if benchmark_klines:
        benchmark = [{"date": k["date"], "close": k["close"]} for k in benchmark_klines]
    else:
        benchmark = []

    # 确定交易日列表（用数据最全的ETF的日期）
    trading_days = _get_common_trading_days(valid_etfs, start_date)

    print(f"[回测] 数据就绪: {len(valid_etfs)}只ETF, {len(trading_days)}个交易日", file=sys.stderr)
    return {
        "etfs": valid_etfs,
        "benchmark": benchmark,
        "trading_days": trading_days,
    }

def _get_common_trading_days(valid_etfs, start_date, min_etfs=5):
    """
    从ETF数据中提取多数ETF共有的交易日。
    不强求所有ETF都有数据，只要min_etfs只以上有即可。
    """
    if not valid_etfs:
        return []

    # 统计每天有多少只ETF有数据
    date_count = {}
    for code, data in valid_etfs.items():
        for k in data["klines"]:
            d = k["date"]
            date_count[d] = date_count.get(d, 0) + 1

    # 保留至少min_etfs只ETF有数据的日期，且在起始日之后
    result = sorted(d for d, cnt in date_count.items()
                    if cnt >= min_etfs and d >= start_date)
    return result

# ============================================================
# 回测核心
# ============================================================
def run_backtest(data, initial_capital=DEFAULT_CAPITAL):
    """
    执行回测。

    参数:
        data: prepare_backtest_data() 的输出
        initial_capital: 初始资金

    返回: {
        equity_curve: [{date, nav, cash, holdings_value, benchmark_nav}],
        trades: [{date, code, name, action, price, shares, amount, reason}],
        metrics: {total_return, annual_return, sharpe, max_dd, ...},
        summary: str
    }
    """
    valid_etfs = data["etfs"]
    trading_days = data["trading_days"]
    benchmark_data = data.get("benchmark", [])

    # 构建基准查找表
    bench_map = {}
    for b in benchmark_data:
        bench_map[b["date"]] = b["close"]

    # 跳过前250天（用于指标计算）
    if len(trading_days) < 251:
        print("[回测] 交易日不足250天，无法回测", file=sys.stderr)
        return None

    equity_days = trading_days[250:]  # 实际回测期

    # 初始状态
    cash = initial_capital
    holdings = {}  # {code: {shares, cost, name}}
    equity_curve = []
    trades_log = []

    initial_bench = None
    for d in equity_days:
        if d in bench_map:
            initial_bench = bench_map[d]
            break

    print(f"[回测] 开始: {equity_days[0]} ~ {equity_days[-1]} ({len(equity_days)}天)", file=sys.stderr)
    print(f"[回测] 初始资金: ¥{initial_capital:,.0f}", file=sys.stderr)

    for day_idx, today in enumerate(equity_days):
        # 1. 获取今天的数据切片
        etf_scores = []
        for code, edata in valid_etfs.items():
            # 找到今天的索引位置
            all_dates = [k["date"] for k in edata["klines"]]
            if today not in all_dates:
                continue
            pos = all_dates.index(today)

            # 切片到今天为止的K线（包含今天，用于计算指标）
            kline_slice = edata["klines"][:pos+1]
            if len(kline_slice) < 30:
                continue

            closes = [k["close"] for k in kline_slice]
            highs = [k["high"] for k in kline_slice]
            lows = [k["low"] for k in kline_slice]
            volumes = [k["volume"] for k in kline_slice]

            # 计算指标和因子得分
            try:
                indicators = compute_indicators(closes, highs, lows, volumes)
                factors = score_factors(indicators)

                # 加权评分
                weights = CURRENT_WEIGHTS
                weighted_sum = sum(factors[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
                max_weighted = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
                score = round(weighted_sum / max_weighted * 100) if max_weighted > 0 else 50

                etf_scores.append({
                    "code": code,
                    "name": edata["name"],
                    "score": score,
                    "price": closes[-1],
                    "rsi": round(indicators["rsi"], 1),
                })
            except Exception:
                continue

        if not etf_scores:
            # 无数据日，沿用前一日净值
            if equity_curve:
                prev = equity_curve[-1]
                equity_curve.append(dict(prev, date=today))
            continue

        # 2. 简化大盘择时（只用价格信号）
        # 用沪深300ETF(510300)作为市场代表
        hs300_etf = valid_etfs.get("510300")
        market_bullish = True
        if hs300_etf:
            hs300_closes = [k["close"] for k in hs300_etf["klines"]]
            all_dates = [k["date"] for k in hs300_etf["klines"]]
            if today in all_dates:
                pos = all_dates.index(today)
                hs300_slice = hs300_closes[:pos+1]
                if len(hs300_slice) >= 60:
                    ma20 = sma(hs300_slice, 20)
                    ma60 = sma(hs300_slice, 60)
                    ma60_prev = sma(hs300_slice[:-1], 60) if len(hs300_slice) > 60 else ma60
                    s1 = hs300_slice[-1] > ma20      # 在20日线上方
                    s2 = ma60 > ma60_prev             # 60日线向上
                    bull_signals = sum([s1, s2])
                    if bull_signals == 0:
                        market_bullish = 0.20   # 双熊 → 20%仓位
                    elif bull_signals == 1:
                        market_bullish = 0.50   # 单牛 → 50%仓位
                    else:
                        market_bullish = 1.0    # 双牛 → 满仓

        target_invested = initial_capital * market_bullish

        # 3. 按得分排序
        etf_scores.sort(key=lambda x: x["score"], reverse=True)
        top_codes = set(s["code"] for s in etf_scores[:MAX_HOLDINGS])

        # 4. 卖出：不在top5的持仓
        for code in list(holdings.keys()):
            if code not in top_codes:
                pos = holdings[code]
                sell_price = next((s["price"] for s in etf_scores if s["code"] == code), pos["cost"])
                sell_amount = pos["shares"] * sell_price * (1 - COMMISSION)
                cash += sell_amount
                pnl_pct = (sell_price / pos["cost"] - 1) * 100

                trades_log.append({
                    "date": today,
                    "code": code,
                    "name": pos["name"],
                    "action": "SELL",
                    "price": sell_price,
                    "shares": pos["shares"],
                    "amount": round(sell_amount, 2),
                    "reason": f"得分排名下滑 (浮盈{pnl_pct:+.1f}%)"
                })
                del holdings[code]

        # 5. 买入：在top5但未持仓且资金充足
        needed_holdings = MAX_HOLDINGS - len(holdings)
        if needed_holdings > 0:
            buy_candidates = [s for s in etf_scores[:MAX_HOLDINGS] if s["code"] not in holdings]
            per_slot_cash = cash / max(needed_holdings, 1)

            for candidate in buy_candidates[:needed_holdings]:
                budget = min(per_slot_cash, initial_capital * SINGLE_WEIGHT)

                # 检查是否满足仓位目标
                current_invested = sum(h["shares"] * h.get("price", h["cost"]) for h in holdings.values())
                if current_invested >= target_invested:
                    break

                price = candidate["price"]
                if price <= 0:
                    continue

                shares = int(budget / price / 100) * 100  # 整手
                if shares < 100:
                    continue

                cost = shares * price * (1 + COMMISSION)
                if cost > cash:
                    # 调整到可用资金
                    affordable = int(cash / (price * (1 + COMMISSION)) / 100) * 100
                    if affordable < 100:
                        continue
                    shares = affordable
                    cost = shares * price * (1 + COMMISSION)

                cash -= cost
                holdings[candidate["code"]] = {
                    "shares": shares,
                    "cost": price,
                    "name": candidate["name"],
                    "price": price,
                }

                trades_log.append({
                    "date": today,
                    "code": candidate["code"],
                    "name": candidate["name"],
                    "action": "BUY",
                    "price": price,
                    "shares": shares,
                    "amount": round(cost, 2),
                    "reason": f"评分{candidate['score']}分 排名TOP{MAX_HOLDINGS}"
                })

        # 6. 更新持仓市值
        holdings_value = 0
        for code, pos in holdings.items():
            current_price = next((s["price"] for s in etf_scores if s["code"] == code), pos.get("price", pos["cost"]))
            pos["price"] = current_price
            holdings_value += pos["shares"] * current_price

        nav = cash + holdings_value

        # 7. 基准净值
        bench_close = bench_map.get(today)
        if bench_close:
            if initial_bench is None:
                initial_bench = bench_close
            bench_nav = bench_close / initial_bench * initial_capital if initial_bench else initial_capital
        else:
            bench_nav = equity_curve[-1]["benchmark_nav"] if equity_curve else initial_capital

        equity_curve.append({
            "date": today,
            "nav": round(nav, 2),
            "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2),
            "holdings_count": len(holdings),
            "benchmark_nav": round(bench_nav, 2),
        })

        if (day_idx + 1) % 50 == 0:
            ret = (nav / initial_capital - 1) * 100
            print(f"  [{day_idx+1}/{len(equity_days)}] {today} | 净值: ¥{nav:,.0f} | 收益: {ret:+.1f}%", file=sys.stderr)

    # 计算绩效指标
    metrics = _calc_metrics(equity_curve, initial_capital, trades_log)

    return {
        "equity_curve": equity_curve,
        "trades": trades_log,
        "metrics": metrics,
        "config": {
            "start_date": equity_days[0],
            "end_date": equity_days[-1],
            "initial_capital": initial_capital,
            "commission": COMMISSION,
            "pool_size": len(valid_etfs),
            "trading_days": len(equity_days),
        }
    }

# ============================================================
# 绩效计算
# ============================================================
def _calc_metrics(equity_curve, initial_capital, trades):
    """计算全套绩效指标"""
    if len(equity_curve) < 2:
        return {}

    navs = [e["nav"] for e in equity_curve]
    bench_navs = [e["benchmark_nav"] for e in equity_curve]
    dates = [e["date"] for e in equity_curve]

    final_nav = navs[-1]
    total_return = (final_nav / initial_capital - 1) * 100

    # 日收益率
    daily_returns = []
    for i in range(1, len(navs)):
        if navs[i-1] > 0:
            daily_returns.append(navs[i] / navs[i-1] - 1)

    # 年化收益率
    n_days = len(daily_returns)
    n_years = n_days / 252
    annual_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    # 夏普比率
    if daily_returns:
        avg_daily = sum(daily_returns) / len(daily_returns)
        std_daily = math.sqrt(sum((r - avg_daily)**2 for r in daily_returns) / (len(daily_returns)-1)) if len(daily_returns) > 1 else 0.01
        if std_daily > 0:
            sharpe = (avg_daily / std_daily) * math.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # 最大回撤 & 回撤持续天数
    peak = navs[0]
    max_dd = 0
    max_dd_start = None
    max_dd_end = None
    max_dd_days = 0
    current_dd_start = None
    in_dd = False

    for i, nav in enumerate(navs):
        if nav > peak:
            peak = nav
            if in_dd:
                dd_days = i - current_dd_start
                if dd_days > max_dd_days:
                    max_dd_days = dd_days
                in_dd = False
        else:
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_start = dates[i]
            if not in_dd:
                in_dd = True
                current_dd_start = i

    # 卡玛比率
    calmar = annual_return / (max_dd * 100) if max_dd > 0 else 0

    # 基准对比
    bench_return = (bench_navs[-1] / initial_capital - 1) * 100 if bench_navs else 0
    alpha = total_return - bench_return

    # 胜率
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    win_count = 0
    for sell in sell_trades:
        # 匹配同代码的买入
        code_buys = [b for b in buy_trades if b["code"] == sell["code"]]
        if code_buys:
            avg_buy_price = sum(b["price"] for b in code_buys) / len(code_buys)
            if sell["price"] > avg_buy_price:
                win_count += 1
    win_rate = (win_count / len(sell_trades) * 100) if sell_trades else 0

    # 波动率
    if daily_returns:
        volatility = math.sqrt(sum((r - avg_daily)**2 for r in daily_returns) / (len(daily_returns)-1)) * math.sqrt(252) * 100 if len(daily_returns) > 1 else 0
    else:
        volatility = 0

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "max_drawdown_days": max_dd_days,
        "calmar": round(calmar, 2),
        "volatility": round(volatility, 2),
        "benchmark_return": round(bench_return, 2),
        "alpha": round(alpha, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "sell_trades": len(sell_trades),
        "buy_trades": len(buy_trades),
    }

# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="量化策略回测引擎")
    parser.add_argument("--start", default=DEFAULT_START, help="回测起始日 YYYYMMDD")
    parser.add_argument("--end", default=TODAY, help="回测截止日 YYYYMMDD")
    parser.add_argument("--cap", type=float, default=DEFAULT_CAPITAL, help="初始资金")
    parser.add_argument("--pool", default=None, help="ETF池，逗号分隔（默认15只）")
    parser.add_argument("--output", default=None, help="结果JSON输出路径")

    args = parser.parse_args()

    # 解析ETF池
    pool = BACKTEST_POOL
    if args.pool:
        codes = args.pool.split(",")
        pool = {c.strip(): KEY_ETFS.get(c.strip(), c.strip()) for c in codes}

    # 准备数据
    print("=" * 60, file=sys.stderr)
    print("  量化策略回测引擎 v1.0", file=sys.stderr)
    print(f"  回测区间: {args.start} → {args.end}", file=sys.stderr)
    print(f"  初始资金: ¥{args.cap:,.0f}", file=sys.stderr)
    print(f"  ETF池: {len(pool)}只", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    data = prepare_backtest_data(args.start, args.end, pool)

    if len(data["trading_days"]) < 251:
        print("[回测] 错误: 交易日不足250天，请缩短回测区间或更换ETF池", file=sys.stderr)
        return

    # 执行回测
    result = run_backtest(data, args.cap)

    if not result:
        print("[回测] 回测失败", file=sys.stderr)
        return

    # 输出结果
    m = result["metrics"]
    print("\n" + "=" * 60)
    print("  📊 回测结果")
    print("=" * 60)
    print(f"  回测区间: {result['config']['start_date']} → {result['config']['end_date']}")
    print(f"  交易天数: {result['config']['trading_days']}")
    print(f"  初始资金: ¥{args.cap:,.0f}")
    print(f"  最终净值: ¥{result['equity_curve'][-1]['nav']:,.0f}")
    print(f"  {'─'*50}")
    print(f"  累计收益: {m['total_return']:+.2f}%")
    print(f"  年化收益: {m['annual_return']:+.2f}%")
    print(f"  基准收益: {m['benchmark_return']:+.2f}%  (沪深300)")
    print(f"  超额收益: {m['alpha']:+.2f}%")
    print(f"  {'─'*50}")
    print(f"  夏普比率: {m['sharpe']:.2f}")
    print(f"  卡玛比率: {m['calmar']:.2f}")
    print(f"  年化波动: {m['volatility']:.2f}%")
    print(f"  最大回撤: -{m['max_drawdown']:.2f}%")
    print(f"  回撤天数: {m['max_drawdown_days']}天")
    print(f"  {'─'*50}")
    print(f"  交易次数: {m['total_trades']} (买入{m['buy_trades']} / 卖出{m['sell_trades']})")
    print(f"  胜率: {m['win_rate']:.1f}%")
    print(f"  {'─'*50}")
    print(f"  手续费率: {COMMISSION*100:.2f}%")
    print("=" * 60)

    # 保存结果
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"backtest_{TODAY}.json"
    )
    # 精简版保存（方便dashboard读取）
    save_data = {
        "generated": TODAY,
        "config": result["config"],
        "metrics": result["metrics"],
        "equity_curve": result["equity_curve"],
        "trades": result["trades"],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[回测] 结果已保存: {output_path}", file=sys.stderr)

    return result

if __name__ == "__main__":
    main()
