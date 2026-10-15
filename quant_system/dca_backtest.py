"""Compare contribution policies using identical cash flows and execution costs.

Input: {price_basis: "raw", corporate_actions_complete: true, targets: {code:
weight, CASH: weight}, instruments: {code: {lot_size, is_qdii, bars: [{date,
open, close, dividend_per_share: 0, split_factor: 1, premium_pct: ...}]}}}.

Prices must be unadjusted; dated cash distributions and share changes must be
complete. The simulator rejects adjusted cache prices as executable prices.
No optimizer or account files are read or written by compare_dca().
"""
import argparse
import json
import math
from pathlib import Path

from backtest_engine import COMMISSION_RATE, MIN_COMMISSION, SLIPPAGE_BPS
from cashflow_performance import cashflow_performance, date_key


def compare_dca(dataset, monthly_budget=1000, review_day=15, max_wait_sessions=10):
    if dataset.get('price_basis') != 'raw' or dataset.get('corporate_actions_complete') is not True:
        raise ValueError('需要未复权成交价及完整分红/份额变动记录；复权缓存不能作为真实成交数据')
    targets = dataset['targets']
    if any(not math.isfinite(v) or v < 0 for v in targets.values()) or abs(sum(targets.values())-1) > 1e-6:
        raise ValueError('目标权重必须为非负且合计100%')
    if not math.isfinite(monthly_budget) or monthly_budget <= 0 or not 1 <= review_day <= 28 or max_wait_sessions < 1:
        raise ValueError('预算、复核日或等待期无效')
    codes = [c for c, w in targets.items() if c != 'CASH' and w > 0]
    if not codes:
        raise ValueError('至少需要一个投资标的')
    series, lots = {}, {}
    for c in codes:
        item = dataset['instruments'][c]
        lot = item['lot_size']
        if not isinstance(lot, int) or lot < 1:
            raise ValueError('交易单位必须为正整数')
        lots[c] = lot
        series[c] = {}
        for row in item['bars']:
            day = date_key(row['date'])
            if day in series[c]:
                raise ValueError('重复交易日')
            for field in ('open', 'close', 'split_factor', 'dividend_per_share'):
                value = row[field]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('行情或公司行动数据无效')
                if field != 'dividend_per_share' and value <= 0:
                    raise ValueError('价格/份额倍数必须为正数')
            series[c][day] = row
    dates = sorted(set.union(*(set(series[c]) for c in codes)))
    # Explicitly reject missing sessions; do not quietly join different dates or carry stale prices.
    if any(set(series[c]) != set(dates) for c in codes):
        raise ValueError('各标的交易日期不齐，需先核对交易日和停牌；不允许按行号拼接')
    if len(dates) < 22:
        raise ValueError('至少需要22个共同交易日')

    results = {}
    for policy in ('fixed_date', 'underweight', 'bounded_timing'):
        cash, contributions, fees, distributions = 0.0, 0.0, 0.0, 0.0
        positions = {c: 0 for c in codes}
        spent_by_code = {c: 0.0 for c in codes}
        closes = {c: [] for c in codes}
        reports, flows, trades = [], [], []
        pending, active_month, month_spent, bought, waiting = None, None, 0.0, False, 0
        idle_total = 0.0
        for index, day in enumerate(dates):
            if day[:6] != active_month:
                active_month = day[:6]
                cash += monthly_budget
                contributions += monthly_budget
                flows.append({'date': day, 'type': 'deposit', 'amount': monthly_budget})
                month_spent, bought, waiting, pending = 0.0, False, 0, None
            for c in codes:
                row = series[c][day]
                # Dividend amounts are per pre-split share, on cash payment date.
                dividend = positions[c] * row['dividend_per_share']
                cash += dividend
                distributions += dividend
                positions[c] *= row['split_factor']
            if pending is not None and not bought:
                c = pending
                raw_price = series[c][day]['open']
                # Current execution quote can block a purchase; cannot select using future close.
                premium = series[c][day].get('premium_pct')
                qdii = dataset['instruments'][c].get('is_qdii', False)
                valid_premium = premium is not None and math.isfinite(premium)
                if (not qdii or valid_premium) and (not valid_premium or premium < 2):
                    price = raw_price * (1 + SLIPPAGE_BPS / 10000)
                    cash_target = targets.get('CASH', 0) * (cash + sum(positions[k] * series[k][day]['open'] for k in codes))
                    available = max(0, min(cash - cash_target, monthly_budget - month_spent))
                    units = int(max(0, available - MIN_COMMISSION) / (price * (1 + COMMISSION_RATE)) / lots[c]) * lots[c]
                    if units >= lots[c]:
                        amount = units * price
                        commission = max(MIN_COMMISSION, amount * COMMISSION_RATE)
                        cost = amount + commission
                        if cost <= available + 1e-8:
                            cash -= cost
                            positions[c] += units
                            spent_by_code[c] += cost
                            month_spent += cost
                            fees += commission + units * (price - raw_price)
                            bought = True
                            trades.append({'date': day, 'code': c, 'shares': units,
                                           'price': raw_price, 'total_cost': cost,
                                           'signal_date': dates[index-1]})
                pending = None
            total = cash + sum(positions[c] * series[c][day]['close'] for c in codes)
            reports.append({'date': day, 'portfolio': {'total_assets': total}})
            idle_total += cash
            for c in codes:
                # Split-normalized and dividend-inclusive price moves for the timing signal.
                previous = series[c][dates[index-1]]['close'] if index else series[c][day]['close']
                factor = (series[c][day]['close'] * series[c][day]['split_factor'] + series[c][day]['dividend_per_share']) / previous
                closes[c].append((closes[c][-1] if closes[c] else 1.0) * (factor if index else 1))
            if bought or int(day[-2:]) < review_day or index == len(dates)-1:
                continue
            if policy == 'fixed_date':
                # Predetermined funding allocation: ignore market appreciation.
                c = max(codes, key=lambda k: contributions * targets[k] - spent_by_code[k])
            else:
                c = max(codes, key=lambda k: total * targets[k] - positions[k] * series[k][day]['close'])
            waiting += 1
            recent = closes[c][-20:]
            pullback = len(recent) == 20 and recent[-1] <= sum(recent)/20
            if policy != 'bounded_timing' or pullback or waiting >= max_wait_sessions:
                pending = c
        perf = cashflow_performance(reports, flows)
        results[policy] = {'final_wealth': total, 'contributions': contributions,
                           'profit': total-contributions, 'fees': fees, 'cash_dividends': distributions,
                           'average_idle_cash': idle_total / len(dates), 'final_cash': cash,
                           'trades': trades, 'performance': perf}
    return {'start': dates[0], 'end': dates[-1], 'sessions': len(dates), 'monthly_budget': monthly_budget,
            'research_only': True, 'results': results,
            'limitations': ['同一月入金、同标的和费率，收盘决策/次日开盘成交，每月最多买入一笔；不卖出',
                            '固定日期方案按累计投入缺口选择；再平衡方案按市值缺口选择；择时方案最多等待指定交易日',
                            '未投入余额留现金但不自动提高下月预算；月末未执行信号作废',
                            '交易日缺失会拒绝运行；现金利息默认0；历史溢价缺失的QDII禁止成交',
                            '短样本仅检验流程，不能证明择时优于定投；需多市场周期及样本外验证']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_dca(json.loads(args.dataset.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
