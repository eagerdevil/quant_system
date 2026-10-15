"""Explainable contribution reviews. Scores are priorities, never return forecasts."""
import math

from backtest_engine import COMMISSION_RATE, MIN_COMMISSION, SLIPPAGE_BPS


def purchase_example(price, lot_size, available, spread_pct=0):
    """Size whole lots including commission/slippage, not just nominal amount."""
    empty = {'shares': 0, 'amount': 0.0, 'commission': 0.0, 'slippage': 0.0,
             'cash_required': 0.0, 'cash_left': available, 'cost_fraction': None,
             'spread_cost_estimate': None, 'reference_price': price}
    try:
        price, lot, available = float(price), float(lot_size), float(available)
        if not all(math.isfinite(v) for v in (price, lot, available)) or price <= 0 or lot < 1 or lot != int(lot) or available <= 0:
            return empty
        lot = int(lot)
        shares = max(0, int(available / (price * lot))) * lot
        while shares:
            amount = shares * price
            commission = max(MIN_COMMISSION, amount * COMMISSION_RATE)
            slip = amount * SLIPPAGE_BPS / 10000
            if amount + commission + slip <= available + 1e-9:
                spread_cost = amount * max(0, spread_pct or 0) / 200
                return {'shares': shares, 'amount': round(amount, 4),
                        'commission': round(commission, 4), 'slippage': round(slip, 4),
                        'cash_required': round(amount + commission + slip, 4),
                        'cash_left': round(available - amount - commission - slip, 4),
                        'cost_fraction': (commission + slip + spread_cost) / amount,
                        'spread_cost_estimate': round(spread_cost, 4), 'reference_price': price,
                        'note': '仅为整数份额示例；费用含佣金与假设滑点，经济成本另含半边价差估计；不提交委托'}
            shares -= lot
    except (ValueError, TypeError, OverflowError):
        return empty
    return empty


def disclosed_overlap(code, catalog, holdings, total_assets):
    """Known lower bound from disclosed direct stocks, not an ETF look-through claim."""
    meta = catalog.get(code, {})
    weights = {r['code']: r['weight_pct'] for r in meta.get('disclosed_stock_holdings', [])}
    matches, weighted = [], 0.0
    for h in holdings:
        other = h['code']
        if other == code or h.get('value', 0) <= 0:
            continue
        held_meta = catalog.get(other, {})
        same_index = bool(meta.get('index_name') and meta['index_name'] == held_meta.get('index_name'))
        held_weights = {r['code']: r['weight_pct'] for r in held_meta.get('disclosed_stock_holdings', [])}
        overlap = (100.0 if same_index else weights.get(other, 0) if held_meta.get('instrument_type') == 'stock'
                   else sum(min(w, held_weights.get(c, 0)) for c, w in weights.items()))
        if overlap > 0:
            exposure = h['value'] / total_assets if total_assets > 0 else 0
            weighted += exposure * overlap
            matches.append({'code': other, 'name': h.get('name', other),
                            'overlap_pct': round(overlap, 3),
                            'basis': '同一跟踪标的' if same_index else '已披露直接股票交集下限'})
    return {'weighted_overlap_pct': round(min(100, weighted), 3), 'matches': matches,
            'coverage': 'partial', 'note': '只比较已披露直接股票或同一标的；未穿透境外目标ETF，未匹配不等于无重叠'}


def contribution_priority(observation, score, config, overlap):
    example = observation['purchase_example']
    gap = observation['allocation_gap_pct']
    premium = observation['premium_pct']
    cost = example['cost_fraction']
    rsi = (score.get('indicators') or {}).get('rsi')
    r5 = (score.get('returns') or {}).get('r5d')
    ma20 = (score.get('vs_ma') or {}).get('pct_ma20')
    hot = any(v is not None and v >= limit for v, limit in ((rsi,70), (r5,8), (ma20,8)))
    components = {
        '配置缺口': min(40, max(0, gap) * 2),
        '交易成本': 0 if cost is None else max(0, 20 * (1 - cost / config['max_fee_fraction'])),
        '溢价成本': 0 if premium is None else max(0, min(15, 15 * (1 - max(0,premium)/config['max_premium_pct']))),
        '已知重复暴露': max(0, 15 * (1 - overlap['weighted_overlap_pct']/100)),
        '短期时机': 0 if hot else 10,
    }
    reasons = [f'配置缺口{gap:+.1f}个百分点',
               '短期偏热，复核基础计划时保持克制' if hot else '未触发短期过热参考线']
    if example['shares']:
        reasons.append(f"示例{example['shares']}份，经济成本约{example['cost_fraction']:.2%}")
    if overlap['matches']:
        reasons.append('与已有持仓存在已披露重叠')
    else:
        reasons.append('已披露范围未发现重叠，穿透资料仍不完整')
    return {'score': round(sum(components.values()), 2), 'components': components,
            'overheated': hot, 'reasons': reasons,
            'note': '按固定规则排列复核优先级，不是收益预测、胜率或自动买入指令'}


def report_changes(observations, allocations, holdings, history, today, stale_codes):
    previous = [r for r in history if str(r.get('_date') or r.get('date','')).replace('-','') < today]
    if not previous:
        return {'previous_date': None, 'items': ['首次记录，后续日报将比较变化'], 'events': []}
    prev = max(previous, key=lambda r: str(r.get('_date') or r.get('date','')).replace('-',''))
    prev_date = str(prev.get('_date') or prev.get('date','')).replace('-','')
    scores = {s['code']: s for s in prev.get('scores', [])}
    events = []
    for o in observations:
        old = scores.get(o['code'], {})
        before, after = (old.get('premium_info') or {}).get('premium_pct'), o['premium_pct']
        old_source = (old.get('premium_info') or {}).get('data_source')
        old_nav_date = (old.get('premium_info') or {}).get('nav_date')
        comparable = old_source == o['premium_source'] and old_source is not None
        # A stale official NAV denominator can create fake premium movements.
        comparable = comparable and (old_source == 'iopv' or old_nav_date == o.get('nav_date'))
        if o['code'] not in stale_codes and comparable and before is not None and after is not None and abs(after-before)>=0.5:
            events.append({'kind':'premium', 'code':o['code'], 'text':f"{o['name']}溢价由{before:.2f}%变为{after:.2f}%（同口径）"})
    old_allocs = {a['bucket']: a for a in prev.get('long_term', {}).get('allocations', [])}
    if not stale_codes:
        for a in allocations:
            old = old_allocs.get(a['bucket'])
            if old and abs(a['actual_pct']-old['actual_pct'])>=2:
                events.append({'kind':'allocation', 'text':f"{a['name']}账户占比变化{a['actual_pct']-old['actual_pct']:+.1f}个百分点；可能含入金或交易影响"})
    old_holdings = {h['code']: h for h in prev.get('portfolio',{}).get('holdings',[])}
    for h in holdings:
        old = old_holdings.get(h['code'])
        if old and h['code'] not in stale_codes and old.get('price',0)>0 and abs(h.get('price',0)/old['price']-1)>=0.02:
            events.append({'kind':'holding', 'code':h['code'], 'text':f"{h['name']}较上次报告价格变化{h['price']/old['price']-1:+.1%}"})
    return {'previous_date': prev_date, 'events': events[:8],
            'items': [x['text'] for x in events[:8]] or ['未发现达到展示阈值的重大变化；数据不全的项目不作比较']}
