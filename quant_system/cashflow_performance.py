"""Cash-flow-aware observations; no account writes or network access.

Modified Dietz per observed interval, geometrically linked. With daily closing
valuations this approximates TWR; intraday flow times are unavailable. Never call
it exact TWR. Unknown/placeholder flows fail closed instead of fabricating NAV.
"""
import math
from datetime import datetime


def account_performance(reports, portfolio, placeholder_policy='block'):
    """Accepted accounting differences are metadata, never fabricated cash flows."""
    flows = portfolio.get('_cash_flows', [])
    legacy = [f for f in flows if '占位' in str(f.get('note',''))] if placeholder_policy == 'reference_only' else []
    result = cashflow_performance(reports, [f for f in flows if f not in legacy])
    accepted = [a for a in portfolio.get('_reconciliation_adjustments', [])
                if a.get('status') == 'accepted_unverified']
    if accepted or legacy:
        amount = sum(float(a['amount']) for a in accepted + legacy)
        result['reconciliation_note'] = (
            f'历史账务差额{amount:.2f}元按参考口径处理，原因及发生日期未核实；未当作真实出入金或已确认手续费。'
            '账户总盈亏继续使用券商已核验本金，历史区间收益为参考。'
        )
        if result.get('available'):
            result['method_note'] += '；' + result['reconciliation_note']
    return result


def date_key(value):
    key = str(value).replace('-', '')[:8]
    datetime.strptime(key, '%Y%m%d')
    return key


def cashflow_performance(reports, flows, reconciliation_pending=False):
    if reconciliation_pending or any(
        f.get('pending') or '占位' in str(f.get('note', '')) for f in flows
    ):
        return {'available': False, 'reason': '资金流水存在待核对差额，暂停收益率与回撤结论'}
    try:
        observations = {}
        for r in reports:
            p = r.get('portfolio', {})
            if p.get('stale_codes'):
                continue
            value = float(p.get('total_assets', 0))
            if value > 0 and math.isfinite(value):
                observations[date_key(r.get('_date') or r.get('date'))] = value
        points = sorted(observations.items())
        if len(points) < 2:
            return {'available': False, 'reason': '至少需要两次有效账户估值'}
        parsed = []
        for f in flows:
            amount = float(f['amount'])
            if not math.isfinite(amount) or amount < 0 or f['type'] not in ('deposit', 'withdraw'):
                raise ValueError('invalid flow')
            parsed.append((date_key(f['date']), amount if f['type'] == 'deposit' else -amount))
        nav, peak, max_dd, net_flow = 1.0, 1.0, 0.0, 0.0
        curve = [{'date': points[0][0], 'nav': nav}]
        for (d0, v0), (d1, v1) in zip(points, points[1:]):
            start = datetime.strptime(d0, '%Y%m%d')
            end = datetime.strptime(d1, '%Y%m%d')
            duration = (end - start).days
            interval = [(d, a) for d, a in parsed if d0 < d <= d1]
            cf = sum(a for _, a in interval)
            # Mid-day convention for a dated flow; its valuation is end-of-day.
            weighted = sum(a * ((end - datetime.strptime(d, '%Y%m%d')).days + 0.5) / duration
                           for d, a in interval)
            denominator = v0 + weighted
            if denominator <= 0:
                raise ValueError('nonpositive invested capital')
            ret = (v1 - v0 - cf) / denominator
            if not math.isfinite(ret) or ret <= -1:
                raise ValueError('invalid period return')
            nav *= 1 + ret
            peak = max(peak, nav)
            max_dd = min(max_dd, nav / peak - 1)
            net_flow += cf
            curve.append({'date': d1, 'nav': nav})
        investor_flows = [(points[0][0], -points[0][1])]
        investor_flows += [(d, -a) for d, a in parsed if points[0][0] < d <= points[-1][0]]
        investor_flows.append((points[-1][0], points[-1][1]))
        return {
            'available': True, 'method': 'linked_modified_dietz',
            'method_note': '按实际资金流计算区间Modified Dietz并连乘；日内时点未知，为近似时间加权收益',
            'start_date': points[0][0], 'end_date': points[-1][0],
            'return_pct': (nav - 1) * 100, 'drawdown_pct': (nav / peak - 1) * 100,
            'max_drawdown_pct': max_dd * 100,
            'profit': points[-1][1] - points[0][1] - net_flow,
            'xirr_pct': xirr(investor_flows), 'curve': curve,
            'xirr_note': '跟踪区间年化资金加权收益；短区间不宜外推，非终身收益',
        }
    except (TypeError, ValueError, KeyError, OverflowError):
        return {'available': False, 'reason': '估值或资金流水无效，暂停收益率与回撤结论'}


def xirr(flows):
    """Conservative XIRR: return None for short spans or multiple sign changes."""
    grouped = {}
    for d, a in flows:
        grouped[d] = grouped.get(d, 0) + a
    points = [(datetime.strptime(d, '%Y%m%d'), a) for d, a in sorted(grouped.items()) if abs(a) > 1e-9]
    if len(points) < 2 or (points[-1][0] - points[0][0]).days < 365:
        return None
    signs = [a > 0 for _, a in points]
    if sum(a != b for a, b in zip(signs, signs[1:])) != 1:
        return None
    origin = points[0][0]
    def npv(log_rate):
        return sum(a * math.exp(-log_rate * (d - origin).days / 365.0) for d, a in points)
    lo, hi = -5.0, 5.0
    try:
        if npv(lo) * npv(hi) > 0:
            return None
        for _ in range(100):
            mid = (lo + hi) / 2
            if npv(lo) * npv(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return math.expm1((lo + hi) / 2) * 100
    except OverflowError:
        return None
