"""Compact daily reading view shared by real and paper account reports."""
import html
import math


def number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def fmt(value, suffix='', signed=False):
    value = number(value)
    return '—' if value is None else (f'{value:+.1f}' if signed else f'{value:,.1f}') + suffix


def account_total(report):
    p = report.get('portfolio') or {}
    if report.get('type') == 'paper':
        label, value = '模拟盘总盈亏', p.get('total_pnl_all')
    elif number(p.get('lifetime_pnl')) is not None:
        label, value = '累计总盈亏（2024年入市至今）', p['lifetime_pnl']
    else:
        label, value = '今年总盈亏', p.get('total_pnl_all')
    return label, '—' if number(value) is None else f'{value:+.2f}元'


def market_snapshot(indices, day):
    """Keep only date-verified closing changes; never label cached prices as today."""
    result = []
    for code in ('000001', '399001', '399006', '000300'):
        info = indices.get(code, {})
        bars = info.get('data', [])
        bars = sorted((b for b in bars if str(b.get('date', '')).replace('-', '') <= day),
                      key=lambda b: str(b.get('date', '')).replace('-', ''))
        if len(bars) < 2:
            continue
        last, prev = bars[-1], bars[-2]
        close, previous = number(last.get('close')), number(prev.get('close'))
        if close is None or previous is None or previous <= 0:
            continue
        result.append({'code': code, 'name': info.get('name', code),
                       'date': str(last.get('date', '')).replace('-', ''),
                       'change_pct': (close / previous - 1) * 100})
    return result


def observation_rows(report):
    scores = report.get('scores', [])
    observations = report.get('long_term', {}).get('observations', [])
    if not scores:
        scores = [dict(o, returns={'r5d': o.get('r5d'), 'r60d': o.get('r60d')}) for o in observations]
    has_watchlist = any('is_watchlist' in s for s in scores)
    allowed = {o['code'] for o in observations} if observations else None
    stale = set(report.get('stale_codes') or [])
    rows = []
    for s in scores:
        code = str(s.get('code', ''))
        if s.get('is_stock') or code.startswith(('0', '3', '6')):
            continue
        if has_watchlist and not s.get('is_watchlist'):
            continue
        if allowed is not None and code not in allowed:
            continue
        score = number(s.get('blended_technical', s.get('technical_score')))
        ret = s.get('returns') or {}
        r5, r60 = number(ret.get('r5d')), number(ret.get('r60d'))
        ma20 = number((s.get('vs_ma') or {}).get('pct_ma20'))
        trend_ok = ma20 >= 0 if ma20 is not None else r60 is not None and r60 > 0
        if code in stale or score is None:
            label = '待更新' if code in stale else '数据不足'
        elif score >= 65 and trend_ok and r5 is not None and r5 >= 0:
            label = '偏乐观'
        elif score >= 65:
            label = '高分待确认'
        elif score >= 55:
            label = '中性观察'
        else:
            label = '偏弱'
        premium = number((s.get('premium_info') or {}).get('premium_pct', s.get('premium_pct')))
        if label == '偏乐观' and premium is not None and premium >= 3:
            label = '偏强 · 高溢价'
        rows.append({'code': code, 'name': s.get('name', code), 'score': score,
                     'r5': r5, 'r60': r60, 'label': label, 'stale': code in stale})
    return sorted(rows, key=lambda r: (r['stale'], -(r['score'] if r['score'] is not None else -1), r['code']))


def market_lines(report):
    timing = report.get('timing') or {}
    lines = []
    for index in report.get('market_snapshot', []):
        current = index.get('date') == str(report.get('date', '')).replace('-', '')
        lines.append(f"{index['name']} {fmt(index.get('change_pct'), '%', True) if current else '待更新'}")
    if not lines:
        lines.append('指数当日涨跌未记录')
    sentiment = timing.get('sentiment') or {}
    signals = sentiment.get('signals') or {}
    volume = number(signals.get('成交额(亿)'))
    if volume is not None:
        lines.append(f"成交额 {fmt(volume / 10000, '万亿')}" + ('（估算）' if sentiment.get('estimated') else ''))
    regime = None if timing.get('data_insufficient') else timing.get('regime_name')
    if regime:
        lines.append('趋势偏弱' if regime == '危机模式' else regime)
    return lines


def brief_notes(report):
    notes = []
    if report.get('preview_note'):
        notes.append(report['preview_note'])
    if report.get('stale_codes'):
        notes.append('行情待更新：' + '、'.join(report['stale_codes']) + '；相关估值与评分仅供参考。')
    if (report.get('timing') or {}).get('data_insufficient'):
        notes.append('大盘数据不足，市场状态暂不作结论。')
    warnings = (report.get('execution_warnings') or []) + (report.get('research', {}).get('warnings') or [])
    if warnings:
        warnings = list(dict.fromkeys(map(str, warnings)))
        notes.append(f'本期有{len(warnings)}项数据或执行提醒：' + warnings[0][:100] + ('…' if len(warnings[0]) > 100 else ''))
    return notes


def render_daily_text(report):
    title = '模拟盘日报' if report.get('type') == 'paper' else '长期ETF观察日报'
    rows = observation_rows(report)
    lines = [f"{title} | {report.get('date', '')}", ' · '.join(market_lines(report))]
    lines += brief_notes(report)
    lines += ['', 'ETF观察池 · 技术分从高到低']
    lines += [f"{r['name']}({r['code']}) {'—' if r['stale'] else fmt(r['score'])}分 · {r['label']} · 5日{'—' if r['stale'] else fmt(r['r5'], '%', True)}" for r in rows]
    if not rows:
        lines.append('暂无可用的观察池评分')
    p = report.get('portfolio') or {}
    lines += ['', f"账户 {fmt(p.get('total_assets'), '元')} · 本期盈亏 {fmt(p.get('total_daily_pnl'), '元', True)}",
              ' '.join(account_total(report)),
              '技术分反映走势强弱；偏乐观需趋势配合，不代表收益预测。']
    return '\n'.join(lines)


def render_daily_html(report):
    e = lambda value: html.escape(str(value))
    rows = observation_rows(report)
    paper = report.get('type') == 'paper'
    title = '模拟盘日报' if paper else '长期ETF观察日报'
    day = str(report.get('date') or report.get('long_term', {}).get('as_of', ''))
    display_day = f'{day[:4]}.{day[4:6]}.{day[6:8]}' if len(day) == 8 else day
    optimistic = [r['name'] for r in rows if r['label'] == '偏乐观']
    focus = '、'.join(optimistic[:3]) if optimistic else '暂没有趋势配合的偏乐观标的'
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '''<style>
body{margin:0;background:#eef2f6;color:#233247;font:14px/1.6 "Microsoft YaHei",Arial,sans-serif}
main{max-width:780px;margin:32px auto;padding:0 16px}header{background:#152e46;color:#fff;border-radius:16px;padding:26px 30px}
h1{font-size:26px;margin:5px 0 8px;letter-spacing:1px}h2{font-size:17px;margin:0 0 15px}.eyebrow{font-size:11px;letter-spacing:2px;color:#98b3cb}
.date{color:#c6d5e2;font-size:13px}.market{margin-top:20px;padding-top:16px;border-top:1px solid #3a5065;color:#e2edf6}
section{background:#fff;border:1px solid #e1e7ee;border-radius:14px;margin:16px 0;padding:22px 26px}
.focus{border-left:3px solid #209889;padding:2px 0 2px 14px;margin:0 0 18px}.muted,small{color:#8290a0;font-size:12px}
table{width:100%;border-collapse:collapse;table-layout:fixed}th{text-align:right;color:#8290a0;font-size:11px;font-weight:normal;padding:0 0 12px}
td{padding:13px 0;border-top:1px solid #edf0f4;text-align:right;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left;width:32%}
.score{font-size:22px;font-weight:bold;color:#244f73}.bar{width:52px;height:3px;background:#edf2f6;margin:4px 0 0 auto;border-radius:4px}.bar span{display:block;height:3px;background:#4d8ba6;border-radius:4px}
.tag{display:inline-block;font-size:11px;padding:3px 7px;border-radius:5px;color:#778493;background:#f1f4f7}.good{background:#e8f5ef;color:#27806b}.warm{background:#fff3df;color:#ac7c2a}
.up{color:#cf5961}.down{color:#248775}.account{font-size:13px;color:#65768a}.notice{background:#fff7e7;border:1px solid #efdfbb;padding:10px 14px;margin:12px 0;border-radius:8px;font-size:12px;color:#8b6d32}
footer{font-size:11px;color:#8e9aaa;padding:0 6px 24px}p{margin:6px 0}
@media(max-width:520px){main{margin:12px auto;padding:0 10px}header{padding:21px}section{padding:18px 14px}h1{font-size:23px}td{font-size:12px}.score{font-size:19px}.tag{padding:3px 4px;font-size:10px}.r60{display:none}td:first-child,th:first-child{width:34%}}
</style></head><body><main>''',
             f'<header><div class="eyebrow">DAILY MARKET / ETF WATCH</div><h1>{title}</h1><div class="date">{e(display_day)} · 收盘观察</div>',
             '<div class="market">' + e(' · '.join(market_lines(report))) + '</div></header>']
    parts += ['<div class="notice">'+e(n)+'</div>' for n in brief_notes(report)]
    parts += ['<section><h2>ETF 观察池</h2><div class="focus"><small>今日偏乐观</small><p><b>'+e(focus)+'</b></p></div>',
              '<table><thead><tr><th>标的</th><th>技术分</th><th>近5日</th><th class="r60">近60日</th><th>状态</th></tr></thead><tbody>']
    for r in rows:
        score = '—' if r['stale'] else fmt(r['score'])
        width = max(0, min(100, r['score'] or 0)) if not r['stale'] else 0
        tag = ' good' if r['label'] == '偏乐观' else ' warm' if '确认' in r['label'] or '溢价' in r['label'] else ''
        change = lambda value: '<span class="'+('up' if (value or 0) >= 0 else 'down')+'">'+fmt(value, '%', True)+'</span>'
        parts.append(f'<tr><td><b>{e(r["name"])}</b><br><small>{e(r["code"])}</small></td><td><span class="score">{score}</span><div class="bar"><span style="width:{width}%"></span></div></td><td>{change(r["r5"]) if not r["stale"] else "—"}</td><td class="r60">{change(r["r60"]) if not r["stale"] else "—"}</td><td><span class="tag{tag}">{e(r["label"])}</span></td></tr>')
    if not rows:
        parts.append('<tr><td colspan="5">暂无可用的观察池评分</td></tr>')
    p = report.get('portfolio') or {}
    total_label, total_value = account_total(report)
    parts += ['</tbody></table><p class="muted" style="margin-top:14px">按技术分排序 · 65分起结合均线与动量标注偏乐观</p></section>',
              '<section class="account"><h2>'+('模拟账户' if paper else '账户快照')+'</h2>',
              f'<p>总资产 <b>{fmt(p.get("total_assets"), "元")}</b> &nbsp; · &nbsp; {e(p.get("pnl_period_label") or "本期盈亏")} <b>{fmt(p.get("total_daily_pnl"), "元", True)}</b></p><p>{e(total_label)} <b>{total_value}</b></p></section>',
              '<footer>技术分反映走势强弱；偏乐观需趋势配合，不代表收益预测。<br>观察池包含跨境等资产，大盘概况仅指 A 股。</footer></main></body></html>']
    return ''.join(parts)
