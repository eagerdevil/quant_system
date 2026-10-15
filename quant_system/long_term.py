"""Long-term ETF observation and contribution review. Pure advisory, no orders."""
import calendar
import html
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from cashflow_performance import date_key, account_performance
from dca_advisor import evaluate_dca_guidance
from contribution_review import purchase_example, disclosed_overlap, contribution_priority, report_changes

ROOT = Path(__file__).resolve().parent
BUCKET_NAMES = {
    'a_broad': 'A股宽基', 'us_broad': '美国宽基', 'japan_broad': '日本宽基',
    'europe_broad': '欧洲宽基', 'hongkong_broad': '港股宽基', 'dividend': '红利策略',
    'gold': '黄金', 'defensive': '现金/低波动防御', 'duration_bond': '中长期利率债',
    'satellite': '个股/行业及风格集中资产', 'unclassified': '待分类',
}
REQUIRED_FUND_FIELDS = {
    'index_name': '跟踪指数', 'annual_fee_pct': '年度费率',
    'aum_cny': '基金规模',
    'lot_size': '交易单位', 'source_url': '核验来源', 'verified_on': '核验日期',
}


def load_inputs(root=ROOT):
    def read(name):
        with (Path(root) / name).open(encoding='utf-8') as f:
            return json.load(f)
    return read('long_term_config.json'), read('fund_catalog.json')['funds'], read('contribution_ledger.json')


def validate_config(config):
    if config.get('mode') != 'long_term':
        raise ValueError('实盘配置必须为long_term；交易策略请使用独立模拟盘')
    weights = config.get('target_weights', {})
    if not weights or any(k not in BUCKET_NAMES or not isinstance(v, (int, float))
                          or not math.isfinite(v) or v < 0 for k, v in weights.items()):
        raise ValueError('目标配置无效')
    if abs(sum(weights.values()) - 1) > 1e-6:
        raise ValueError('目标配置比例合计必须为100%')
    if config.get('allocation_status') not in ('draft', 'approved'):
        raise ValueError('目标配置状态无效')
    for key in ('monthly_budget', 'max_premium_pct', 'max_fee_fraction',
                'single_stock_review_weight', 'single_fund_review_weight'):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key}必须是正数')
    if not 1 <= config['review_day'] <= 28 or not 1 <= config['max_wait_days'] <= 31:
        raise ValueError('复核日期或等待期限无效')
    if config['max_orders_per_month'] < 1:
        raise ValueError('月交易次数必须至少为1')


def budget_status(config, ledger, today, cash):
    month = today[:6]
    status = {'monthly_budget': config['monthly_budget'], 'spent': None, 'remaining': None,
              'cash': max(0, cash), 'available_for_review': None, 'orders': None,
              'known': False, 'needs_confirmation': True, 'confirmed_through': ledger.get('reconciled_through'),
              'note': '本月交易尚未核对，预算余额未知；账户现金不等于可投入预算'}
    try:
        if not ledger.get('reconciled_through') or date_key(ledger['reconciled_through'])[:6] != month:
            return status
        confirmed = date_key(ledger['reconciled_through'])
        buys = []
        for tx in ledger.get('transactions', []):
            day = date_key(tx['date'])
            if tx['side'] not in ('buy', 'sell'):
                raise ValueError('side')
            amount, fee = float(tx['amount']), float(tx['fee'])
            if not all(math.isfinite(v) and v >= 0 for v in (amount, fee)):
                raise ValueError('amount')
            if day[:6] == month and day <= today and tx['side'] == 'buy':
                buys.append(amount + fee)
        spent = sum(buys)
        remaining = max(0, config['monthly_budget'] - spent)
        available = min(remaining, max(0, cash))
        if len(buys) >= config['max_orders_per_month']:
            available = 0
        status.update(known=True, spent=spent, remaining=remaining, orders=len(buys),
                      available_for_review=available,
                      needs_confirmation=confirmed < today,
                      note='按已核对实际买入及费用扣减；卖出收入不增加月预算，未花预算不自动滚入下月')
        if confirmed < today:
            status['note'] = f'截至{confirmed}已核对；其后按已记录交易估算余额，交易前确认期间有无遗漏，不必为看日报每天确认'
    except (KeyError, TypeError, ValueError):
        status['note'] = '交易台账字段无效，预算余额暂停计算'
    return status


def fund_readiness(meta, today):
    if meta.get('instrument_type') == 'stock':
        return {'ready': False, 'missing': ['个股需独立基本面研究'], 'label': '个股观察，不纳入ETF加仓候选'}
    missing = [name for key, name in REQUIRED_FUND_FIELDS.items() if meta.get(key) is None or meta.get(key) == '']
    for key in ('annual_fee_pct', 'tracking_difference_pct', 'aum_cny', 'top10_weight_pct', 'spread_pct', 'lot_size'):
        if key in meta and meta[key] is not None:
            try:
                value = float(meta[key])
                if not math.isfinite(value) or (key != 'tracking_difference_pct' and value < 0):
                    raise ValueError()
                if key in ('aum_cny', 'lot_size') and value <= 0:
                    raise ValueError()
                if key == 'lot_size' and value != int(value):
                    raise ValueError()
                if key in ('top10_weight_pct', 'spread_pct') and value > 100:
                    raise ValueError()
            except (TypeError, ValueError):
                missing.append(REQUIRED_FUND_FIELDS.get(key, {
                    'tracking_difference_pct': '实际跟踪差异', 'top10_weight_pct': '股票集中度',
                    'spread_pct': '买卖价差'}.get(key, key)) + '无效')
    if meta.get('verified_on'):
        try:
            age = (datetime.strptime(today, '%Y%m%d') - datetime.strptime(date_key(meta['verified_on']), '%Y%m%d')).days
            if not 0 <= age <= 180:
                missing.append('资料需重新核验')
        except ValueError:
            missing.append('核验日期无效')
    if meta.get('source_url') and not str(meta['source_url']).startswith('https://'):
        missing.append('来源链接无效')
    supplemental = []
    if meta.get('tracking_difference_pct') is None:
        supplemental.append('实际跟踪收益差异未取得；不作为本次日常决策硬门槛')
    if meta.get('top10_weight_pct') is None and meta.get('stock_concentration_applicable', True):
        supplemental.append(meta.get('holdings_note', '股票集中度待补'))
    return {'ready': not missing, 'missing': missing, 'supplemental_notes': supplemental,
            'label': '基础资料已核验，结合估值与风险复核' if not missing else '长期适配度未评级'}


def build_long_term_report(scores, portfolio, summary, today, config, catalog, ledger,
                           stale_codes=(), history=(), risk=None):
    validate_config(config)
    today = date_key(today)
    stale = set(stale_codes) | set(summary.get('stale_codes', []))
    total = summary.get('total_assets', 0)
    budget = budget_status(config, ledger, today, summary.get('available_cash', 0))
    bucket_values = {'defensive': summary.get('available_cash', 0)}
    alerts = []
    for h in summary.get('holdings', []):
        code = h['code']
        meta = catalog.get(code, {})
        bucket = meta.get('bucket', 'unclassified')
        bucket_values[bucket] = bucket_values.get(bucket, 0) + h.get('value', 0)
        weight = h.get('value', 0) / total if total > 0 else 0
        stock = meta.get('instrument_type') == 'stock' or str(code).startswith(('6', '0', '3'))
        cap = config['single_stock_review_weight'] if stock else config['single_fund_review_weight']
        if weight > cap:
            alerts.append(f"{h.get('name', code)}占账户{weight:.1%}，超过复核参考线{cap:.0%}；优先评估用新增资金逐步分散")
    allocations = []
    for bucket in dict.fromkeys(list(config['target_weights']) + list(bucket_values)):
        actual = bucket_values.get(bucket, 0) / total if total > 0 else 0
        target = config['target_weights'].get(bucket, 0)
        allocations.append({'bucket': bucket, 'name': BUCKET_NAMES.get(bucket, bucket),
                            'actual_pct': actual * 100, 'target_pct': target * 100,
                            'gap_pct': (target - actual) * 100})
    gaps = {a['bucket']: a['gap_pct'] for a in allocations}
    if stale:
        alerts.append('行情未确认更新：' + '、'.join(sorted(stale)) + '；涉及的估值与配置偏离仅供参考')
    if config['allocation_status'] == 'draft':
        alerts.append('配置比例为待审阅草案，偏离值仅供讨论，不生成具体买入指令')
    observations = []
    for s in scores:
        if any('is_watchlist' in x for x in scores) and not (
            s.get('is_watchlist') or s.get('is_holding') or s.get('is_stock') or s['code'] in portfolio
        ):
            continue
        code = s['code']
        meta = catalog.get(code, {})
        bucket = meta.get('bucket', 'unclassified')
        fit = fund_readiness(meta, today)
        # Each instrument's own trend is descriptive; A-share regime never gates overseas assets.
        own_trend = '数据不足'
        r60 = (s.get('returns') or {}).get('r60d')
        if r60 is not None:
            own_trend = '近60日上涨' if r60 > 0 else '近60日下跌' if r60 < 0 else '近60日持平'
        guidance = s.get('dca_guidance') or {}
        timing_note = guidance.get('action', '仅观察技术状态')
        if timing_note.startswith('按你原定计划'):
            timing_note = '自身技术指标未触发明显过热条件；是否投入仍取决于配置、资料和预算'
        elif timing_note.startswith('基础定投可照常'):
            timing_note = '自身短期指标偏热；避免把强势评分理解为额外加码理由'
        elif timing_note.startswith('回撤分批候选'):
            timing_note = '短期回撤特征成立；仅作为时机观察，需通过其余复核条件'
        premium = s.get('premium_info') or {}
        premium_value = premium.get('premium_pct')
        blockers = []
        if code in stale:
            blockers.append('行情过期/缺失')
        if not fit['ready']:
            blockers.append(fit['label'])
        if premium_value is None:
            blockers.append('溢价未核实')
        elif not isinstance(premium_value, (int, float)) or not math.isfinite(premium_value):
            blockers.append('溢价无效')
        else:
            if premium.get('is_qdii') and premium.get('data_source') != 'iopv':
                blockers.append('跨境净值口径可能滞后，需复核实时估值')
            if premium_value >= config['max_premium_pct']:
                blockers.append(f"溢价{premium_value:.2f}%达到复核上限{config['max_premium_pct']:.1f}%")
        spread = meta.get('spread_pct')
        spread_stamp = str(meta.get('spread_as_of', '')).replace('-', '')
        if not isinstance(spread, (int, float)) or not math.isfinite(spread) or spread < 0 or spread_stamp[:8] != today:
            blockers.append('当日盘口价差待核对')
        elif spread > config.get('max_spread_pct', 0.3):
            blockers.append(f'盘口价差{spread:.2f}%偏大，需评估流动性成本')
        if guidance.get('action', '').startswith(('数据不足', '本期先跳过')):
            blockers.append(guidance['action'])
        if gaps.get(bucket, 0) <= 0:
            blockers.append('该类别未低于配置目标')
        if config['allocation_status'] != 'approved':
            blockers.append('配置草案待审阅')
        if not budget['known']:
            blockers.append('月预算尚未核对')
        elif budget['available_for_review'] <= 0:
            blockers.append('本月预算/交易次数或账户现金不足')
        reference_price = meta.get('ask') if spread_stamp[:8] == today and meta.get('ask',0)>0 else s.get('price',0)
        example = purchase_example(reference_price, meta.get('lot_size'), budget['available_for_review'] or 0,
                                   spread if isinstance(spread,(int,float)) and math.isfinite(spread) else 0)
        if budget['known'] and budget['available_for_review'] > 0:
            if not example['shares']:
                blockers.append('可用金额不足一手或交易单位待核实')
            elif example['cost_fraction'] > config['max_fee_fraction']:
                blockers.append('按实际整数份额估算的费用占比偏高，考虑累计资金或核实低成本渠道')
        overlap = disclosed_overlap(code, catalog, summary.get('holdings', []), total)
        observations.append({
            'code': code, 'name': s.get('name', code), 'bucket': BUCKET_NAMES.get(bucket, bucket),
            'price': s.get('price'), 'technical_score': s.get('score'),
            'r5d': (s.get('returns') or {}).get('r5d'), 'r60d': r60, 'trend': own_trend,
            'premium_pct': premium_value, 'premium_source': premium.get('data_source', '未知'),
            'nav_date': premium.get('nav_date'), 'fit': fit,
            'fund_profile': {k: meta.get(k) for k in (
                'index_name', 'annual_fee_pct', 'management_fee_pct', 'custody_fee_pct',
                'aum_cny', 'aum_as_of', 'top10_weight_pct', 'holdings_as_of',
                'tracking_error_annual_pct', 'tracking_error_as_of', 'tracking_error_note',
                'spread_pct', 'spread_as_of', 'spread_source_url', 'lot_size',
                'source_url', 'verified_on', 'fee_basis', 'holdings_note')},
            'timing_note': timing_note,
            'purchase_example': example, 'overlap': overlap,
            'needs_budget_confirmation': budget['needs_confirmation'],
            'review_candidate': not blockers, 'blockers': blockers,
            'allocation_gap_pct': gaps.get(bucket, 0),
        })
        observations[-1]['priority'] = contribution_priority(observations[-1], s, config, overlap)
    perf_reports = [r for r in history if date_key(r.get('_date') or r.get('date')) < today]
    perf_reports.append({'date': today, 'portfolio': summary})
    performance = account_performance(perf_reports, portfolio, config.get('legacy_placeholder_policy','block'))
    reviews = []
    current_date = datetime.strptime(today, '%Y%m%d')
    for label, start in [('本周', (current_date - timedelta(days=current_date.weekday())).strftime('%Y%m%d')),
                         ('本月', today[:6] + '01')]:
        before = [r for r in perf_reports if date_key(r.get('_date') or r.get('date')) < start]
        within = [r for r in perf_reports if date_key(r.get('_date') or r.get('date')) >= start]
        period = sorted(before, key=lambda r: date_key(r.get('_date') or r.get('date')))[-1:] + within
        reviews.append({'label': label, 'start': start, 'end': today,
                        'performance': account_performance(period, portfolio, config.get('legacy_placeholder_policy','block'))})
    if not performance['available']:
        alerts.append(performance['reason'])
    day = int(today[-2:])
    month_days = calendar.monthrange(int(today[:4]), int(today[4:6]))[1]
    deadline = today[:6] + f"{min(month_days, config['max_wait_days']):02d}"
    candidates = [o for o in observations if o['review_candidate']]
    candidates.sort(key=lambda o: (o['priority']['overheated'], -o['priority']['score'], o['code']))
    # Draft configuration may be discussed, but never becomes an actionable candidate.
    draft_reviews = [o for o in observations if o['blockers'] == ['配置草案待审阅']]
    draft_reviews.sort(key=lambda o: (o['priority']['overheated'], -o['priority']['score'], o['code']))
    priorities = candidates[:2] if config['allocation_status'] == 'approved' else draft_reviews[:2]
    action = '今天以观察为主；没有已满足条件的加仓候选'
    if candidates and day >= config['review_day']:
        action = '到达月度复核窗口，可审阅配置不足类别的候选；执行前核对实时价格与费用'
    elif candidates:
        action = '有待复核候选，留待月度复核日统一评估，减少小额交易'
    if candidates and budget['needs_confirmation']:
        action += '；预算沿用最近记录，交易前确认未核对期间有无新增交易'
    if today >= deadline:
        action += '；已到本月等待期限，完成一次复核；资料/溢价/预算门槛不因期限到达而放松'
    changes = report_changes(observations, allocations, summary.get('holdings', []), history, today, stale)
    return {'mode': 'long_term', 'version': 2, 'as_of': today, 'action': action,
            'budget': budget, 'allocation_status': config['allocation_status'],
            'allocation_note': config['allocation_note'], 'allocations': allocations,
            'review_day': config['review_day'], 'review_deadline': deadline,
            'holdings': summary.get('holdings', []),
            'alerts': alerts, 'observations': observations, 'performance': performance,
            'priorities': priorities, 'changes': changes,
            'risk': risk or {}, 'period_reviews': reviews,
            'score_note': '技术分仅描述相对走势，不是买入评级、胜率或长期质量',
            'plan': {'mode': 'long_term', 'buy_list': [], 'sell_list': [], 'hold_list': [],
                     'review_candidates': [o['code'] for o in candidates], 'available': budget['available_for_review'],
                     'timing_advice': action}}


def format_long_term_text(data, summary):
    b = data['budget']
    money = lambda v: '待核对' if v is None else f'{v:,.2f}元'
    lines = [f"长期ETF观察日报 | {data['as_of']}", data['action'], '',
             f"账户总资产 {money(summary.get('total_assets'))} | 现金 {money(b['cash'])}",
             f"月预算 {money(b['monthly_budget'])} | 本月已用 {money(b['spent'])} | 剩余 {money(b['remaining'])}",
             b['note'], f"每月{data['review_day']}日复核，本月最迟复核日 {data['review_deadline']}", '',
             '需要关注：']
    changes = data.get('changes', {})
    lines[2:2] = [f"相比上一报告（{changes.get('previous_date') or '首次记录'}）"] + changes.get('items', []) + ['']
    lines.extend('- ' + a for a in data['alerts'])
    if data.get('priorities'):
        lines += ['', '本月优先复核（配置草案仅供讨论）' if data['allocation_status']=='draft' else '本月优先复核']
        for o in data['priorities']:
            x = o['purchase_example']
            lines.append(f"{o['name']}：复核优先级{o['priority']['score']:.1f}；" + '；'.join(o['priority']['reasons']))
            lines.append(f"  参考价{x['reference_price']:.3f}，示例{x['shares']}份，成交额{x['amount']:.2f}元，佣金{x['commission']:.2f}元，滑点{x['slippage']:.2f}元，合计需现金{x['cash_required']:.2f}元，剩余{x['cash_left']:.2f}元")
    lines += ['', '实际持仓']
    lines.extend(f"{h['name']}({h['code']})：市值{h['value']:.2f}元，仓位{h['weight']:.1f}%，持仓盈亏{h['pnl']:+.2f}元"
                 for h in data.get('holdings', []))
    lines += ['', '配置对照（待审阅草案）' if data['allocation_status'] == 'draft' else '配置对照']
    lines.extend(f"{a['name']}：当前{a['actual_pct']:.1f}% / 目标{a['target_pct']:.1f}% / 缺口{a['gap_pct']:+.1f}个百分点"
                 for a in data['allocations'])
    lines += [data['allocation_note'], '', '各观察池行情与加仓复核', data['score_note']]
    for o in data['observations']:
        lines.append(f"{o['name']}({o['code']}) | {o['bucket']} | 技术分{o['technical_score']} | {o['trend']} | 近5日{o['r5d']}%")
        lines.append('  ' + ('可进入人工复核' if o['review_candidate'] else '；'.join(o['blockers'])))
        lines.append('  ' + o['timing_note'])
        lines.append(f"  复核优先级{o['priority']['score']:.1f}："+'；'.join(o['priority']['reasons']))
        f = o['fund_profile']
        if f.get('source_url'):
            annual = '未取得' if f.get('annual_fee_pct') is None else f"{f['annual_fee_pct']:.2f}%/年（列示费用）"
            aum = '未取得' if f.get('aum_cny') is None else f"{f['aum_cny']/1e8:.2f}亿元（{f.get('aum_as_of')}）"
            lines.append(f"  指数：{f.get('index_name')} | 费率：{annual} | 规模：{aum}")
            lines.append('  来源：' + f['source_url'])
    p = data['performance']
    lines += ['', '账户绩效']
    if p['available']:
        lines += [f"区间盈亏{p['profit']:+.2f}元 | 近似时间加权收益{p['return_pct']:+.2f}% | 最大回撤{p['max_drawdown_pct']:.2f}%", p['method_note']]
        if p['xirr_pct'] is not None:
            lines.append(f"区间XIRR {p['xirr_pct']:+.2f}%（年化）")
    else:
        lines.append(p['reason'])
    for review in data.get('period_reviews', []):
        rp = review['performance']
        lines.append(review['label'] + '复盘：' +
                     (f"盈亏{rp['profit']:+.2f}元，资金流调整后收益{rp['return_pct']:+.2f}%" if rp['available'] else rp['reason']))
    if data.get('risk'):
        from risk_engine import format_risk_section
        lines += ['', format_risk_section(data['risk'])]
    return '\n'.join(lines)


def render_long_term_html(report):
    data = report['long_term']
    e = lambda value: html.escape(str(value))
    b = data['budget']
    money = lambda value: '待核对' if value is None else f'{value:,.2f}元'
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<style>body{font:15px/1.65 "Microsoft YaHei",sans-serif;color:#172b4d;background:#f3f6fa;margin:0;padding:24px}main{max-width:1100px;margin:auto}section{background:white;padding:22px;margin:18px 0;border-radius:12px}h1{font-size:26px}h2{font-size:19px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e4e9ef;vertical-align:top}.scroll{overflow-x:auto}small{color:#53667b}.tag{color:#175b7d}li{margin:7px 0}</style></head><body><main>',
             f"<h1>长期ETF观察日报</h1><p>{e(data['as_of'])} · 月度积累 / 每日观察</p>",
             f"<section><h2>今天是否需要行动</h2><p>{e(data['action'])}</p>",
             f"<p>月预算 <b>{money(b['monthly_budget'])}</b> · 已用 {money(b['spent'])} · 剩余 {money(b['remaining'])} · 账户现金 {money(b['cash'])}</p>",
             f"<small>{e(b['note'])}。每月{data['review_day']}日复核，最迟 {data['review_deadline']}。</small></section>",
             '<section><h2>持仓与资料提醒</h2><ul>' + ''.join('<li>'+e(a)+'</li>' for a in data['alerts']) + '</ul></section>',
             '<section><h2>配置草案 · 待审阅</h2>' if data['allocation_status'] == 'draft' else '<section><h2>配置对照</h2>',
             '<div class="scroll"><table><tr><th>类别</th><th>当前</th><th>目标</th><th>缺口</th></tr>']
    changes = data.get('changes', {})
    brief = '<section><h2>相比上一报告发生了什么</h2><small>比较日期：' + e(changes.get('previous_date') or '首次记录') + '</small><ul>'
    brief += ''.join('<li>'+e(item)+'</li>' for item in changes.get('items', [])) + '</ul></section>'
    parts.insert(6, brief)
    if data.get('priorities'):
        priority_html = '<section><h2>本月优先复核' + (' · 配置草案待审阅' if data['allocation_status']=='draft' else '') + '</h2>'
        priority_html += '<p><small>排序用于比较复核顺序，不是收益预测；整数份额仅作费用示例，不提交委托。</small></p>'
        for o in data['priorities']:
            x = o['purchase_example']
            priority_html += '<p><b>'+e(o['name'])+'</b> · 优先级 '+e(o['priority']['score'])+'<br>'+e('；'.join(o['priority']['reasons']))+'</p>'
            priority_html += f"<p>参考价{x['reference_price']:.3f} · 示例{x['shares']}份 · 成交额{x['amount']:.2f}元<br>佣金{x['commission']:.2f}元 + 假设滑点{x['slippage']:.2f}元 · 共需现金{x['cash_required']:.2f}元 · 剩余{x['cash_left']:.2f}元</p>"
            if o['needs_budget_confirmation']:
                priority_html += '<small>预算沿用最近记录，交易前需确认期间无遗漏。</small>'
            priority_html += '<small>'+e(o['overlap']['note'])+'</small>'
        parts.insert(7, priority_html+'</section>')
    for a in data['allocations']:
        parts.append(f"<tr><td>{e(a['name'])}</td><td>{a['actual_pct']:.1f}%</td><td>{a['target_pct']:.1f}%</td><td>{a['gap_pct']:+.1f}个百分点</td></tr>")
    parts += ['</table></div><p>'+e(data['allocation_note'])+'</p></section>',
              '<section><h2>实际持仓</h2><div class="scroll"><table><tr><th>标的</th><th>市值</th><th>仓位</th><th>持仓盈亏</th></tr>']
    for h in data.get('holdings', []):
        parts.append(f"<tr><td>{e(h['name'])} ({e(h['code'])})</td><td>{h['value']:.2f}元</td><td>{h['weight']:.1f}%</td><td>{h['pnl']:+.2f}元</td></tr>")
    parts += ['</table></div><small>持仓盈亏不是账户总收益；总收益需计入已实现损益、费用和资金流。</small></section>',
              '<section><h2>观察池行情</h2><p>'+e(data['score_note'])+'</p><div class="scroll"><table><tr><th>标的 / 类别</th><th>走势</th><th>溢价 / 口径</th><th>长期资料 / 本期复核</th></tr>']
    for o in data['observations']:
        premium = '未知' if o['premium_pct'] is None else f"{o['premium_pct']:.2f}%"
        parts.append(f"<tr><td><b>{e(o['name'])}</b><br>{e(o['code'])} · {e(o['bucket'])}<br><small>复核优先级 {e(o['priority']['score'])}</small></td><td>5日 {e(o['r5d'])}% / 60日 {e(o['r60d'])}%<br>技术分 {e(o['technical_score'])}<br><small>{e(o['timing_note'])}</small></td><td>{e(premium)}<br><small>{e(o['premium_source'])} / 净值日{e(o['nav_date'] or '未知')}</small></td><td>{e(o['fit']['label'])}<br>{e('可进入人工复核' if o['review_candidate'] else '；'.join(o['blockers']))}<br><small>{e('；'.join(o['priority']['reasons']))}<br>待补：{e('、'.join(o['fit']['missing']) or '无')}</small></td></tr>")
    parts += ['</table></div></section><section><h2>基金基础资料</h2><p><small>获取日期和财报披露日期分别列示。费率为列示经常费率之和，不是完整总费用率；跟踪误差与实际收益差异分开记录。</small></p><div class="scroll"><table><tr><th>基金 / 指数</th><th>列示年费率 / 规模</th><th>持仓与跟踪</th><th>盘口 / 来源</th></tr>']
    for o in data['observations']:
        f = o.get('fund_profile') or {}
        if not f.get('source_url'):
            continue
        annual = '未取得' if f.get('annual_fee_pct') is None else f"{f['annual_fee_pct']:.2f}%"
        aum = '未取得' if f.get('aum_cny') is None else f"{f['aum_cny']/1e8:.2f}亿元"
        top10 = '不适用/未披露直接股票' if f.get('top10_weight_pct') is None else f"直接股票前十合计{f['top10_weight_pct']:.2f}%（{f.get('holdings_as_of')}）"
        error = '未披露实际年化跟踪误差' if f.get('tracking_error_annual_pct') is None else f"年化跟踪误差{f['tracking_error_annual_pct']:.2f}%（{f.get('tracking_error_as_of') or '统计日期未提供'}）"
        spread = '待核对' if f.get('spread_pct') is None else f"{f['spread_pct']:.3f}%"
        parts.append(f"<tr><td>{e(o['name'])} ({e(o['code'])})<br><small>{e(f.get('index_name'))}</small></td><td>{e(annual)}<br>{e(aum)}<br><small>规模日期 {e(f.get('aum_as_of'))}</small></td><td>{e(top10)}<br><small>{e(error)}<br>{e(f.get('holdings_note') or '')}</small></td><td>买卖价差 {e(spread)}<br><small>报价时间 {e(f.get('spread_as_of') or '未知')}<br>最低买入 {e(f.get('lot_size'))}份</small><br><a href=\"{e(f['source_url'])}\">查看基金档案</a><br><small>核验 {e(f.get('verified_on'))}</small></td></tr>")
    parts += ['</table></div></section><section><h2>账户绩效</h2>']
    p = data['performance']
    if p['available']:
        parts.append(f"<p>区间盈亏 {p['profit']:+.2f}元 · 近似时间加权收益 {p['return_pct']:+.2f}% · 最大回撤 {p['max_drawdown_pct']:.2f}%</p><small>{e(p['method_note'])}</small>")
    else:
        parts.append('<p>'+e(p['reason'])+'</p>')
    for review in data.get('period_reviews', []):
        rp = review['performance']
        detail = f"盈亏{rp['profit']:+.2f}元，资金流调整后收益{rp['return_pct']:+.2f}%" if rp['available'] else rp['reason']
        parts.append('<p>'+e(review['label']+'复盘：'+detail)+'</p>')
    parts += ['</section>']
    if data.get('risk'):
        from risk_engine import format_risk_section
        parts.append('<section><h2>实际持仓风险</h2><pre style="white-space:pre-wrap;overflow-wrap:anywhere;font:inherit">' + e(format_risk_section(data['risk'])) + '</pre></section>')
    if report.get('preview_note'):
        parts.insert(3, '<p style="padding:15px;background:#fff0ca">'+e(report['preview_note'])+'</p>')
    parts += ['<p><small>观察池不等于持仓；候选仅供复核。长期模式不根据短期评分生成买卖订单。历史数据不保证未来表现。</small></p></main></body></html>']
    return ''.join(parts)
