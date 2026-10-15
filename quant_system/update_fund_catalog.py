"""Refresh public fund facts. No account changes, orders, or messaging.

Source snapshots retain extracted facts and URLs, not credentials. Basic archive
dates and disclosure dates are distinct from retrieval dates. Quote spreads are
timestamped; they are never treated as timeless product attributes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import time

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://fundf10.eastmoney.com/'}
LOT_SOURCE = 'https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734754.shtml'
LOF_LOT_SOURCE = 'https://docs.static.szse.cn/www/fund/guide/W020220128519514648198.pdf'


def fetch(url):
    error = None
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=25)
            response.raise_for_status()
            return response.content.decode('utf-8-sig')
        except (requests.RequestException, UnicodeDecodeError) as exc:
            error = exc
            time.sleep(0.4 * (attempt+1))
    raise ValueError(str(error))


def number(value):
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value).replace(',', ''))
    return float(match.group()) if match else None


def parse_basic(document, code):
    soup = BeautifulSoup(document, 'html.parser')
    table = next((t for t in soup.find_all('table') if '管理费率' in t.get_text()), None)
    if table is None:
        raise ValueError('基本概况表缺失')
    facts = {}
    for row in table.find_all('tr'):
        cells = [c.get_text(' ', strip=True) for c in row.find_all(['th', 'td'], recursive=False)]
        for i in range(0, len(cells)-1, 2):
            facts[cells[i]] = cells[i+1]
    if code not in facts.get('基金代码', ''):
        raise ValueError('基金代码不匹配')
    aum = facts.get('净资产规模', '')
    # Only accept a dated, explicitly denominated net asset figure.
    aum_match = re.search(r'([\d.,]+)亿元.*?(\d{4})年(\d{2})月(\d{2})日', aum)
    management, custody = number(facts.get('管理费率')), number(facts.get('托管费率'))
    sales_text = facts.get('销售服务费率', '')
    sales = number(sales_text)
    result = {
        'name': facts.get('基金简称'), 'full_name': facts.get('基金全称'),
        'fund_type': facts.get('基金类型'), 'manager': facts.get('基金管理人'),
        'custodian': facts.get('基金托管人'), 'index_name': facts.get('跟踪标的'),
        'benchmark_description': facts.get('业绩比较基准'),
        'management_fee_pct': management, 'custody_fee_pct': custody,
        'sales_service_fee_pct': sales,
        'annual_fee_pct': management+custody+(sales or 0) if management is not None and custody is not None else None,
        'fee_basis': '列示管理费+托管费+销售服务费（如有），不含交易费用及其他运营支出；非完整总费用率',
        'sales_service_fee_note': '平台列示为---，不另加计，保留原始标记' if sales is None else '平台列示年度费率',
        'aum_cny': float(aum_match[1].replace(',', '')) * 1e8 if aum_match else None,
        'aum_as_of': '-'.join(aum_match.groups()[1:]) if aum_match else None,
        'is_cross_border': '海外' in facts.get('基金类型', '') or 'QDII' in facts.get('基金全称', ''),
        'instrument_type': 'LOF' if 'LOF' in facts.get('基金全称', '') else 'ETF',
    }
    inception = re.search(r'(\d{4})年(\d{2})月(\d{2})日', facts.get('成立日期/规模', ''))
    result['inception_date'] = '-'.join(inception.groups()) if inception else None
    return {k: v for k, v in result.items() if v is not None}, facts


def parse_tracking(document):
    soup = BeautifulSoup(document, 'html.parser')
    for table in soup.find_all('table'):
        if '年化跟踪误差' not in table.get_text():
            continue
        rows = table.find_all('tr')
        for row in rows:
            cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
            if len(cells) >= 3:
                value = number(cells[1])
                text = table.parent.get_text(' ', strip=True)
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', text) or re.findall(r'\d{4}-\d{2}-\d{2}', soup.get_text(' ', strip=True))
                return {'tracking_error_annual_pct': value,
                        'tracking_error_as_of': dates[-1] if dates else None,
                        'tracking_error_note': '平台实际统计年化跟踪误差；不是收益差异，也不是合同目标' if value is not None else '平台年化跟踪误差显示--，不填0'}
    return {'tracking_error_note': '来源未列出实际年化跟踪误差'}


def parse_holdings(document):
    match = re.search(r'content:\s*"((?:\\.|[^"\\])*)"', document, flags=re.S)
    if not match:
        raise ValueError('持仓接口内容缺失')
    content = json.loads('"' + match[1] + '"')
    soup = BeautifulSoup(content, 'html.parser')
    boxes = soup.select('.boxitem') or [soup]
    if not boxes:
        return {}
    box = boxes[0]
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', box.get_text(' ', strip=True))
    rows = []
    table = box.find('table')
    if table:
        headings = [c.get_text(' ', strip=True) for c in table.find_all('th')]
        weight_index = next((i for i, x in enumerate(headings) if '占净值' in x), None)
        if weight_index is not None:
            for row in table.find_all('tr'):
                cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
                if len(cells) > weight_index:
                    weight = number(cells[weight_index])
                    if weight is not None:
                        rows.append({'code': cells[1], 'name': cells[2], 'weight_pct': weight})
    return {'top10_weight_pct': round(sum(r['weight_pct'] for r in rows[:10]), 4) if rows else None,
            'holdings_as_of': date_match.group() if date_match else None,
            'disclosed_stock_holdings': rows[:10], 'holdings_count_disclosed': len(rows[:10]),
            'holdings_note': '公开披露前十只直接股票占基金净值比例；不等于穿透后的指数成分权重' if rows else '平台未返回直接股票持仓；不将其解释为0%或已穿透分散'}


def refresh_one(item, as_of):
    code, old = item
    result = dict(old)
    sources, errors = {}, []
    pages = {}
    basic_url = f'https://fundf10.eastmoney.com/jbgk_{code}.html'
    try:
        pages['basic'] = fetch(basic_url)
        facts, raw = parse_basic(pages['basic'], code)
        result.update(facts)
        result.update(source_url=basic_url, verified_on=as_of, basic_verified_on=as_of)
        sources['basic'] = {'url': basic_url, 'retrieved_on': as_of, 'fields': raw}
    except ValueError as exc:
        errors.append('基本资料：' + str(exc))
    tracking_url = f'https://fundf10.eastmoney.com/tsdata_{code}.html'
    try:
        result.update(parse_tracking(fetch(tracking_url)))
        sources['tracking'] = {'url': tracking_url, 'retrieved_on': as_of}
    except ValueError as exc:
        errors.append('跟踪指标：' + str(exc))
    holdings_url = f'https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={code}&topline=10&year={as_of[:4]}&month='
    try:
        result.update(parse_holdings(fetch(holdings_url)))
        sources['holdings'] = {'url': holdings_url, 'page_url': f'https://fundf10.eastmoney.com/ccmx_{code}.html', 'retrieved_on': as_of}
    except ValueError as exc:
        errors.append('持仓：' + str(exc))
    result['tracking_difference_pct'] = None
    result['tracking_difference_note'] = '未取得同币种、同分红与区间口径的实际收益差异；不使用跟踪误差替代。非本次日常决策硬门槛。'
    # This is the exchange-traded secondary-market minimum, not ETF creation unit.
    result.update(lot_size=100, lot_size_source=LOF_LOT_SOURCE if result.get('instrument_type') == 'LOF' else LOT_SOURCE)
    if result.get('bucket') in ('gold', 'duration_bond') or code == '161226':
        result['holdings_note'] = '非股票类基金，前十大股票指标不适用；应按黄金、债券久期或商品期货风险评估'
        result['top10_weight_pct'] = None
        result['stock_concentration_applicable'] = False
    else:
        result['stock_concentration_applicable'] = True
    if code in ('513520', '513800'):
        result['stock_concentration_applicable'] = False
        result['holdings_note'] = '主要通过境外目标ETF投资；直接股票前十不适用，尚未取得底层指数穿透集中度，不表示没有股票风险'
        result['fee_basis'] += '；未将境外目标ETF费用穿透加总'
    result['sources'] = sources
    result['refresh_errors'] = errors
    return code, result


def render_catalog_summary(catalog):
    """Human-readable source ledger for every watched fund."""
    lines = ['# ETF观察池基础资料', '', f"获取日期：{catalog['refreshed_on']}。规模和持仓使用各自披露日期，不代表实时规模。", '',
             '年费率为列示管理费、托管费及销售服务费之和，不是完整总费用率；盘口价差会随行情变化。', '',
             '| 代码/名称 | 跟踪标的 | 列示年费率 | 规模（亿元）/日期 | 直接股票前十 | 实际年化跟踪误差 | 资料来源 |',
             '|---|---|---:|---|---|---|---|']
    for code, m in catalog['funds'].items():
        if m.get('instrument_type') == 'stock':
            continue
        fee = '未取得' if m.get('annual_fee_pct') is None else f"{m['annual_fee_pct']:.2f}%"
        aum = '未取得' if m.get('aum_cny') is None else f"{m['aum_cny']/1e8:.2f} / {m.get('aum_as_of')}"
        stocks = '不适用/未披露' if m.get('top10_weight_pct') is None else f"{m['top10_weight_pct']:.2f}% / {m.get('holdings_as_of')}"
        error = '平台未提供' if m.get('tracking_error_annual_pct') is None else f"{m['tracking_error_annual_pct']:.2f}% / {m.get('tracking_error_as_of') or '日期未提供'}"
        lines.append(f"| {code} {m['name']} | {m.get('index_name','未取得')} | {fee} | {aum} | {stocks} | {error} | [东方财富档案]({m.get('source_url','')}) |")
    lines += ['', '## 口径说明', '',
              '- 513520和513800主要通过境外目标ETF投资，直接股票表不适用；底层股票集中度仍需穿透。',
              '- 黄金、国债、白银期货基金不适用前十大股票指标，需分别观察黄金、久期与期货风险。',
              '- 实际跟踪收益差异与跟踪误差不是同一指标；未取得的收益差异保持空值，不作为本次日常决策门槛。',
              '- 当日价差、报价时间与出处保存在fund_catalog.json；次日需要重新核对，日报运行时会刷新盘口。',
              '- 3只个股继续独立观察，不伪装为ETF资料。',
              f'- ETF最低交易单位核验：[上交所说明]({LOT_SOURCE})。',
              '- 资料完整不等于已认定适合长期持有，配置草案仍需审阅。']
    return '\n'.join(lines) + '\n'


def fetch_spreads(codes):
    symbols = ','.join(('sh' if c.startswith('5') else 'sz') + c for c in codes)
    url = 'https://qt.gtimg.cn/q=' + symbols
    response = requests.get(url, headers=HEADERS, timeout=25)
    response.raise_for_status()
    response.encoding = 'gbk'
    results = {}
    for line in response.text.split(';'):
        if '="' not in line:
            continue
        fields = line.split('"')[1].split('~')
        if len(fields) < 31 or fields[2] not in codes:
            continue
        try:
            bid, ask = float(fields[9]), float(fields[19])
            stamp = fields[30]
            datetime.strptime(stamp, '%Y%m%d%H%M%S')
            if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
                continue
            results[fields[2]] = {'bid': bid, 'ask': ask, 'spread_pct': (ask-bid)/((ask+bid)/2)*100,
                                  'spread_as_of': stamp, 'spread_source_url': 'https://qt.gtimg.cn/q=' + ('sh' if fields[2].startswith('5') else 'sz') + fields[2],
                                  'spread_note': '盘口买一/卖一中间价相对价差；仅对该报价时刻有效，委托前需重新核对'}
        except (ValueError, IndexError):
            continue
    return results


def refresh_catalog(root=ROOT):
    root = Path(root)
    now = datetime.now(timezone(timedelta(hours=8)))
    as_of = now.strftime('%Y-%m-%d')
    catalog = json.loads((root/'fund_catalog.json').read_text(encoding='utf-8'))
    items = [(c, m) for c,m in catalog['funds'].items() if m.get('instrument_type') != 'stock']
    with ThreadPoolExecutor(max_workers=3) as pool:
        updates = list(pool.map(lambda item: refresh_one(item, as_of), items))
    for code, meta in updates:
        catalog['funds'][code] = meta
        print(code, '基本资料已核验' if meta.get('basic_verified_on') == as_of else '基本资料待核验',
              '持仓' + str(meta.get('holdings_as_of')), '异常' + str(len(meta.get('refresh_errors', []))))
    try:
        quotes = fetch_spreads([c for c,_ in items])
    except requests.RequestException as exc:
        quotes = {}
        print('盘口采集失败：', exc)
    for c, quote in quotes.items():
        catalog['funds'][c].update(quote)
    catalog.update(schema_version=2, refreshed_on=as_of,
                   note='基础资料来源东方财富公开档案；费用为列示经常费率之和，非完整总费用率。披露日期与获取日期分别保存。实际跟踪差异不是本次日常决策硬门槛；实时价差和溢价仍需当日核对。')
    output = root/'review_outputs'
    output.mkdir(exist_ok=True)
    payload = json.dumps(catalog, ensure_ascii=False, indent=2)
    (output/f'fund_catalog_sources_{now:%Y%m%d}.json').write_text(payload, encoding='utf-8')
    (root/'FUND_CATALOG_REVIEW.md').write_text(render_catalog_summary(catalog), encoding='utf-8')
    # One replacement only after all source attempts; failed fields retain explicit status.
    temporary = root/'fund_catalog.json.tmp'
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(root/'fund_catalog.json')
    return catalog


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--if-due', action='store_true')
    parser.add_argument('--max-age-days', type=int, default=7)
    args = parser.parse_args()
    if args.if_due:
        existing = json.loads((ROOT/'fund_catalog.json').read_text(encoding='utf-8'))
        now = datetime.now(timezone(timedelta(hours=8))).date()
        stamps = [m.get('basic_verified_on') for m in existing['funds'].values()
                  if m.get('instrument_type') != 'stock']
        try:
            fresh = bool(stamps) and all(0 <= (now-datetime.strptime(s,'%Y-%m-%d').date()).days < args.max_age_days for s in stamps)
        except (TypeError,ValueError):
            fresh = False
        if fresh:
            print('基金基础档案仍在更新周期内，跳过重复抓取；盘口由日报当日刷新')
            sys.exit(0)
    refresh_catalog()
