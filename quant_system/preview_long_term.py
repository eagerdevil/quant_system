"""Build a local historical preview, without fetching data or sending messages.

python preview_long_term.py --date 20261013
"""
import argparse
import copy
import json
from pathlib import Path

from daily_runner import assemble_long_term_output
from performance_tracker import load_history
from report_mailer import generate_html_report


def build_preview(root, day):
    root = Path(root)
    with (root / f'report_{day}.json').open(encoding='utf-8') as f:
        old = json.load(f)
    with (root / 'portfolio.json').open(encoding='utf-8') as f:
        current_metadata = json.load(f)
    summary = copy.deepcopy(old['portfolio'])
    # Reconstruct holdings from the report date; never project today's holdings backwards.
    portfolio = {k: v for k, v in current_metadata.items() if k.startswith('_')}
    portfolio.update({h['code']: {'name': h['name'], 'shares': h['shares'],
                                'cost': h['cost'], 'current_price': h['price']}
                      for h in summary['holdings']})
    data = {}
    for s in old['scores']:
        code = s['code']
        paths = [root / 'kline_cache' / f'{code}_qf1.json', root / 'kline_cache' / f'{code}.json']
        # Stock collector uses the unsuffixed cache name despite requesting qfq.
        if s.get('is_stock') or str(code).startswith(('6', '0', '3')):
            paths.reverse()
        for p in paths:
            if p.exists():
                cached = json.loads(p.read_text(encoding='utf-8'))
                bars = cached.get('data', cached.get('klines', cached.get('kline', []))) if isinstance(cached, dict) else cached
                bars = [b for b in bars if str(b.get('date', '')).replace('-', '') <= day]
                data[code] = {'name': s['name'], 'kline': bars}
                break
    report = assemble_long_term_output(copy.deepcopy(old['scores']), old.get('timing', {}),
                                       portfolio, summary, {'etfs': data}, day,
                                       load_history(str(root), days=10000))
    report['preview_note'] = f'历史样例：行情与持仓截至{day}；规则、配置草案与基金档案采用当前版本（可能晚于行情日），不是无前视回测，也不能作为今日行情或交易依据。'
    output_dir = root / 'review_outputs'
    output_dir.mkdir(exist_ok=True)
    (output_dir / f'long_term_preview_{day}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / f'long_term_preview_{day}.html').write_text(generate_html_report(report), encoding='utf-8')
    (output_dir / f'long_term_preview_{day}.txt').write_text(report['preview_note']+'\n'+report['report'], encoding='utf-8')
    return output_dir / f'long_term_preview_{day}.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.date is None:
        root = Path(__file__).resolve().parent
        candidates = [p.stem.removeprefix('report_') for p in root.glob('report_????????.json')
                      if p.stem.removeprefix('report_').isdigit()]
        if not candidates:
            parser.error('没有可用的本地实盘历史报告')
        args.date = max(candidates)
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error('date must be YYYYMMDD')
    print(build_preview(Path(__file__).resolve().parent, args.date))
