import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from daily_brief import observation_rows, market_snapshot, render_daily_html, render_daily_text


def etf(code, technical, penalized=10, r5=2, ma20=1, **extra):
    return dict(code=code, name=code, technical_score=technical, score=penalized,
                returns={'r5d': r5, 'r60d': 4}, vs_ma={'pct_ma20': ma20},
                is_watchlist=True, **extra)


def test_rank_uses_technical_score_and_only_watchlist_etfs():
    scores = [etf('510300', 80), etf('510500', 60, penalized=99),
              etf('600000', 99, is_stock=True), dict(etf('588000', 90), is_watchlist=False)]
    rows = observation_rows({'scores': scores})
    assert [r['code'] for r in rows] == ['510300', '510500']
    assert rows[0]['label'] == '偏乐观'


def test_high_score_does_not_imply_optimism_with_weak_trend_or_stale_data():
    report = {'scores': [etf('510300', 85, ma20=-5), etf('510500', 95)],
              'stale_codes': ['510500']}
    rows = observation_rows(report)
    assert [r['label'] for r in rows] == ['高分待确认', '待更新']
    assert '95.0' not in render_daily_html(report)
    assert '95.0' not in render_daily_text(report)


def test_missing_score_not_coerced_to_zero_and_dynamic_content_escaped():
    report = {'scores': [dict(etf('510300', None), name='<script>bad</script>')]}
    assert observation_rows(report)[0]['label'] == '数据不足'
    rendered = render_daily_html(report)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered


def test_snapshot_uses_consecutive_closes_and_preserves_stale_date():
    snapshot = market_snapshot({'000300': {'name': '沪深300', 'data': [
        {'date': '2026-10-13', 'close': 100}, {'date': '2026-10-14', 'close': 101},
        {'date': '2026-10-16', 'close': 150}]}}, '20261015')
    assert snapshot[0]['date'] == '20261014'
    report = {'date': '20261015', 'market_snapshot': snapshot}
    assert '沪深300 待更新' in render_daily_text(report)
    assert '+1.0%' not in render_daily_text(report)
