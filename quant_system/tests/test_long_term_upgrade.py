"""Behavioral regressions for the real-account long-term mode."""
import copy
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashflow_performance import cashflow_performance, xirr
from long_term import load_inputs, budget_status, build_long_term_report, fund_readiness, render_long_term_html
from daily_runner import assemble_long_term_output
from risk_engine import portfolio_risk_report, compute_var_cvar, compute_correlation_matrix
from dca_backtest import compare_dca
from report_mailer import generate_html_report, generate_wechat_markdown
from dca_advisor import evaluate_dca_guidance


def observations(values):
    return [{'date': f'202610{day:02d}', 'portfolio': {'total_assets': value}}
            for day, value in enumerate(values, 12)]


def test_flows_do_not_create_drawdown_or_profit():
    flows = [{'date': '20261013', 'type': 'withdraw', 'amount': 800},
             {'date': '20261014', 'type': 'deposit', 'amount': 2000}]
    p = cashflow_performance(observations([1000, 200, 2200]), flows)
    assert p['available'] and p['return_pct'] == 0 and p['profit'] == 0
    assert p['drawdown_pct'] == 0 and p['max_drawdown_pct'] == 0


def test_real_loss_and_recovery_survive_normalization():
    p = cashflow_performance(observations([1000, 800, 900]), [])
    assert p['return_pct'] == pytest.approx(-10)
    assert p['max_drawdown_pct'] == pytest.approx(-20)


def test_start_day_deposit_not_double_subtracted():
    p = cashflow_performance(observations([1000, 1100]), [{'date':'20261012','type':'deposit','amount':1000}])
    assert p['return_pct'] == pytest.approx(10)


@pytest.mark.parametrize('flows', [
    [{'date': '20261013', 'type': 'withdraw', 'amount': 12.34, 'note': '占位待核对'}],
    [{'date': '20261013', 'type': 'deposit', 'amount': float('nan')}],
])
def test_unreconciled_or_invalid_flow_suppresses_performance(flows):
    assert not cashflow_performance(observations([1000, 1000]), flows)['available']


def test_xirr_not_annualized_for_short_history():
    assert xirr([('20260101', -1000), ('20260201', 1100)]) is None
    assert xirr([('20250101', -1000), ('20260101', 1100)]) == pytest.approx(10)


def setup_account():
    config, _, _ = load_inputs()
    config['allocation_status'] = 'approved'
    score = {'code':'510300', 'name':'测试ETF', 'score':90, 'price':1.25,
             'is_watchlist':True, 'premium_info':{'premium_pct':0.2, 'data_source':'official_nav'},
             'indicators':{'rsi':50}, 'returns':{'r5d':1, 'r60d':5}, 'vs_ma':{},
             'dca_guidance':{'action':'按预算复核'}}
    meta = {'bucket':'a_broad', 'index_name':'测试指数', 'annual_fee_pct':0.2,
            'tracking_difference_pct':-0.3, 'aum_cny':1e9, 'top10_weight_pct':30,
            'spread_pct':0.01, 'spread_as_of':'20261015150000', 'lot_size':100, 'source_url':'https://example.org/fund',
            'verified_on':'20261015'}
    ledger = {'reconciled_through':'20261015', 'transactions':[]}
    summary = {'total_assets':1000, 'available_cash':1000, 'holdings':[], 'stale_codes':[]}
    portfolio = {'_available_cash':1000, '_cash_flows':[]}
    return config, score, meta, ledger, summary, portfolio


def build(config=None, score=None, meta=None, ledger=None, stale=()):
    c,s,m,l,pf,p = setup_account()
    return build_long_term_report([score or s], p, pf, '20261015', config or c,
                                 {'510300':meta or m}, ledger or l, stale)


def test_eligible_candidate_is_advisory_without_orders():
    r = build()
    assert r['observations'][0]['review_candidate']
    assert r['plan']['buy_list'] == r['plan']['sell_list'] == []


def test_draft_cannot_become_actionable():
    c,*_ = setup_account()
    c['allocation_status'] = 'draft'
    r = build(config=c)
    assert not r['observations'][0]['review_candidate']
    assert '配置草案待审阅' in r['observations'][0]['blockers']


@pytest.mark.parametrize('ledger', [
    {'reconciled_through':None, 'transactions':[]},
    {'reconciled_through':'20260930', 'transactions':[]},
])
def test_unknown_budget_is_not_zero_spent(ledger):
    r = build(ledger=ledger)
    assert r['budget']['spent'] is None and r['budget']['remaining'] is None
    assert not r['observations'][0]['review_candidate']


def test_sales_do_not_refill_month_budget():
    c,*_ = setup_account()
    ledger = {'reconciled_through':'20261015','transactions':[
        {'date':'20261010','side':'buy','amount':900,'fee':5},
        {'date':'20261012','side':'sell','amount':1000,'fee':5}]}
    b=budget_status(c,ledger,'20261015',2000)
    assert b['spent']==905 and b['remaining']==95 and b['available_for_review']==0


def test_stale_data_high_premium_and_unknown_metadata_block_candidates():
    assert not build(stale=['510300'])['observations'][0]['review_candidate']
    c,s,m,*_=setup_account()
    s['premium_info']['premium_pct']=5
    assert not build(score=s)['observations'][0]['review_candidate']
    del m['annual_fee_pct']
    assert not build(meta=m)['observations'][0]['review_candidate']


def test_qdii_stale_nav_never_becomes_live_premium():
    c,s,*_=setup_account()
    s['premium_info'].update(is_qdii=True, premium_pct=0.1)
    assert not build(score=s)['observations'][0]['review_candidate']
    s['premium_info']['data_source']='iopv'
    assert build(score=s)['observations'][0]['review_candidate']


def test_yesterdays_spread_blocks_new_candidate_even_when_basic_archive_is_current():
    c,s,m,*_=setup_account()
    m['spread_as_of']='20261014150000'
    result=build(meta=m)['observations'][0]
    assert not result['review_candidate'] and '当日盘口价差待核对' in result['blockers']


def test_unpublished_tracking_difference_does_not_block_daily_review():
    c,s,m,*_=setup_account()
    m['tracking_difference_pct']=None
    assert build(meta=m)['observations'][0]['review_candidate']


def test_tiny_absolute_premium_is_not_a_high_risk_percentile():
    c,s,*_=setup_account()
    s['premium_info']={'premium_pct':0.05, 'is_qdii':True, 'data_source':'iopv',
                       'history':{'has_history':True,'percentile':99,'current_source':'iopv'}}
    guidance=evaluate_dca_guidance(s,True,200)
    assert not guidance['action'].startswith('本期先跳过')


def bars(n=75, start='20260701'):
    dt=datetime.strptime(start,'%Y%m%d')
    return [{'date':(dt+timedelta(days=i)).strftime('%Y%m%d'),
             'close':2+0.01*math.sin(i), 'high':2.1, 'low':1.9, 'volume':1000} for i in range(n)]


def test_watchlist_cannot_change_account_risk():
    p={'A':{'shares':100,'cost':2,'current_price':2}}
    data={'A':{'kline':bars()}}
    base=portfolio_risk_report(p,data)
    data.update({'B':{'kline':bars()},'C':{'kline':bars()}})
    augmented=portfolio_risk_report(p,data)
    assert base['correlation']==augmented['correlation']
    assert base['concentration']==augmented['concentration']


def test_missing_held_risk_not_treated_as_cash():
    p={'A':{'shares':100,'current_price':2},'B':{'shares':100,'current_price':2}}
    r=compute_var_cvar(400,{'A':{'kline':bars()}},p)
    assert r['error']=='incomplete_holdings' and r['missing_codes']==['B']


def test_single_company_concentration_cannot_receive_perfect_score():
    from risk_engine import detect_concentration_risk
    holdings={'600900':{'shares':375,'current_price':1},
              '510300':{'shares':625,'current_price':1}}
    corr={'warning_pairs':[],'avg_correlation':0,'n_assets':2}
    result=detect_concentration_risk(corr,holdings,{})
    assert result['score']<75
    assert any('个股' in issue for issue in result['issues'])


def test_independent_performance_summary_uses_flow_reconciliation(tmp_path):
    from performance_tracker import generate_performance_summary
    (tmp_path/'portfolio.json').write_text(json.dumps({'_cash_flows':[
        {'date':'20261013','type':'withdraw','amount':12.34,'note':'占位'}]}),encoding='utf-8')
    assert '待核对' in generate_performance_summary(str(tmp_path))


def test_different_dates_are_not_positionally_correlated():
    data={'A':{'kline':bars()},'B':{'kline':bars(start='20240101')}}
    r=compute_var_cvar(400,data,{'A':{'shares':100,'current_price':2},'B':{'shares':100,'current_price':2}})
    assert r['error']=='insufficient_aligned_dates'
    assert compute_correlation_matrix(data)['matrix']['A']['B']==0


def test_live_rendering_has_no_legacy_orders_or_probability():
    c,s,m,l,summary,p=setup_account()
    b=bars(start='20260802')  # final date 20261015
    output=assemble_long_term_output([s], {'regime':'CRISIS','base_position':0.05,'regime_confidence':0.9},
                                      p, summary, {'etfs':{'510300':{'kline':b}}}, '20261015',
                                      inputs=(c,{'510300':m},l))
    assert output['plan']['sell_list']==output['plan']['buy_list']==[]
    assert 'base_position' not in output['timing']
    assert 'regime_confidence' not in output['timing']
    html=generate_html_report(output)
    text=generate_wechat_markdown(output)
    for document in (html,text):
        assert '强烈买入' not in document and '止损触发' not in document and '置信度90%' not in document
    assert '<table>' in html


def test_html_escapes_data():
    c,s,*_=setup_account()
    s['name']='<script>alert(1)</script>'
    html=render_long_term_html({'long_term':build(score=s)})
    assert '<script>' not in html and '&lt;script&gt;' in html


def test_live_main_uses_long_term_path_without_orders_or_email(monkeypatch,tmp_path):
    import daily_runner as runner
    c,s,m,l,summary,p=setup_account()
    s['blended_technical']=s['score']
    b=bars(start='20260802')
    all_data={'etfs':{'510300':{'kline':b,'name':'测试ETF'}}, 'stocks':{}}
    monkeypatch.setattr(runner,'TODAY','20261015')
    monkeypatch.setattr(runner,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(runner,'configure_logging',lambda:None)
    monkeypatch.setattr(runner,'is_rest_day',lambda:(False,''))
    monkeypatch.setattr(runner,'load_portfolio',lambda *args:copy.deepcopy(p))
    monkeypatch.setattr(runner,'load_inputs',lambda:(c,{'510300':m},l))
    monkeypatch.setattr(runner,'refresh_fund_quotes',lambda catalog:catalog)
    monkeypatch.setattr(runner,'collect_all_data',lambda *args,**kwargs:all_data)
    monkeypatch.setattr(runner,'fetch_sw_industry_returns',lambda **kwargs:{})
    monkeypatch.setattr(runner,'score_all_etfs_cross_sectional',lambda *args,**kwargs:[copy.deepcopy(s)])
    monkeypatch.setattr(runner,'fetch_etf_fund_nav',lambda code:None)
    monkeypatch.setattr(runner,'compute_etf_premium',lambda *args:s['premium_info'])
    class Timing:
        def __init__(self,*args,**kwargs): pass
        def position_advice(self): return {'regime':'CRISIS','base_position':0.05}
    monkeypatch.setattr(runner,'MarketTiming',Timing)
    def forbidden(*args,**kwargs): raise AssertionError('legacy order engine/email invoked')
    monkeypatch.setattr(runner,'TradeDecider',forbidden)
    monkeypatch.setattr(runner,'send_email',forbidden)
    monkeypatch.setattr(sys,'argv',['daily_runner.py','--no-email','--force'])
    runner.main()
    output=json.loads((tmp_path/'report_20261015.json').read_text(encoding='utf-8'))
    assert output['mode']=='long_term'
    assert output['scores'][0]['dca_guidance']['long_term_fit'].startswith('基础资料已核验')
    assert output['plan']['buy_list']==output['plan']['sell_list']==[]
    assert '长期ETF观察日报' in (tmp_path/'report_20261015.html').read_text(encoding='utf-8')


def dataset():
    days=[(datetime(2025,1,1)+timedelta(days=i)).strftime('%Y%m%d') for i in range(410)
          if (datetime(2025,1,1)+timedelta(days=i)).weekday()<5]
    return {'price_basis':'raw','corporate_actions_complete':True,'targets':{'A':0.6,'B':0.4},
            'instruments':{code:{'lot_size':100,'is_qdii':False,'bars':[
                {'date':d,'open':2.0,'close':2.0,'split_factor':1,'dividend_per_share':0}
                for d in days]} for code in ('A','B')}}


def test_dca_comparison_identical_flows_costs_and_no_same_day_fills():
    result=compare_dca(dataset())
    outcomes=list(result['results'].values())
    assert len({r['contributions'] for r in outcomes})==1
    for r in outcomes:
        assert r['final_wealth']==pytest.approx(r['contributions']-r['fees'])
        assert all(t['date']>t['signal_date'] for t in r['trades'])
        assert len({t['date'][:6] for t in r['trades']})==len(r['trades'])
        assert r['final_cash']>=0
        assert all(t['total_cost']<=1000 for t in r['trades'])


def test_dca_rejects_adjusted_execution_prices_and_date_holes():
    d=dataset(); d['price_basis']='qfq'
    with pytest.raises(ValueError): compare_dca(d)
    d=dataset(); d['instruments']['A']['bars'].pop(5)
    with pytest.raises(ValueError): compare_dca(d)


def test_qdii_without_historical_premium_cannot_trade():
    d=dataset()
    for instrument in d['instruments'].values(): instrument['is_qdii']=True
    results=compare_dca(d)['results']
    assert all(not r['trades'] and r['final_wealth']==r['contributions'] for r in results.values())


def test_dca_accounts_for_distributions_and_splits():
    d=dataset()
    for instrument in d['instruments'].values():
        for i,row in enumerate(instrument['bars']):
            if i==80: row['split_factor']=2
            if i>=80: row['open']=row['close']=1
            if i==100: row['dividend_per_share']=0.05
    for r in compare_dca(d)['results'].values():
        assert r['cash_dividends']>0
        assert r['final_wealth']==pytest.approx(r['contributions']-r['fees']+r['cash_dividends'])
