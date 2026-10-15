"""Public-source unit/date parsing and accepted accounting difference regressions."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashflow_performance import account_performance
from long_term import fund_readiness
from update_fund_catalog import parse_basic, parse_tracking, parse_holdings, fetch_spreads


def test_accepted_difference_does_not_block_or_become_a_withdrawal():
    reports=[{'date':'20261012','portfolio':{'total_assets':1000}},
             {'date':'20261013','portfolio':{'total_assets':1010}}]
    portfolio={'_cash_flows':[], '_reconciliation_adjustments':[
        {'amount':12.34,'status':'accepted_unverified'}]}
    original=copy.deepcopy(portfolio)
    result=account_performance(reports,portfolio)
    assert result['available'] and result['profit']==10
    assert result['return_pct']==pytest.approx(1)
    assert '12.34' in result['method_note'] and '未核实' in result['method_note']
    assert portfolio==original


def test_only_accepted_difference_is_exempt_other_pending_flows_still_fail():
    p={'_cash_flows':[{'date':'20261013','type':'withdraw','amount':100,'pending':True}],
       '_reconciliation_adjustments':[{'amount':12.34,'status':'accepted_unverified'}]}
    result=account_performance([],p)
    assert not result['available']


def test_reference_policy_ignores_legacy_placeholder_without_changing_account():
    reports=[{'date':'20261012','portfolio':{'total_assets':1000}},
             {'date':'20261013','portfolio':{'total_assets':1010}}]
    p={'_cash_flows':[{'date':'20260601','type':'withdraw','amount':12.34,'note':'占位'}]}
    original=copy.deepcopy(p)
    result=account_performance(reports,p,'reference_only')
    assert result['available'] and result['profit']==10 and '参考口径' in result['method_note']
    assert p==original


def test_basic_parser_keeps_fee_units_and_disclosure_dates():
    page='''<table><tr><th>基金代码</th><td>159529（主代码）</td><th>基金类型</th><td>指数型-海外股票</td></tr>
    <tr><th>基金简称</th><td>标普消费ETF</td><th>基金全称</th><td>测试ETF(QDII)</td></tr>
    <tr><th>净资产规模</th><td>10.44亿元（截止至：2026年06月30日）</td></tr>
    <tr><th>管理费率</th><td>0.50%（每年）</td><th>托管费率</th><td>0.20%（每年）</td></tr>
    <tr><th>销售服务费率</th><td>---（每年）</td><th>跟踪标的</th><td>测试指数</td></tr></table>'''
    meta,_=parse_basic(page,'159529')
    assert meta['aum_cny']==1044000000 and meta['aum_as_of']=='2026-06-30'
    assert meta['annual_fee_pct']==pytest.approx(0.7)
    assert 'sales_service_fee_pct' not in meta
    assert '---' in meta['sales_service_fee_note']
    assert 'tracking_difference_pct' not in meta
    with pytest.raises(ValueError):parse_basic(page,'513800')


def test_missing_tracking_error_never_becomes_zero_or_difference():
    page='<div><table><tr><th>跟踪指数</th><th>年化跟踪误差</th><th>同类</th></tr><tr><td>东证</td><td>--</td><td>--</td></tr></table><p>截止至：2026-10-14</p></div>'
    meta=parse_tracking(page)
    assert meta['tracking_error_annual_pct'] is None
    assert meta['tracking_error_as_of']=='2026-10-14'
    assert 'tracking_difference_pct' not in meta


def test_stock_holdings_parser_uses_disclosure_weights_not_stock_price():
    content="<div class='boxitem'><h4>股票组合 2026-06-30</h4><table><tr><th>序号</th><th>代码</th><th>名称</th><th>最新价</th><th>占净值比例</th></tr><tr><td>1</td><td>A</td><td>甲</td><td>999</td><td>25.00%</td></tr><tr><td>2</td><td>B</td><td>乙</td><td>888</td><td>15.00%</td></tr></table></div>"
    meta=parse_holdings('var apidata={content:'+json.dumps(content)+'};')
    assert meta['top10_weight_pct']==40
    assert meta['holdings_as_of']=='2026-06-30'
    empty=parse_holdings('var apidata={content:""};')
    assert empty['top10_weight_pct'] is None


def test_quote_spread_uses_bid_ask_midpoint_and_preserves_time(monkeypatch):
    fields=['']*88
    fields[2]='513800';fields[9]='1.852';fields[19]='1.853';fields[30]='20261015150000'
    class Response:
        text='v_sh513800="'+'~'.join(fields)+'";'
        def raise_for_status(self):pass
    monkeypatch.setattr('update_fund_catalog.requests.get',lambda *a,**kw:Response())
    quote=fetch_spreads(['513800'])['513800']
    assert quote['spread_pct']==pytest.approx(0.001/1.8525*100)
    assert quote['spread_as_of']=='20261015150000'


def test_basic_readiness_does_not_require_research_proof_or_stock_data_for_gold():
    meta={'index_name':'黄金9999','annual_fee_pct':0.6,'aum_cny':1e9,'lot_size':100,
          'source_url':'https://fundf10.eastmoney.com/jbgk_518880.html','verified_on':'2026-10-15',
          'top10_weight_pct':None,'tracking_difference_pct':None,'stock_concentration_applicable':False}
    status=fund_readiness(meta,'20261015')
    assert status['ready'] and not status['missing']
    assert status['supplemental_notes']
