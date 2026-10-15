import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contribution_review import purchase_example, disclosed_overlap, contribution_priority, report_changes
from long_term import budget_status, load_inputs, build_long_term_report, format_long_term_text, render_long_term_html


def test_budget_preserves_last_confirmed_month_snapshot_without_daily_reconfirmation():
    config,_,_=load_inputs()
    ledger={'reconciled_through':'20261015','transactions':[
        {'date':'20261016','side':'buy','amount':300,'fee':5}]}
    result=budget_status(config,ledger,'20261020',800)
    assert result['known'] and result['spent']==305 and result['remaining']==695
    assert result['needs_confirmation'] and result['confirmed_through']=='20261015'
    assert result['available_for_review']==0  # one monthly order already recorded
    assert not budget_status(config,ledger,'20261101',800)['known']


def test_lot_cost_example_never_spends_nominal_budget_plus_fees():
    example=purchase_example(2,100,1000,0.1)
    assert example['shares']==400 and example['amount']==800
    assert example['commission']==5
    assert example['cash_required']==pytest.approx(805.08)
    assert example['cash_left']==pytest.approx(194.92)
    assert example['cost_fraction']==pytest.approx((5+0.08+0.4)/800)
    assert purchase_example(135,100,1000)['shares']==0


@pytest.mark.parametrize('price,lot,budget',[(0,100,1000),(2,0,1000),(float('nan'),100,1000),(2,100,None)])
def test_bad_purchase_inputs_do_not_create_share_examples(price,lot,budget):
    assert purchase_example(price,lot,budget)['shares']==0


def test_direct_overlap_is_weighted_and_labelled_as_lower_bound():
    catalog={'A':{'disclosed_stock_holdings':[{'code':'X','weight_pct':30}]},
             'B':{'disclosed_stock_holdings':[{'code':'X','weight_pct':20}]}}
    holdings=[{'code':'B','name':'已持有B','value':500}]
    result=disclosed_overlap('A',catalog,holdings,1000)
    assert result['matches'][0]['overlap_pct']==20
    assert result['weighted_overlap_pct']==10 and result['coverage']=='partial'
    assert '不等于无重叠' in disclosed_overlap('C',catalog,holdings,1000)['note']


def test_priority_does_not_reward_overheat_or_ignore_actual_cost():
    config,_,_=load_inputs()
    obs={'purchase_example':purchase_example(1.25,100,1000,0),
         'allocation_gap_pct':20,'premium_pct':0.2}
    overlap={'weighted_overlap_pct':0,'matches':[]}
    cool=contribution_priority(obs,{'indicators':{'rsi':50}},config,overlap)
    hot=contribution_priority(obs,{'indicators':{'rsi':75}},config,overlap)
    assert hot['overheated'] and hot['score']<cool['score']
    costly=copy.deepcopy(obs);costly['purchase_example']=purchase_example(2,100,417,0)
    assert contribution_priority(costly,{},config,overlap)['score']<cool['score']


def test_changes_use_previous_report_date_and_do_not_compare_premium_sources():
    history=[{'date':'20261013','scores':[{'code':'A','premium_info':{'premium_pct':5,'data_source':'official_nav','nav_date':'20261012'}}]}]
    observations=[{'code':'A','name':'A基金','premium_pct':1,'premium_source':'iopv','nav_date':'20261015'}]
    result=report_changes(observations,[],[],history,'20261015',set())
    assert result['previous_date']=='20261013' and not result['events']
    history[0]['scores'][0]['premium_info']['data_source']='iopv'
    assert report_changes(observations,[],[],history,'20261015',set())['events'][0]['kind']=='premium'
    assert not report_changes(observations,[],[],history,'20261015',{'A'})['events']


def test_draft_priorities_explain_costs_without_becoming_orders():
    config,_,_=load_inputs()
    score={'code':'A','name':'A基金','score':75,'price':1.25,'returns':{'r5d':1,'r60d':5},
           'indicators':{'rsi':50},'premium_info':{'premium_pct':0.1,'data_source':'official_nav'}}
    meta={'bucket':'a_broad','index_name':'宽基','annual_fee_pct':0.2,'aum_cny':1e9,
          'lot_size':100,'verified_on':'2026-10-15','source_url':'https://example.org/A',
          'spread_pct':0.01,'spread_as_of':'20261015150000'}
    summary={'total_assets':1000,'available_cash':1000,'holdings':[]}
    data=build_long_term_report([score],{'_cash_flows':[]},summary,'20261015',config,{'A':meta},
                               {'reconciled_through':'20261015','transactions':[]})
    assert data['allocation_status']=='draft' and data['priorities']
    assert not data['plan']['review_candidates'] and not data['plan']['buy_list']
    assert data['priorities'][0]['purchase_example']['shares']==700
    html=render_long_term_html({'long_term':data})
    assert 'ETF 观察池' in html
    assert '本月优先复核' not in html
    assert '合计需现金' not in format_long_term_text(data,summary)
