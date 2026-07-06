import sys, traceback, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

all_pass = True

def test(name, fn):
    global all_pass
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        all_pass = False

# 1. quant_engine imports
test("quant_engine import", lambda: __import__('quant_engine'))

# 2. risk_engine imports
test("risk_engine import", lambda: __import__('risk_engine'))

# 3. data_engine imports
test("data_engine import", lambda: __import__('data_engine'))

# 4. backtest_engine imports
test("backtest_engine import", lambda: __import__('backtest_engine'))

# 5. optimizer imports
test("optimizer import", lambda: __import__('optimizer'))

# 6. performance_tracker imports
test("performance_tracker import", lambda: __import__('performance_tracker'))

# 7. report_mailer imports
test("report_mailer import", lambda: __import__('report_mailer'))

# 8. quant_engine computation
def test_quant_compute():
    from quant_engine import (compute_indicators, score_factors,
                              score_etf_comprehensive, adx,
                              score_all_etfs_cross_sectional)
    np.random.seed(42)
    c = list(np.cumsum(np.random.randn(100)*0.01)+10)
    h = [x*1.01 for x in c]
    l = [x*0.99 for x in c]
    v = list(np.random.randint(10000,20000,100))
    ind = compute_indicators(c,h,l,v)
    assert ind["rsi"] is not None
    assert ind["volatility"] >= 0
    f = score_factors(ind)
    assert len(f) == 16
    result = score_etf_comprehensive("000001","test",c,h,l,v)
    assert "score" in result
    a = adx(c,h,l)
    assert 0 <= a <= 100
    # cross-sectional
    etf_dict = {"A":{"name":"A","kline":[{"close":float(cc),"high":float(cc*1.01),"low":float(cc*0.99),"volume":int(vv)} for cc,vv in zip(c,v)]}}
    res = score_all_etfs_cross_sectional(etf_dict)
    assert len(res) >= 1
    print(f"    rsi={ind['rsi']:.1f} adx={a:.1f} score={result['score']}")

test("quant_engine compute", test_quant_compute)

# 9. risk_engine computation
def test_risk_compute():
    from risk_engine import compute_correlation_matrix, compute_var_cvar, daily_returns, percentile
    np.random.seed(42)
    c = np.cumsum(np.random.randn(100)*0.01) + 10
    rets = daily_returns(c.tolist())
    assert len(rets) == 99
    p = percentile(rets, 5)
    assert p < 0
    etf_data = {"A":{"name":"A","kline":[{"close":float(x)} for x in c[-61:]]}}
    corr = compute_correlation_matrix(etf_data)
    assert "avg_correlation" in corr
    var = compute_var_cvar(10000, etf_data, {"A":{"shares":100,"current_price":10}})
    assert "var_95" in var
    print(f"    avg_corr={corr['avg_correlation']:.3f} var_95={var['var_95']:.1f}")

test("risk_engine compute", test_risk_compute)

# 10. optimizer spearman_ic (rank_values is internal nested function)
def test_rankdata():
    from optimizer import spearman_ic
    # Test spearman: perfect positive correlation
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    ic = spearman_ic(x, y)
    assert abs(ic - 1.0) < 0.01, f"IC should be ~1.0, got {ic}"
    y_rev = list(reversed(y))
    ic_rev = spearman_ic(x, y_rev)
    assert abs(ic_rev + 1.0) < 0.01, f"IC should be ~-1.0, got {ic_rev}"
    print(f"    spearman OK (ic={ic:.3f}, ic_rev={ic_rev:.3f})")

test("optimizer rankdata/spearman", test_rankdata)

# 11. daily_runner basic import (functions needed by main)
def test_daily_runner():
    from daily_runner import (load_portfolio, update_portfolio_prices,
                               compute_portfolio_summary, analyze_watchlist_etf,
                               is_rest_day)
    print("    daily_runner functions import OK")

test("daily_runner functions", test_daily_runner)

print()
print("=" * 50)
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
