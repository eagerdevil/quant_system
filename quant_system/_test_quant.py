
import sys, numpy as np
sys.path.insert(0, r"E:\Claudecode\quant\quant_system")
from quant_engine import *

np.random.seed(42)
closes = list(np.cumsum(np.random.randn(100) * 0.01) + 10)
highs = [c * 1.01 for c in closes]
lows = [c * 0.99 for c in closes]
volumes = list(np.random.randint(10000, 20000, 100))
ind = compute_indicators(closes, highs, lows, volumes)
print("rsi:", ind["rsi"])
print("volatility:", ind["volatility"])
print("sharpe:", ind["sharpe"])
print("sortino:", ind["sortino"])
print("max_drawdown:", ind["max_drawdown"])
factors = score_factors(ind)
print("factors count:", len(factors))
result = score_etf_comprehensive("000001", "test", closes, highs, lows, volumes)
print("score:", result["score"])
print("grade:", result["grade"])
a = adx(closes, highs, lows)
print("adx:", a)

idx_data = {"000300": {"data": [{"date": "20240101", "close": c, "high": c*1.005, "low": c*0.995} for c in closes]}}
mt = MarketTiming(idx_data, [], 15000, {"limit_down": 5}, {"change": 100})
advice = mt.position_advice()
print("timing base_position:", advice["base_position"])
print("timing regime:", advice["regime"])

etf_dict = {"000001": {"name": "ETF1", "kline": [{"close": c, "high": c*1.01, "low": c*0.99, "volume": v} for c, v in zip(closes, volumes)]}}
results = score_all_etfs_cross_sectional(etf_dict)
print("cross-sectional results:", len(results))
print("ALL TESTS PASSED")
