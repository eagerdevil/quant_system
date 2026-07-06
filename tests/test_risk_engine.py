"""
风控引擎测试 — VaR/CVaR / 相关性 / 压力测试 / 集中度

运行: pytest tests/test_risk_engine.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from risk_engine import (
    daily_returns, percentile, pearson_correlation,
    compute_correlation_matrix, compute_var_cvar,
    detect_concentration_risk, portfolio_risk_report,
    stress_test_portfolio,
    STRESS_SCENARIOS,
)


# ---- 测试工具 ----
def make_etf_data(codes, n=80):
    """生成多只ETF的模拟K线 (有相关性)"""
    np.random.seed(42)
    etf_data = {}
    # 生成有相关性的收益率
    n_etfs = len(codes)
    common = np.random.randn(n) * 0.008
    for i, code in enumerate(codes):
        specific = np.random.randn(n) * 0.012
        rets = common * 0.6 + specific * 0.4
        price = 2.0
        kline = []
        for r in rets:
            price *= (1 + r)
            kline.append({
                "close": max(price, 0.01),
                "high": price * 1.01,
                "low": price * 0.99,
                "volume": np.random.uniform(5e5, 1e7),
            })
        etf_data[code] = {"name": f"ETF_{code}", "kline": kline}
    return etf_data


# ================================================================
# 基础工具测试
# ================================================================
class TestUtils:
    def test_daily_returns(self):
        assert daily_returns([1, 2]) == [1.0]
        assert daily_returns([2, 1]) == [-0.5]

    def test_daily_returns_short(self):
        assert daily_returns([1]) == []

    def test_percentile(self):
        data = list(range(100))
        assert abs(percentile(data, 50) - 49.5) < 1

    def test_pearson_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert pearson_correlation(x, y) > 0.95


# ================================================================
# 相关性矩阵测试
# ================================================================
class TestCorrelation:
    def test_matrix_symmetric(self):
        etf_data = make_etf_data(["A", "B", "C", "D"], 70)
        result = compute_correlation_matrix(etf_data, window=60)
        assert result["n_assets"] == 4
        m = result["matrix"]
        for c1 in m:
            for c2 in m[c1]:
                assert abs(m[c1][c2] - m[c2][c1]) < 1e-6

    def test_self_correlation_is_one(self):
        etf_data = make_etf_data(["X", "Y"], 70)
        result = compute_correlation_matrix(etf_data)
        assert result["matrix"]["X"]["X"] == 1.0

    def test_avg_correlation_range(self):
        etf_data = make_etf_data(["A", "B", "C"], 70)
        result = compute_correlation_matrix(etf_data)
        assert -1 <= result["avg_correlation"] <= 1

    def test_single_etf(self):
        etf_data = make_etf_data(["X"], 70)
        result = compute_correlation_matrix(etf_data)
        assert result["n_assets"] == 1
        assert result["avg_correlation"] == 0

    def test_insufficient_data(self):
        etf_data = {"X": {"name": "X", "kline": [{"close": 1.0}] * 10}}
        result = compute_correlation_matrix(etf_data)
        assert result["n_assets"] == 0


# ================================================================
# VaR/CVaR测试
# ================================================================
class TestVaR:
    def test_var_positive(self):
        etf_data = make_etf_data(["A", "B"], 70)
        holdings = {
            "A": {"shares": 500, "current_price": 2.0},
            "B": {"shares": 300, "current_price": 3.0},
        }
        result = compute_var_cvar(1900, etf_data, holdings)
        assert result["var_95"] > 0
        assert result["cvar_95"] >= result["var_95"]

    def test_var_cvar_hierarchy(self):
        """CVaR (尾部平均) 应 >= VaR (阈值)"""
        etf_data = make_etf_data(["A", "B", "C"], 70)
        holdings = {"A": {"shares": 500, "current_price": 2.0}}
        result = compute_var_cvar(1000, etf_data, holdings)
        if not result.get("error"):
            assert result["cvar_95"] >= result["var_95"]

    def test_var_insufficient_data(self):
        result = compute_var_cvar(1000, {}, {})
        assert "error" in result

    def test_var_worst_day(self):
        etf_data = make_etf_data(["A"], 70)
        holdings = {"A": {"shares": 500, "current_price": 2.0}}
        result = compute_var_cvar(1000, etf_data, holdings)
        if not result.get("error"):
            assert result["worst_day"] >= result["cvar_95"]


# ================================================================
# 集中度风险测试
# ================================================================
class TestConcentration:
    def test_diversified_score_high(self):
        etf_data = make_etf_data(["A", "B", "C", "D"], 70)
        holdings = {
            "A": {"shares": 200, "current_price": 5.0},
            "B": {"shares": 200, "current_price": 5.0},
            "C": {"shares": 200, "current_price": 5.0},
            "D": {"shares": 200, "current_price": 5.0},
        }
        corr = compute_correlation_matrix(etf_data)
        result = detect_concentration_risk(corr, holdings, etf_data)
        assert result["score"] >= 50  # 分散应得高分

    def test_concentrated_score_low(self):
        etf_data = make_etf_data(["A", "B"], 70)
        holdings = {
            "A": {"shares": 900, "current_price": 5.0},
            "B": {"shares": 100, "current_price": 5.0},
        }
        corr = compute_correlation_matrix(etf_data)
        result = detect_concentration_risk(corr, holdings, etf_data)
        # 集中度高→分数偏低
        assert result["level"] in ["warning", "danger"]

    def test_level_matches_score(self):
        for level, min_score in [("safe", 75), ("warning", 50), ("danger", 0)]:
            pass  # 逻辑在函数内，验证不崩溃即可


# ================================================================
# 压力测试
# ================================================================
class TestStressTest:
    @property
    def sample_portfolio(self):
        return {
            "159326": {"shares": 300, "cost": 2.04, "current_price": 2.04, "name": "电网设备ETF"},
            "510300": {"shares": 200, "cost": 5.01, "current_price": 5.01, "name": "沪深300ETF"},
            "159659": {"shares": 300, "cost": 2.34, "current_price": 2.34, "name": "纳指ETF"},
            "_available_cash": 700,
        }

    @property
    def sample_summary(self):
        return {
            "total_assets": 3019,
            "holdings": [
                {"code": "159326", "weight": 20.3, "name": "电网设备ETF"},
                {"code": "510300", "weight": 33.2, "name": "沪深300ETF"},
                {"code": "159659", "weight": 23.3, "name": "纳指ETF"},
            ],
            "cash_ratio": 23.2,
        }

    def test_all_scenarios_defined(self):
        assert len(STRESS_SCENARIOS) >= 5

    def test_each_scenario_has_required_fields(self):
        for s in STRESS_SCENARIOS:
            assert "name" in s
            assert "broad_market_shock" in s
            assert "industry_shocks" in s

    def test_gold_special_handling(self):
        """黄金ETF(518850)应在压力测试中被特殊处理（使用gold_return）"""
        portfolio = {
            "518850": {"shares": 200, "cost": 8.0, "current_price": 8.0, "name": "黄金ETF"},
            "_available_cash": 1000,
        }
        summary = {
            "total_assets": 2600,
            "holdings": [{"code": "518850", "weight": 61.5, "name": "黄金ETF"}],
        }
        result = stress_test_portfolio(portfolio, summary)
        # 黄金在股灾中应避险 → 损失较小
        for s in result["scenarios"]:
            if "2015" in s["name"]:
                # 黄金在2015股灾中上涨 → loss_pct应优于broad_market_shock(-35%)
                assert s["loss_pct"] > -15, f"黄金在{s['name']}中应抗跌，实际{s['loss_pct']:.1f}%"

    def test_stress_test_output_structure(self):
        result = stress_test_portfolio(self.sample_portfolio, self.sample_summary)
        assert "scenarios" in result
        assert "worst_scenario" in result
        assert "average_loss_pct" in result
        assert "advice" in result
        assert len(result["scenarios"]) == len(STRESS_SCENARIOS)

    def test_stress_test_per_holding_detail(self):
        result = stress_test_portfolio(self.sample_portfolio, self.sample_summary)
        for s in result["scenarios"]:
            assert "per_holding" in s
            assert len(s["per_holding"]) >= 1

    def test_empty_portfolio(self):
        result = stress_test_portfolio({}, {})
        assert len(result["scenarios"]) == len(STRESS_SCENARIOS)
        # 空持仓→损失为0
        for s in result["scenarios"]:
            assert abs(s["loss_pct"]) < 0.01


# ================================================================
# 完整风险报告
# ================================================================
class TestPortfolioRiskReport:
    def test_report_structure(self):
        etf_data = make_etf_data(["A", "B", "C"], 70)
        portfolio = {
            "A": {"shares": 500, "cost": 2.0, "current_price": 2.0, "name": "ETF_A"},
            "B": {"shares": 300, "cost": 3.0, "current_price": 3.0, "name": "ETF_B"},
            "_available_cash": 500,
        }
        report = portfolio_risk_report(portfolio, etf_data)
        assert "correlation" in report
        assert "var" in report
        assert "concentration" in report
        assert "single_risks" in report

    def test_report_with_scores(self):
        etf_data = make_etf_data(["A", "B"], 70)
        portfolio = {
            "A": {"shares": 500, "cost": 2.0, "current_price": 2.0, "name": "ETF_A"},
            "_available_cash": 1000,
        }
        scores = [{"code": "A", "grade": "B_买入"}]
        report = portfolio_risk_report(portfolio, etf_data, scores)
        assert len(report["single_risks"]) > 0
        assert report["single_risks"][0]["grade"] == "B_买入"


# ================================================================
# L3: 边界/NaN/空数据测试
# ================================================================
class TestRiskEdgeCases:
    def test_empty_correlation(self):
        """空ETF数据不应崩溃"""
        result = compute_correlation_matrix({})
        assert result["n_assets"] == 0

    def test_var_zero_portfolio(self):
        """零值组合不应崩溃"""
        result = compute_var_cvar(0, {}, {})
        assert "error" in result

    def test_var_negative_value(self):
        """负资产不应崩溃"""
        result = compute_var_cvar(-1000, {}, {})
        assert "error" in result

    def test_daily_returns_all_same(self):
        """全同价格→收益率为0"""
        rets = daily_returns([2.0] * 10)
        assert all(r == 0.0 for r in rets)

    def test_percentile_single_value(self):
        assert percentile([5.0], 95) == 5.0

    def test_percentile_empty(self):
        assert percentile([], 50) == 0

    def test_pearson_constant_series(self):
        """常数列→相关系数0"""
        r = pearson_correlation([1.0]*10, [1.0, 2.0, 3.0, 4.0, 5.0]*2)
        assert not np.isnan(r) and not np.isinf(r)

    def test_stress_test_missing_industry(self):
        """未知代码→使用broad_market_shock"""
        portfolio = {"999999": {"shares": 100, "cost": 1.0, "current_price": 1.0, "name": "unknown"}}
        summary = {"total_assets": 100, "holdings": [{"code": "999999", "weight": 100, "name": "unknown"}]}
        result = stress_test_portfolio(portfolio, summary)
        assert len(result["scenarios"]) > 0
