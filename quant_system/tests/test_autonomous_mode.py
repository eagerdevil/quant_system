"""
9/19全自动模式测试 — 模拟盘 TradeDecider(autonomous=True)

背景(9/19诊断): 模拟盘自8/31起零交易14个交易日。根因是原设计里"目标仓位"只是一条
单向约束 —— 买入预算=目标-现有持仓(下限0), 卖出全是事件驱动(止损/溢价/评分/熔断)。
9/2模型目标降到20%而实持40%, 于是买入预算恒为0、卖出条件一条不触发, 账户彻底僵住。

本次改造(仅模拟盘, autonomous=True):
  1) 目标仓位双向收敛 — 持仓超过"目标+死区(总资产5%)"时, 每日减超额的1/3(约3日收敛)
  2) TREND_DOWN 不再硬禁买 — 放宽为"只准A级 + 规模减半"(CRISIS 维持禁止)
  3) 单笔最低金额 min_order_amount — 低于门槛的零钱补仓不成单

三条硬底线(用户9/19决策保留, 本改造不动): 止损 / 单只+总仓位上限 / 单笔最低金额。

实盘日报(daily_runner)不传 autonomous → 逐字保持改造前行为, 本文件含回归保护。

运行: pytest tests/test_autonomous_mode.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from quant_engine import (
    TradeDecider, AUTONOMOUS_CONFIG, OPTIMIZED_PARAMS, SYSTEM_CONFIG,
)

# 门槛从实际配置读取（WEIGHT_CONFIG 可覆盖），不硬编码 78/65
_TH = OPTIMIZED_PARAMS["grade_thresholds"]
A_MIN = _TH.get("A_强烈买入", 78)
B_MIN = _TH.get("B_买入", 65)
SCORE_A = A_MIN + 5                       # 稳过A级
SCORE_B = (A_MIN + B_MIN) // 2            # 稳落在B级区间
assert B_MIN <= SCORE_B < A_MIN, "B级测试分数构造失效, 检查grade_thresholds配置"


# ================================================================
# 脚手架
# ================================================================
def _score(code, score=SCORE_A, price=1.0, sortino=1.2, vol=25.0, name=None):
    """单只ETF评分记录（字段对齐生产数据流）"""
    return {
        "code": code, "name": name or code, "score": score,
        "grade": "A_强烈买入" if score >= A_MIN else "B_买入",
        "price": price,
        "indicators": {"sortino": sortino, "volatility_pct": vol},
        "returns": {"r20d": 0.0, "r5d": 0.0},
        "premium_info": {},
    }


def _timing(base_position=0.2, regime="CHOPPY", regime_buy_grade_min="B_买入",
            regime_stop_loss=-0.08, circuit_breaker=None):
    return {
        "base_position": base_position,
        "regime": regime,
        "regime_buy_grade_min": regime_buy_grade_min,
        "regime_stop_loss": regime_stop_loss,
        "circuit_breaker": circuit_breaker or {},
        "advice": "测试用择时",
    }


def _holdings(*specs):
    """(code, shares, price, cost) → portfolio 字典; cost缺省=price*1.01(小亏1%, 不触发止损)"""
    p = {}
    for spec in specs:
        code, shares, price = spec[0], spec[1], spec[2]
        cost = spec[3] if len(spec) > 3 else round(price * 1.01, 4)
        p[code] = {"code": code, "name": code, "shares": shares,
                   "cost": cost, "current_price": price}
    return p


def _decide(scores, timing, portfolio, autonomous, min_order_amount=0.0, total=500_000.0):
    d = TradeDecider(scores, timing, portfolio, autonomous=autonomous)
    return d.generate_plan(total_capital=total, min_order_amount=min_order_amount)


# ================================================================
# 1. 目标仓位双向收敛 (trim)
# ================================================================
class TestTargetPositionTrim:
    """持仓40% vs 目标20%: 死区25,000元, 超额100,000元 → 本日减33,333元"""

    TOTAL = 500_000.0

    @staticmethod
    def _over_target(**kw):
        """3只持仓共200,000元(40%), 目标20%(100,000) — 复刻模拟盘9/2-9/18的僵局"""
        return _holdings(("510300", 30_000, 4.0),      # 120,000 (60%)
                         ("513800", 50_000, 1.0),      #  50,000 (25%)
                         ("512890", 30_000, 1.0))      #  30,000 (15%)

    @staticmethod
    def _scores_for(port, **kw):
        return [_score(c, price=port[c]["current_price"]) for c in port]

    def test_trim_triggered_when_over_target(self):
        """超目标+死区 → 生成减仓卖单, 合计≈超额1/3"""
        port = self._over_target()
        plan = _decide(self._scores_for(port), _timing(), port, autonomous=True, total=self.TOTAL)

        cuts = [s for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]]
        assert cuts, "超额100,000元(>死区25,000)必须触发减仓"

        cut_value = sum(s["shares"] * s["price"] for s in cuts)
        expected = (200_000 - self.TOTAL * 0.2) * AUTONOMOUS_CONFIG["trim_batch"]
        # 每只按100股向下取整, 单只最多少减1手(=100×价格); 浮点截断还可能再少1手
        tolerance = sum(s["price"] * 100 for s in cuts) + 200
        assert abs(cut_value - expected) < tolerance, "合计减仓额应≈超额1/3(100股取整误差)"
        assert cut_value <= expected, "截断只会少减不会多减(宁可收敛慢一天, 不可超卖)"

    def test_trim_is_proportional_and_100_share_aligned(self):
        port = self._over_target()
        plan = _decide(self._scores_for(port), _timing(), port, autonomous=True, total=self.TOTAL)

        cuts = {s["code"]: s for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]}
        assert set(cuts) == {"510300", "513800", "512890"}
        for c in cuts.values():
            assert c["shares"] % 100 == 0, "必须整手"
            assert c["shares"] < port[c["code"]]["shares"], "是减仓不是清仓"
        # 按持仓市值分摊: 510300占60% → 减持金额约为513800(25%)的2.4倍
        v_510300 = cuts["510300"]["shares"] * cuts["510300"]["price"]
        v_513800 = cuts["513800"]["shares"] * cuts["513800"]["price"]
        assert abs(v_510300 / v_513800 - 0.60 / 0.25) < 0.15

    def test_trim_uses_partial_shares_not_full_position(self):
        """回归保护: 减仓必须是部分卖出(原熔断块也是部分), 不能整只清掉 —
        否则3只各减1/3会变成"一天清仓", 与分批收敛设计相悖"""
        port = self._over_target()
        plan = _decide(self._scores_for(port), _timing(), port, autonomous=True, total=self.TOTAL)
        for s in plan["sell_list"]:
            assert s["shares"] < port[s["code"]]["shares"]

    def test_no_trim_inside_dead_band(self):
        """超额20,000元 < 死区25,000元(总资产5%) → 不动手, 防抖动交易"""
        port = _holdings(("510300", 20_000, 4.0),      # 80,000
                         ("513800", 40_000, 1.0))      # 40,000  → 合计120,000 (超目标20,000=4%)
        plan = _decide([_score(c, price=port[c]["current_price"]) for c in port],
                       _timing(), port, autonomous=True, total=self.TOTAL)
        assert [s for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]] == []

    def test_no_trim_at_or_below_target(self):
        """持仓低于目标 → 无减仓, 也不应为负"""
        port = _holdings(("510300", 20_000, 4.0),      # 80,000
                         ("513800", 10_000, 1.0))      # 10,000 → 90,000 < 目标100,000
        plan = _decide([_score(c, price=port[c]["current_price"]) for c in port],
                       _timing(), port, autonomous=True, total=self.TOTAL)
        assert plan["sell_list"] == []

    def test_trim_skips_position_below_min_order_amount(self):
        """分到的小额减仓 < 单笔最低金额 → 跳过(该只不减, 其余照减)"""
        port = _holdings(("510300", 29_250, 4.0),      # 117,000
                         ("513800", 50_000, 1.0),      #  50,000
                         ("512890", 30_000, 1.0),      #  30,000
                         ("159920", 3_000, 1.0))       #   3,000 (1.5% → 分到~500元 < 2,000)
        scores = [_score(c, price=port[c]["current_price"]) for c in port]
        plan = _decide(scores, _timing(), port, autonomous=True,
                       min_order_amount=2_000.0, total=self.TOTAL)
        cuts = {s["code"] for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]}
        assert cuts, "大额持仓应照常减仓"
        assert "159920" not in cuts, "低于最低金额的小额减仓应跳过"

    def test_trim_skips_when_cut_rounds_below_one_lot(self):
        """分摊后不足100股 → 跳过"""
        port = _holdings(("510300", 29_900, 4.0),      # 119,600
                         ("513800", 50_400, 1.0),      #  50,400
                         ("512890", 30_000, 1.0),      #  30,000
                         ("159920", 300, 10.0))        #   3,000 → 分摊~500元/10元=50股 <100
        scores = [_score(c, price=port[c]["current_price"]) for c in port]
        plan = _decide(scores, _timing(), port, autonomous=True, total=self.TOTAL)
        cuts = {s["code"] for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]}
        assert "159920" not in cuts

    def test_trimmed_code_is_not_rebought_same_day(self):
        """当日减仓的标的当日不得再买回(否则白付一轮手续费)"""
        port = self._over_target()
        plan = _decide(self._scores_for(port), _timing(), port, autonomous=True, total=self.TOTAL)
        cut_codes = {s["code"] for s in plan["sell_list"]}
        assert not (cut_codes & {b["code"] for b in plan["buy_list"]})

    def test_over_target_produces_no_buys(self):
        """超目标时买入预算为0 — 这是改造前的僵局来源, 但买入侧行为不应改变"""
        port = self._over_target()
        plan = _decide(self._scores_for(port), _timing(), port, autonomous=True, total=self.TOTAL)
        assert plan["buy_list"] == []

    def test_circuit_breaker_takes_precedence_over_trim(self):
        """熔断触发时走熔断块(一次性降到目标), 不再叠加1/3分批减仓"""
        port = self._over_target()
        cb = {"triggered": True, "level": 1, "drawdown": -0.12}
        plan = _decide(self._scores_for(port), _timing(circuit_breaker=cb), port,
                       autonomous=True, total=self.TOTAL)

        assert plan["sell_list"], "熔断降仓应生成卖单"
        assert all("熔断降仓" in s["reason"] for s in plan["sell_list"])
        assert not [s for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]]
        # 每只只出现一次
        codes = [s["code"] for s in plan["sell_list"]]
        assert len(codes) == len(set(codes))
        # 熔断减的是全部超额(不是1/3)
        cut_value = sum(s["shares"] * s["price"] for s in plan["sell_list"])
        assert cut_value > (200_000 - self.TOTAL * 0.2) * 0.7


# ================================================================
# 2. 实盘路径回归保护 — autonomous=False 行为逐字不变
# ================================================================
class TestNonAutonomousUnchanged:
    TOTAL = 500_000.0

    def _over_target(self):
        return _holdings(("510300", 30_000, 4.0), ("513800", 50_000, 1.0),
                         ("512890", 30_000, 1.0))

    def test_non_autonomous_never_trims_to_target(self):
        """实盘日报路径: 超目标持仓不产生减仓(改造前行为, 用户决策: 先只放模拟盘)"""
        port = self._over_target()
        scores = [_score(c, price=port[c]["current_price"]) for c in port]
        plan = _decide(scores, _timing(), port, autonomous=False, total=self.TOTAL)
        assert plan["sell_list"] == []

    def test_default_constructor_signature_still_works(self):
        """daily_runner 的三位置参数调用方式(不传autonomous)必须继续可用"""
        port = self._over_target()
        scores = [_score(c, price=port[c]["current_price"]) for c in port]
        decider = TradeDecider(scores, _timing(), port)
        assert decider.autonomous is False
        plan = decider.generate_plan()          # 连total_capital都不传
        assert plan["sell_list"] == []
        assert set(plan) >= {"buy_list", "sell_list", "hold_list", "target_position"}

    def test_default_min_order_amount_is_zero(self):
        """不传 min_order_amount 时门槛为0 — 老调用方行为不变"""
        scores = [_score("510300")]
        plan_a = _decide(scores, _timing(base_position=0.5), {}, autonomous=False, total=100_000.0)
        plan_b = _decide(scores, _timing(base_position=0.5), {}, autonomous=False,
                         min_order_amount=0.0, total=100_000.0)
        assert plan_a["buy_list"] == plan_b["buy_list"] != []


# ================================================================
# 3. TREND_DOWN 放宽建仓 (只准A级 + 规模减半)
# ================================================================
class TestTrendDownBuyGate:
    TOTAL = 100_000.0

    def _td(self, **kw):
        return _timing(base_position=0.5, regime="TREND_DOWN", regime_buy_grade_min=None, **kw)

    def test_trend_down_still_blocks_non_autonomous(self):
        """实盘路径: TREND_DOWN 硬禁买(regime_buy_grade_min=None)"""
        plan = _decide([_score("510300")], self._td(), {}, autonomous=False, total=self.TOTAL)
        assert plan["buy_list"] == []

    def test_trend_down_autonomous_allows_a_grade(self):
        plan = _decide([_score("510300", score=SCORE_A)], self._td(), {},
                       autonomous=True, total=self.TOTAL)
        assert len(plan["buy_list"]) == 1
        assert plan["buy_list"][0]["code"] == "510300"

    def test_trend_down_autonomous_rejects_b_grade(self):
        """放宽只到A级 — B级仍不得在下跌趋势里建仓"""
        plan = _decide([_score("510300", score=SCORE_B)], self._td(), {},
                       autonomous=True, total=self.TOTAL)
        assert plan["buy_list"] == []

    def test_trend_down_autonomous_halves_position_size(self):
        """同评分同波动下: TREND_DOWN全自动 的仓位应是 正常市况 的一半"""
        s = [_score("510300", score=SCORE_A)]
        td = _decide(s, self._td(), {}, autonomous=True, total=self.TOTAL)
        normal = _decide(s, _timing(base_position=0.5, regime="CHOPPY",
                                    regime_buy_grade_min="B_买入"),
                         {}, autonomous=False, total=self.TOTAL)

        assert td["buy_list"] and normal["buy_list"]
        td_pct = td["buy_list"][0]["final_pct"]
        nm_pct = normal["buy_list"][0]["final_pct"]
        assert abs(td_pct * 2 - nm_pct) <= 0.15, f"{td_pct}% 应为 {nm_pct}% 的一半"
        assert td["buy_list"][0]["amount"] < normal["buy_list"][0]["amount"]

    def test_crisis_still_blocks_buy_when_autonomous(self):
        """CRISIS 现金为王 — 全自动模式也不放宽(只放开TREND_DOWN)"""
        crisis = _timing(base_position=0.1, regime="CRISIS", regime_buy_grade_min=None)
        plan = _decide([_score("510300", score=SCORE_A)], crisis, {},
                       autonomous=True, total=self.TOTAL)
        assert plan["buy_list"] == []

    def test_no_double_halving_when_regime_already_allows_buying(self):
        """CHOPPY 只准A级时 regime_buy_grade_min 非None → 不触发降规模(不重复减半)"""
        s = [_score("510300", score=SCORE_A)]
        chop_a = _decide(s, _timing(base_position=0.5, regime="CHOPPY",
                                    regime_buy_grade_min="A_强烈买入"),
                         {}, autonomous=True, total=self.TOTAL)
        chop_b = _decide(s, _timing(base_position=0.5, regime="CHOPPY",
                                    regime_buy_grade_min="B_买入"),
                         {}, autonomous=False, total=self.TOTAL)
        assert chop_a["buy_list"] and chop_b["buy_list"]
        assert chop_a["buy_list"][0]["final_pct"] == chop_b["buy_list"][0]["final_pct"]


# ================================================================
# 4. 单笔最低金额
# ================================================================
class TestMinOrderAmount:
    TOTAL = 100_000.0

    def _plan(self, min_order_amount):
        return _decide([_score("510300", score=SCORE_A)], _timing(base_position=0.5), {},
                       autonomous=True, min_order_amount=min_order_amount, total=self.TOTAL)

    def test_order_at_threshold_passes(self):
        base = self._plan(0.0)
        assert base["buy_list"], "基准场景应产生买单"
        amount = base["buy_list"][0]["amount"]

        assert self._plan(amount)["buy_list"], "恰好等于门槛应放行(判据是 < )"

    def test_order_below_threshold_skipped(self):
        base = self._plan(0.0)
        amount = base["buy_list"][0]["amount"]
        assert self._plan(amount + 1)["buy_list"] == [], "低于门槛的零钱补仓不成单"

    def test_min_order_amount_does_not_block_sells(self):
        """最低金额只约束买入; 减仓侧按其自身门槛判断, 不受此参数误伤"""
        port = _holdings(("510300", 30_000, 4.0), ("513800", 50_000, 1.0),
                         ("512890", 30_000, 1.0))
        scores = [_score(c, price=port[c]["current_price"]) for c in port]
        plan = _decide(scores, _timing(), port, autonomous=True,
                       min_order_amount=2_000.0, total=500_000.0)
        assert [s for s in plan["sell_list"] if "目标仓位收敛" in s["reason"]]


# ================================================================
# 5. 三条硬底线仍然生效(用户9/19明确保留)
# ================================================================
class TestHardFloorsIntact:
    TOTAL = 100_000.0

    def test_stop_loss_still_fires_in_autonomous_mode(self):
        """止损: -9% 持仓在 TREND_DOWN 全自动下照常清仓"""
        port = _holdings(("510300", 10_000, 4.0, 4.4))   # cost4.4 → -9.1%
        scores = [_score("510300", price=4.0)]
        plan = _decide(scores, _timing(base_position=0.5, regime="TREND_DOWN",
                                       regime_buy_grade_min=None),
                       port, autonomous=True, total=self.TOTAL)
        sells = {s["code"]: s for s in plan["sell_list"]}
        assert "510300" in sells and "止损" in sells["510300"]["reason"]
        assert sells["510300"]["shares"] == 10_000, "止损是清仓"

    def test_max_single_position_cap_holds(self):
        """单只上限25%: 评分再高也不会买到25%以上"""
        plan = _decide([_score("510300", score=95, sortino=3.0, vol=10.0)],
                       _timing(base_position=1.0), {}, autonomous=True, total=self.TOTAL)
        b = plan["buy_list"][0]
        assert b["amount"] <= self.TOTAL * 0.25 + 1

    def test_max_total_holdings_cap_holds(self):
        """总持仓只数上限: 满仓时不再新建仓"""
        n = SYSTEM_CONFIG.get("max_total_holdings", 5)
        codes = [f"51030{i}" for i in range(n)]
        port = _holdings(*[(c, 1_000, 1.0) for c in codes])
        extra = [_score(c, price=1.0) for c in codes] + [_score("999999", price=1.0)]
        plan = _decide(extra, _timing(base_position=1.0), port, autonomous=True, total=self.TOTAL)
        assert "999999" not in {b["code"] for b in plan["buy_list"]}
