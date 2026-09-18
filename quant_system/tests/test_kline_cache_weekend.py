"""
9/19修复回归测试 — 周末缓存滞后导致K线停在上一个交易日之前

背景: _fetch_kline_with_cache 原逻辑"周末必休市 → 无条件返回缓存",
假设缓存必然停在周五。但缓存可能停在更早的日期:
本地9/17凌晨跑过一次(缓存截至9/16), 9/19(周六)再跑就直接返回这份旧数据,
少9/17、9/18两根K线 → MA/量比/250日分位全部失真(实测159529 MA20 1.2817 vs 真实1.2771)。

修复: 周末分支增加"缓存是否已含最近一个交易日"校验; 并把增量/全量两条路径里
"今天该有bar"的判据(datetime.now().weekday()<5 + last_bar<today)统一换成
"最近交易日"(last_bar < _latest_trading_day(today)), 使周末同样能触发备用源。

运行: pytest tests/test_kline_cache_weekend.py -v
"""
import sys, os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import data_engine
from data_engine import _fetch_kline_with_cache, _latest_trading_day


# ================================================================
# 测试脚手架
# ================================================================
def _bars(*dates, base=1.0):
    """构造K线序列(只需date/close参与断言, 其余字段给足解析所需)"""
    return [{"date": d, "open": base, "close": base, "high": base,
             "low": base, "volume": 1000, "amount": 1000} for d in dates]


def _em_payload(*dates):
    """东财增量接口返回格式: {"data": {"klines": ["日期,开,收,高,低,量,额", ...]}}"""
    return {"data": {"klines": [f"{d},1.0,1.0,1.0,1.0,1000,1000" for d in dates]}}


def _freeze(monkeypatch, iso_dt):
    """冻结 data_engine 模块内的时间(now()/get_today()/weekday() 同步生效)"""
    real = datetime

    class Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return iso_dt

    monkeypatch.setattr(data_engine, "datetime", Frozen)


@pytest.fixture
def env(monkeypatch):
    """装配可控环境: 缓存可注入、网络请求计数、写缓存变空操作"""
    state = {"cached": [], "json_calls": [], "json_returns": None,
             "fallback_calls": 0, "saved": None}

    monkeypatch.setattr(data_engine, "_load_kline_cache",
                        lambda code, ver="": (state["cached"], state["cached"][-1]["date"] if state["cached"] else ""))
    monkeypatch.setattr(data_engine, "_save_kline_cache",
                        lambda code, klines, ver="": state.__setitem__("saved", klines))

    def fake_fetch_json(url, timeout=8, retries=1):
        state["json_calls"].append(url)
        return state["json_returns"]

    monkeypatch.setattr(data_engine, "fetch_json", fake_fetch_json)

    def fallback_fn(code, days):
        state["fallback_calls"] += 1
        return state["fallback_bars"] if "fallback_bars" in state else None

    state["fallback_fn"] = fallback_fn
    return state


URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.159529&lmt=5"

# 2026-09-19 是周六, 最近交易日 = 2026-09-18(周五)
SATURDAY = datetime(2026, 9, 19, 1, 16)


# ================================================================
# _latest_trading_day
# ================================================================
class TestLatestTradingDay:
    def test_weekday_returns_itself(self):
        assert _latest_trading_day("20260918") == "20260918"   # 周五
        assert _latest_trading_day("20260921") == "20260921"   # 周一

    def test_weekend_rolls_back_to_friday(self):
        assert _latest_trading_day("20260919") == "20260918"   # 周六→周五
        assert _latest_trading_day("20260920") == "20260918"   # 周日→周五

    def test_holiday_is_not_known(self):
        # 只按周一~周五推算, 不含节假日历: 国庆周中会返回节内工作日(已知局限, 见函数docstring)
        assert _latest_trading_day("20261001") == "20261001"


# ================================================================
# 周末缓存校验 (本次修复核心)
# ================================================================
class TestWeekendStaleCache:
    def test_stale_cache_is_refreshed(self, env, monkeypatch):
        """缓存停在周四(9/16), 周六跑必须增量拉取补回9/17、9/18"""
        _freeze(monkeypatch, SATURDAY)
        env["cached"] = _bars("2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16")
        env["json_returns"] = _em_payload("2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 5, env["fallback_fn"], ver="_test")

        assert out[-1]["date"] == "2026-09-18", "周末不应返回停在9/16的旧缓存"
        assert len(env["json_calls"]) == 1, "应发起一次增量请求"

    def test_fresh_cache_skips_network(self, env, monkeypatch):
        """缓存已含周五bar → 零请求(保留原周末捷径的省流量意图)"""
        _freeze(monkeypatch, SATURDAY)
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")
        env["json_returns"] = None  # 一旦发请求就会污染结果

        out = _fetch_kline_with_cache("159529", URL, 4, env["fallback_fn"], ver="_test")

        assert out[-1]["date"] == "2026-09-18"
        assert env["json_calls"] == [], "缓存完整时不应发起任何请求"

    def test_stale_cache_and_em_down_uses_fallback(self, env, monkeypatch):
        """缓存滞后 + 东财不可达 → 备用源补拉(8/19事故的防护在周末同样生效)"""
        _freeze(monkeypatch, SATURDAY)
        env["cached"] = _bars("2026-09-14", "2026-09-15", "2026-09-16")
        env["json_returns"] = None
        env["fallback_bars"] = _bars("2026-09-16", "2026-09-17", "2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 3, env["fallback_fn"], ver="_test")

        assert env["fallback_calls"] == 1, "周末也应调用备用源"
        assert out[-1]["date"] == "2026-09-18"

    def test_weekend_increment_missing_friday_triggers_fallback(self, env, monkeypatch):
        """东财增量'成功但缺周五bar' → 备用源补(原逻辑因weekday()<5恒False而漏掉)"""
        _freeze(monkeypatch, SATURDAY)
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17")
        env["json_returns"] = _em_payload("2026-09-17")          # 增量只到周四
        env["fallback_bars"] = _bars("2026-09-17", "2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 3, env["fallback_fn"], ver="_test")

        assert env["fallback_calls"] == 1
        assert out[-1]["date"] == "2026-09-18"


# ================================================================
# 工作日行为不变 (回归保护: 8/17、8/19 两次修复不能被改坏)
# ================================================================
class TestWeekdayBehaviorUnchanged:
    def test_same_day_cache_hits_zero_requests(self, env, monkeypatch):
        _freeze(monkeypatch, datetime(2026, 9, 18, 18, 4))
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")
        env["json_returns"] = _em_payload("2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 4, env["fallback_fn"], ver="_test")

        assert out[-1]["date"] == "2026-09-18"
        assert env["json_calls"] == []

    def test_weekday_increment_missing_today_uses_fallback(self, env, monkeypatch):
        """8/17场景: 工作日增量响应成功但缺当日bar → 备用源补当日"""
        _freeze(monkeypatch, datetime(2026, 9, 18, 18, 4))
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17")
        env["json_returns"] = _em_payload("2026-09-16", "2026-09-17")   # 缺9/18
        env["fallback_bars"] = _bars("2026-09-17", "2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 3, env["fallback_fn"], ver="_test")

        assert env["fallback_calls"] == 1
        assert out[-1]["date"] == "2026-09-18"

    def test_weekday_em_unreachable_uses_fallback(self, env, monkeypatch):
        """8/19场景: 东财整体不可达 → 备用源(原逻辑已修, 此处锁死)"""
        _freeze(monkeypatch, datetime(2026, 9, 18, 18, 4))
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17")
        env["json_returns"] = None
        env["fallback_bars"] = _bars("2026-09-16", "2026-09-17", "2026-09-18")

        out = _fetch_kline_with_cache("159529", URL, 3, env["fallback_fn"], ver="_test")

        assert env["fallback_calls"] == 1
        assert out[-1]["date"] == "2026-09-18"

    def test_weekday_both_sources_down_degrades_to_cache(self, env, monkeypatch):
        """两路都挂 → 降级用缓存(不抛异常, 由调用方标stale)"""
        _freeze(monkeypatch, datetime(2026, 9, 18, 18, 4))
        env["cached"] = _bars("2026-09-15", "2026-09-16", "2026-09-17")
        env["json_returns"] = None
        env["fallback_bars"] = None

        out = _fetch_kline_with_cache("159529", URL, 3, env["fallback_fn"], ver="_test")

        assert out[-1]["date"] == "2026-09-17"
