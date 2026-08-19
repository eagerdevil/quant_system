"""
溢价计算修复回归测试 — IOPV当日口径优先于官方净值T+1滞后口径

背景(2026-08-19): QDII官方净值T+1公布(实测滞后1-2天, 盘中最新净值只到8/17)，
旧逻辑 compute_etf_premium 官方净值永远优先 + compute_etf_premium_history 的
current 取 series 末位(官方口径)，把几天前的溢价当今天报：
  159529 官方口径 +3.24%(8/17) vs 腾讯IOPV实时 +1.19%(支付宝同源) — 相差2个百分点。

修复: fetch_etf_iopv 腾讯字段77/78；compute_etf_premium QDII时IOPV优先；
compute_etf_premium_history.current 用当日IOPV并标记 current_source。

运行: pytest tests/test_premium_iopv.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import pytest

import data_engine


def _mk_kline(days=60, last_close=1.315):
    """构造最近days个交易日K线（真实日历，结束于2026-08-19），最后一天收盘=last_close
    收盘价从 last_close-(days-1)*0.001 匀速涨到 last_close"""
    klines = []
    dates = []
    d = datetime(2026, 8, 19)
    while len(dates) < days:  # 从8/19往回数days个工作日
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()
    price = last_close - (days - 1) * 0.001
    for dd in dates:
        klines.append({"date": dd.strftime("%Y%m%d"), "close": round(price, 4)})
        price += 0.001
    return klines


def _mk_nav_hist(code=None, days=80):
    """构造历史官方净值（恒定1.30，使溢价≈+1%）——最新只到2026-08-17，
    模拟QDII官方净值T+1滞后2个交易日（8/19盘中时官方净值只到8/17）"""
    nav = {}
    d = datetime(2026, 5, 1)
    end = datetime(2026, 8, 17)
    while d <= end:
        if d.weekday() < 5:
            nav[d.strftime("%Y-%m-%d")] = 1.30
        d += timedelta(days=1)
    return nav


class TestComputeEtfPremiumPriority:
    """compute_etf_premium 优先级: IOPV > 官方净值 > 东财IOPV"""

    def test_qdii_iopv_wins_even_with_official_nav(self, monkeypatch):
        """QDII有实时IOPV时优先用IOPV，即使官方净值可用（旧逻辑会错用滞后官方口径）"""
        monkeypatch.setattr(data_engine, "fetch_etf_iopv",
                            lambda code: {"price": 1.292, "iopv": 1.2768, "premium_pct": 1.19})
        monkeypatch.setattr(data_engine, "fetch_etf_realtime", lambda code: None)
        r = data_engine.compute_etf_premium(
            "159529", 1.292, nav_data={"nav": 1.2737, "date": "2026-08-17"})
        assert r["data_source"] == "iopv"
        assert r["premium_pct"] == 1.19
        assert r["nav"] == 1.2768

    def test_qdii_falls_back_to_official_nav(self, monkeypatch):
        """IOPV不可用时退回官方净值（滞后口径兜底）"""
        monkeypatch.setattr(data_engine, "fetch_etf_iopv", lambda code: None)
        monkeypatch.setattr(data_engine, "fetch_etf_realtime", lambda code: None)
        r = data_engine.compute_etf_premium(
            "159529", 1.315, nav_data={"nav": 1.2737, "date": "2026-08-17"})
        assert r["data_source"] == "official_nav"
        assert abs(r["premium_pct"] - 3.24) < 0.05

    def test_qdii_unavailable_when_all_sources_dead(self, monkeypatch):
        """QDII所有源都失败 → unavailable + premium_pct=None（不误报0%）"""
        monkeypatch.setattr(data_engine, "fetch_etf_iopv", lambda code: None)
        monkeypatch.setattr(data_engine, "fetch_etf_realtime", lambda code: None)
        r = data_engine.compute_etf_premium("159529", 1.30, nav_data=None)
        assert r["data_source"] == "unavailable"
        assert r["premium_pct"] is None

    def test_non_qdii_skips_iopv(self, monkeypatch):
        """非QDII不调IOPV（A股ETF申赎套利溢价≈0，走原逻辑）"""
        called = {"iopv": 0}
        monkeypatch.setattr(data_engine, "fetch_etf_iopv",
                            lambda code: called.__setitem__("iopv", called["iopv"] + 1) or None)
        r = data_engine.compute_etf_premium(
            "510300", 4.70, nav_data={"nav": 4.70, "date": "2026-08-18"})
        assert called["iopv"] == 0
        assert r["data_source"] == "official_nav"
        assert abs(r["premium_pct"]) < 0.01


class TestComputeEtfPremiumHistoryCurrent:
    """current 必须用当日IOPV口径，而不是滞后的官方序列末位"""

    def test_current_uses_iopv(self, monkeypatch):
        """IOPV可用: current=实时口径, current_source=iopv, 历史统计仍用官方序列"""
        monkeypatch.setattr(data_engine, "fetch_etf_nav_history", _mk_nav_hist)
        monkeypatch.setattr(data_engine, "fetch_etf_iopv",
                            lambda code: {"price": 1.292, "iopv": 1.2768, "premium_pct": 1.19})
        klines = _mk_kline(last_close=1.315)  # 末位是8/17收盘1.315 → 旧逻辑会报+1.15%
        p = data_engine.compute_etf_premium_history("159529", klines)
        assert p["has_history"] is True
        assert p["current"] == 1.19          # IOPV口径
        assert p["current_source"] == "iopv"
        # 历史序列仍为官方口径，末位是8/17（K线1.313 vs 净值1.30 → +1.0%）
        assert abs(p["series"][-1]["premium"] - 1.0) < 0.05

    def test_current_falls_back_to_official(self, monkeypatch):
        """IOPV不可用: current=官方序列末位, current_source=official_nav（不破坏旧行为）"""
        monkeypatch.setattr(data_engine, "fetch_etf_nav_history", _mk_nav_hist)
        monkeypatch.setattr(data_engine, "fetch_etf_iopv", lambda code: None)
        klines = _mk_kline(last_close=1.315)
        p = data_engine.compute_etf_premium_history("159529", klines)
        assert p["current_source"] == "official_nav"
        assert abs(p["current"] - 1.0) < 0.05  # 8/17官方口径末位

    def test_insufficient_history(self, monkeypatch):
        """历史不足10个交易日 → has_history=False（与旧行为一致）"""
        monkeypatch.setattr(data_engine, "fetch_etf_nav_history", _mk_nav_hist)
        monkeypatch.setattr(data_engine, "fetch_etf_iopv",
                            lambda code: {"price": 1.292, "iopv": 1.2768, "premium_pct": 1.19})
        p = data_engine.compute_etf_premium_history("159529", _mk_kline(days=5))
        assert p["has_history"] is False


def _fake_tencent_raw(price=1.292, premium=1.19, iopv=1.2768):
    """按腾讯真实返回构造80字段行情串（字段3=现价, 77=溢价%, 78=IOPV）"""
    fields = ["0"] * 80
    fields[0] = "51"
    fields[1] = "标普消费ETF景顺"
    fields[2] = "159529"
    fields[3] = str(price)
    fields[77] = str(premium)
    fields[78] = str(iopv)
    return f'v_sz159529="{chr(126).join(fields)}";\n'


class _FakeResp:
    """支持 with 上下文管理的伪HTTP响应（fetch_etf_iopv 用 with urlopen()）"""

    def __init__(self, body_bytes):
        self._body = body_bytes

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFetchEtfIopvParsing:
    """腾讯行情字段77/78解析"""

    def test_parses_valid_quote(self, monkeypatch):
        monkeypatch.setattr(data_engine.urllib.request, "urlopen",
                            lambda req, timeout=10: _FakeResp(_fake_tencent_raw().encode("gbk")))
        r = data_engine.fetch_etf_iopv("159529")
        assert r is not None
        assert r["price"] == 1.292
        assert r["iopv"] == 1.2768
        assert r["premium_pct"] == 1.19

    def test_rejects_absurd_iopv(self, monkeypatch):
        """IOPV偏离现价>50% → 拒绝（防止假溢价进决策链）"""
        monkeypatch.setattr(data_engine.urllib.request, "urlopen",
                            lambda req, timeout=10: _FakeResp(
                                _fake_tencent_raw(price=1.292, iopv=0.50).encode("gbk")))
        assert data_engine.fetch_etf_iopv("159529") is None
