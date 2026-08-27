"""
实时行情价格尺度修复回归测试 — 东财f43单位漂移(8/27)

背景(2026-08-27): 下午时段东财push2对部分ETF返回异常单位的f43:
  159920 恒生ETF → 现价1501元(真值1.5, 偏差1000倍)
  159750 港股科技50 → 现价8.71元(真值0.871, 偏差10倍)
根因1: _pick_scale 误差公式中尺度s可约分 → 投票对尺度不敏感(数学上必然),
       浮点噪声会在区间内随机选尺度(100 vs 1000), 无法靠投票本身区分;
       且旧版"任何尺度都不匹配就兜底1000"把垃圾值当正常价。
根因2: 10倍级漂移(8.71)落在0.1~500区间内, 区间兜底也拦不住 → 需K线交叉验证。

修复: 1) 价格区间约束移入投票循环, 全不匹配返回None → 转新浪备用
     2) K线交叉验证承担全部尺度判别: 当日K线偏差>10% 或 比值是"干净的
        10次幂"(10/100/0.1, 误差<5%, 即使K线非当日) → 以K线为准重建价格字段。
        注: 单日真实波动绝不可能是精确10倍, 干净10次幂只能是单位漂移。

运行: pytest tests/test_realtime_scale.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import data_engine


def _mk_em_resp(f43, f60, f170, f44=None, f45=None, f46=None, f58="测试ETF", f57="159529"):
    """构造东财push2响应(f43现价/f60昨收/f170涨跌基点, 其余字段可缺省)"""
    return {"data": {
        "f43": str(f43), "f44": str(f44) if f44 else "0", "f45": str(f45) if f45 else "0",
        "f46": str(f46) if f46 else "0", "f47": "1000", "f48": "1500",
        "f50": "", "f57": f57, "f58": f58, "f60": str(f60), "f169": "0",
        "f170": str(f170), "f171": "0", "f2": "0", "f3": "0", "f4": "0",
        "f15": "0", "f16": "0", "f17": "0", "f18": "0",
    }}


def _mk_kline_today(prev=1.508, last=1.5):
    """构造当日K线(昨日+今日), 收盘=last"""
    today = data_engine.get_today()
    return [
        {"date": "2026-08-26", "close": prev, "volume": 1000.0},
        {"date": today, "close": last, "volume": 1000.0},
    ]


class TestPickScaleGarbageToSina:
    """尺度无法解释(159920型: f43=1501000) → 弃用东财, 转新浪备用"""

    def test_1000x_garbage_goes_to_sina(self, monkeypatch):
        """f43=1501000对任何尺度都不在0.1~500区间 → 返回None → 新浪"""
        sentinel = {"price": 1.5, "source": "sina"}
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(1501000, 1501, -60))
        monkeypatch.setattr(data_engine, "_fetch_sina_realtime", lambda code: sentinel)
        r = data_engine.fetch_etf_realtime("159920")
        assert r is sentinel
        assert r["price"] == 1.5

    def test_sina_fallback_itself_fails_returns_none(self, monkeypatch):
        """新浪也失败 → None(调用方按无实时数据处理, 不返回垃圾)"""
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(1501000, 1501, -60))
        monkeypatch.setattr(data_engine, "_fetch_sina_realtime", lambda code: None)
        assert data_engine.fetch_etf_realtime("159920") is None


class TestPickScaleNormal:
    """正常尺度不被误伤"""

    def test_li_unit(self, monkeypatch):
        """正常厘单位: f43=1501(1.501元) → 价格正确, 无论投票落在1000还是100
        (误差公式尺度不变, 浮点噪声随机选; 选100→15.01→干净10次幂→锚定修正)"""
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(1501, 1500, 60))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: _mk_kline_today(prev=1.500, last=1.501))
        r = data_engine.fetch_etf_realtime("159529")
        assert r["price"] == 1.501
        # 未锚定 → change_pct=0.6(东财f170); 锚定 → 0.07(K线推导), 两条路径都对
        assert r["change_pct"] in (0.6, 0.07)

    def test_yuan_unit(self, monkeypatch):
        """元单位: f43=1.5 → 尺度1, 不触发锚定"""
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(1.5, 1.508, -53))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: _mk_kline_today(prev=1.508, last=1.5))
        r = data_engine.fetch_etf_realtime("159920")
        assert r["price"] == 1.5
        assert not r.get("scale_fixed")


class TestKlineAnchor:
    """K线交叉验证(159750型: f43=8710落在区间内但真值0.871)"""

    def test_10x_drift_anchored_to_kline(self, monkeypatch):
        """实时8.71 vs K线0.871 偏差900% → 以K线重建: price/prev/change全部修正"""
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(
            8710, 8740, -34, f44=8730, f45=8700, f46=8720, f58="港股科技50ETF招商", f57="159750"))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: _mk_kline_today(prev=0.874, last=0.871))
        r = data_engine.fetch_etf_realtime("159750")
        assert r["price"] == 0.871
        assert r["prev_close"] == 0.874
        assert abs(r["change_pct"] - (-0.34)) < 0.02
        assert abs(r["high"] - 0.873) < 0.001
        assert r["scale_fixed"] is True

    def test_10x_drift_with_stale_kline_still_anchored(self, monkeypatch):
        """K线非当日(周末/缓存滞后)但比值是干净10倍 → 仍锚定: 真实波动绝不可能是精确10倍"""
        stale = [
            {"date": "2026-08-25", "close": 0.873, "volume": 1000.0},
            {"date": "2026-08-26", "close": 0.874, "volume": 1000.0},
        ]
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(8710, 8740, -34))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: stale)
        r = data_engine.fetch_etf_realtime("159750")
        assert r["price"] == 0.874  # 以stale K线最后一根收盘重建(比10倍垃圾好; 日报另有K线滞后告警)
        assert r["scale_fixed"] is True

    def test_stale_kline_weird_ratio_not_anchored(self, monkeypatch):
        """残留风险: K线非当日且比值非干净10次幂 → 锚定无法触发, 区间内垃圾(8.71或87.1,
        由投票浮点噪声决定)原样返回, scale_fixed不置位。
        实际概率极低: daily_runner 18:04运行时当日K线几乎总是可得的, 且日报另有K线滞后告警。"""
        stale = [
            {"date": "2026-08-25", "close": 2.01, "volume": 1000.0},
            {"date": "2026-08-26", "close": 2.0, "volume": 1000.0},
        ]
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(8710, 8740, -34))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: stale)
        r = data_engine.fetch_etf_realtime("159750")
        assert r["price"] in (8.71, 87.1)  # 尺度1000或100, 均为区间内候选
        assert not r.get("scale_fixed")

    def test_drift_within_10pct_not_anchored(self, monkeypatch):
        """正常盘中波动(如2%) → 不触发锚定"""
        monkeypatch.setattr(data_engine, "fetch_json", lambda url, **kw: _mk_em_resp(1520, 1500, 133))
        monkeypatch.setattr(data_engine, "fetch_etf_kline", lambda code, days=250: _mk_kline_today(prev=1.500, last=1.52))
        r = data_engine.fetch_etf_realtime("159920")
        assert r["price"] == 1.52
        assert not r.get("scale_fixed")
