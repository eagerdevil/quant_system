#!/usr/bin/env python
"""
量化系统 核心引擎
=================
模块2: 数据预处理（去极值+标准化+中性化）
模块3: 多因子模型（动量/波动/基本面/资金/情绪）
模块4: 大盘择时与仓位管理
模块5: 交易决策生成
"""
import json, math, sys
from datetime import datetime

TODAY = datetime.now().strftime("%Y%m%d")

# ============================================================
# 模块2: 数据预处理
# ============================================================
def mad_outlier_filter(series, n=5.0):
    """MAD法去极值：5倍绝对中位差"""
    if len(series) < 3: return series
    sorted_s = sorted(series)
    median = sorted_s[len(sorted_s)//2]
    abs_dev = sorted(abs(x - median) for x in series)
    mad = abs_dev[len(abs_dev)//2] * 1.4826  # 一致估计量
    if mad == 0: return series
    upper = median + n * mad
    lower = median - n * mad
    return [min(max(x, lower), upper) for x in series]

def zscore_normalize(series):
    """Z-score标准化"""
    if len(series) < 2: return [0]*len(series)
    mean = sum(series)/len(series)
    std = math.sqrt(sum((x-mean)**2 for x in series)/(len(series)-1))
    if std == 0: return [0]*len(series)
    return [(x-mean)/std for x in series]

# ============================================================
# 模块3: 多因子计算
# ============================================================
def sma(data, n):
    if len(data) < n: return data[-1] if data else 0
    return sum(data[-n:])/n

def ema(data, n):
    if len(data) < n: return data[-1] if data else 0
    alpha = 2/(n+1)
    val = data[0]
    for v in data[1:]: val = alpha*v + (1-alpha)*val
    return val

def rsi(closes, n=14):
    if len(closes) < n+1: return 50
    gains, losses = [], []
    for i in range(-n, 0):
        d = closes[i] - closes[i-1]
        gains.append(d if d>0 else 0)
        losses.append(-d if d<0 else 0)
    avg_gain = sum(gains)/n
    avg_loss = sum(losses)/n
    if avg_loss == 0: return 100
    return 100 - 100/(1 + avg_gain/avg_loss)

def macd_calc(closes):
    """返回 (DIF, DEA, Hist) 当前值"""
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    dif = e12 - e26
    # 简化DEA
    if len(closes) >= 35:
        difs = []
        for i in range(26, len(closes)):
            e12_i = ema(closes[:i+1], 12)
            e26_i = ema(closes[:i+1], 26)
            difs.append(e12_i - e26_i)
        dea = ema(difs, 9) if difs else dif
    else:
        dea = dif
    return dif, dea, (dif-dea)*2

def max_drawdown(closes):
    peak, md = closes[0], 0
    for c in closes:
        if c > peak: peak = c
        dd = (peak-c)/peak
        if dd > md: md = dd
    return md

def volatility(closes, n=20):
    if len(closes) < n+1: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(-n, 0)]
    avg = sum(rets)/n
    return math.sqrt(sum((r-avg)**2 for r in rets)/n) * math.sqrt(252)

def sharpe(closes, rf=0.025):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    std_ret = math.sqrt(sum((r-avg_ret)**2 for r in rets)/(len(rets)-1)) if len(rets)>1 else 0.01
    if std_ret == 0: return 0
    return (avg_ret - rf/252)/std_ret * math.sqrt(252)

def sortino(closes, rf=0.025):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    down_rets = [r for r in rets if r < 0]
    if len(down_rets) < 2: return 3.0 if avg_ret>0 else -3.0
    down_std = math.sqrt(sum(r**2 for r in down_rets)/(len(down_rets)-1))
    if down_std == 0: return 0
    return (avg_ret - rf/252)/down_std * math.sqrt(252)

def consecutive_up(closes):
    cnt = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] > closes[i-1]: cnt += 1
        else: break
    return cnt

def consecutive_down(closes):
    cnt = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] < closes[i-1]: cnt += 1
        else: break
    return cnt

def ret_n(closes, n):
    if len(closes) < n+1: return 0
    return (closes[-1]/closes[-n-1] - 1)*100

def vol_ratio(volumes, n=5):
    if len(volumes) < n*2: return 1.0
    recent = sum(volumes[-n:])/n
    prior = sum(volumes[-n*2:-n])/n
    return recent/prior if prior>0 else 1

def pma(closes, n):
    ma = sma(closes, n)
    if ma == 0: return 0
    return (closes[-1]/ma - 1)*100

def ma_alignment(closes):
    """MA多头排列程度：MA5>MA10>MA20>MA60"""
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60) if len(closes)>=60 else sma(closes, len(closes))
    score = 0
    if ma5 > ma10: score += 1
    if ma10 > ma20: score += 1
    if ma20 > ma60: score += 1
    return score

def bollinger_position(closes, n=20):
    """布林带位置 0-100"""
    if len(closes) < n: return 50
    window = closes[-n:]
    avg = sum(window)/n
    std = math.sqrt(sum((x-avg)**2 for x in window)/n)
    if std == 0: return 50
    upper, lower = avg + 2*std, avg - 2*std
    if upper == lower: return 50
    return (closes[-1] - lower)/(upper - lower)*100

# ============================================================
# 综合因子评分（ETF级别）
# ============================================================
def score_etf_comprehensive(code, name, closes, highs, lows, volumes,
                             north_flow_5d=None, industry_return=None):
    """
    15因子综合评分系统
    返回: {score, grade, factor_details, indicators}
    """
    cur = closes[-1]
    n = len(closes)

    # ---- 计算所有指标 ----
    rsi_now = rsi(closes, 14)
    dif, dea, hist = macd_calc(closes)
    vol = volatility(closes, 20)
    maxdd = max_drawdown(closes)
    shp = sharpe(closes)
    srt = sortino(closes)
    cons_up = consecutive_up(closes)
    cons_down = consecutive_down(closes)
    ma_align = ma_alignment(closes)
    bb_pos = bollinger_position(closes, 20)
    atr_val = sum(abs(closes[i]-closes[i-1]) for i in range(-14, 0))/14 if len(closes)>=15 else 0
    atr_pct = atr_val/cur*100 if cur>0 else 0

    r5 = ret_n(closes, 5)
    r10 = ret_n(closes, 10)
    r20 = ret_n(closes, 20)
    r60 = ret_n(closes, 60)
    r120 = ret_n(closes, 120) if len(closes)>=121 else 0

    v_ratio = vol_ratio(volumes, 5)
    v_ratio_10 = vol_ratio(volumes, 10)

    pm5 = pma(closes, 5)
    pm10 = pma(closes, 10)
    pm20 = pma(closes, 20)
    pm60 = pma(closes, 60) if len(closes)>=60 else pm20

    # ---- 15因子评分 ----
    factors = {}

    # F1: 趋势强度(0-10)
    f1 = 5
    if dif > 0 and hist > 0: f1 = 10
    elif dif > 0: f1 = 7
    elif dif > -0.01 and hist > 0: f1 = 6
    elif dif > -0.01: f1 = 4
    else: f1 = 2
    if ma_align >= 2: f1 = min(10, f1+1)
    factors['F1_趋势强度'] = f1

    # F2: 动量(0-8)
    f2 = 4
    if 0 <= r20 <= 10: f2 = 8
    elif -3 <= r20 < 0: f2 = 6
    elif 10 < r20 <= 20: f2 = 5
    elif r20 < -10: f2 = 3
    else: f2 = 2
    factors['F2_动量'] = f2

    # F3: 反转信号(0-8)
    f3 = 4
    if r5 < -5: f3 = 8
    elif -5 <= r5 < -2: f3 = 6
    elif r5 < 0: f3 = 5
    elif 0 <= r5 <= 3: f3 = 4
    elif 3 < r5 <= 8: f3 = 3
    else: f3 = 1
    factors['F3_反转'] = f3

    # F4: RSI位置(0-8)
    f4 = 4
    if 40 <= rsi_now <= 50: f4 = 8
    elif 35 <= rsi_now < 40: f4 = 7
    elif 50 < rsi_now <= 58: f4 = 6
    elif 30 <= rsi_now < 35: f4 = 5
    elif 58 < rsi_now <= 65: f4 = 4
    elif 65 < rsi_now <= 72: f4 = 2
    else: f4 = 1
    factors['F4_RSI'] = f4

    # F5: 均线偏离(0-6)
    f5 = 3
    if -3 <= pm5 <= 3: f5 = 6
    elif -5 <= pm5 < -3: f5 = 5
    elif 3 < pm5 <= 8: f5 = 4
    elif -8 <= pm5 < -5: f5 = 3
    else: f5 = 2
    factors['F5_均线偏离'] = f5

    # F6: 低波动(0-6)
    f6 = 3
    if 15 <= vol*100 <= 30: f6 = 6
    elif 10 <= vol*100 < 15: f6 = 5
    elif 30 < vol*100 <= 40: f6 = 4
    elif vol*100 < 10: f6 = 3
    else: f6 = 2
    factors['F6_低波动'] = f6

    # F7: 成交量健康(0-6)
    f7 = 3
    if 0.85 <= v_ratio <= 1.20: f7 = 6
    elif 0.70 <= v_ratio <= 1.40: f7 = 4
    else: f7 = 2
    factors['F7_成交量'] = f7

    # F8: 回调质量(0-10)
    f8 = 5
    if cons_down >= 3: f8 = 10
    elif cons_down >= 2: f8 = 8
    elif cons_down == 1: f8 = 7
    elif cons_up == 0: f8 = 7
    elif cons_up == 1: f8 = 6
    elif cons_up == 2: f8 = 4
    else: f8 = 2
    factors['F8_回调'] = f8

    # F9: Sortino(0-6)
    f9 = 3
    if srt > 1.5: f9 = 6
    elif srt > 0.8: f9 = 5
    elif srt > 0.3: f9 = 4
    elif srt > -0.3: f9 = 3
    else: f9 = 2
    factors['F9_Sortino'] = f9

    # F10: 最大回撤(0-6)
    f10 = 3
    if maxdd < 0.10: f10 = 6
    elif maxdd < 0.18: f10 = 5
    elif maxdd < 0.25: f10 = 4
    elif maxdd < 0.35: f10 = 3
    else: f10 = 2
    factors['F10_MaxDD'] = f10

    # F11: 布林带位置(0-6)
    f11 = 3
    if 15 <= bb_pos <= 55: f11 = 6
    elif 5 <= bb_pos < 15: f11 = 5
    elif 55 < bb_pos <= 75: f11 = 4
    elif 75 < bb_pos <= 90: f11 = 2
    else: f11 = 1
    factors['F11_布林带'] = f11

    # F12: 多周期收益(0-6)
    f12 = 3
    if r5 > 0 and r20 > 0 and r60 > 0: f12 = 6
    elif r20 > 0 and r60 > 0: f12 = 5
    elif r60 > 0: f12 = 4
    elif r20 > 0: f12 = 3
    else: f12 = 2
    factors['F12_多周期'] = f12

    # F13: 均线排列(0-4)
    f13 = ma_align
    factors['F13_均线排列'] = f13

    # F14: 长期收益(0-4)
    f14 = 2
    if r120 > 10: f14 = 4
    elif r120 > 0: f14 = 3
    elif r120 > -10: f14 = 2
    else: f14 = 1
    factors['F14_长期'] = f14

    # F15: 夏普比率(0-6)
    f15 = 3
    if shp > 1.5: f15 = 6
    elif shp > 0.8: f15 = 5
    elif shp > 0.3: f15 = 4
    elif shp > -0.3: f15 = 3
    else: f15 = 2
    factors['F15_夏普'] = f15

    total = sum(factors.values())
    max_score = 100

    # Grade
    if total >= 78: grade = "A_强烈买入"
    elif total >= 65: grade = "B_买入"
    elif total >= 55: grade = "C_观察"
    elif total >= 42: grade = "D_谨慎"
    else: grade = "E_回避"

    return {
        "code": code, "name": name,
        "score": total, "max_score": max_score,
        "grade": grade,
        "price": round(cur, 4),
        "indicators": {
            "rsi": round(rsi_now, 1), "volatility_pct": round(vol*100, 1),
            "sharpe": round(shp, 2), "sortino": round(srt, 2),
            "max_dd_pct": round(maxdd*100, 1),
            "consecutive_up": cons_up, "consecutive_down": cons_down,
            "ma_alignment": ma_align, "bb_position": round(bb_pos, 1),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio_5d": round(v_ratio, 2),
            "macd_dif": round(dif, 4), "macd_hist": round(hist, 4)
        },
        "returns": {"r5d": round(r5,1), "r10d": round(r10,1), "r20d": round(r20,1),
                     "r60d": round(r60,1), "r120d": round(r120,1)},
        "vs_ma": {"pct_ma5": round(pm5,1), "pct_ma10": round(pm10,1),
                   "pct_ma20": round(pm20,1), "pct_ma60": round(pm60,1)},
        "factors": factors
    }

# ============================================================
# 模块4: 大盘择时引擎
# ============================================================
class MarketTiming:
    """市场择时信号"""

    def __init__(self, index_data, north_flow, total_vol, breadth, margin):
        """
        index_data: {code: {name, data: [{date,close,volume}...]}}
        north_flow: [{date, net_flow}]
        total_vol: float (亿元)
        breadth: {limit_up, limit_down, up_count, down_count}
        margin: {balance, change}
        """
        self.hs300 = None
        if "000300" in index_data:
            self.hs300 = [d["close"] for d in index_data["000300"]["data"]]

        self.north_flow = north_flow or []
        self.total_vol = total_vol or 0
        self.breadth = breadth or {}
        self.margin = margin or {}

    def calc_signals(self):
        """计算6个择时信号，返回 {signal_name: bool}"""
        signals = {}

        # S1: 沪深300在20日均线上方
        if self.hs300 and len(self.hs300) >= 20:
            ma20 = sma(self.hs300, 20)
            signals['S1_HS300_above_MA20'] = self.hs300[-1] > ma20
        else:
            signals['S1_HS300_above_MA20'] = False

        # S2: 沪深300的60日均线向上
        if self.hs300 and len(self.hs300) >= 61:
            ma60_now = sma(self.hs300, 60)
            ma60_prev = sma(self.hs300[:-1], 60)
            signals['S2_HS300_MA60_up'] = ma60_now > ma60_prev
        else:
            signals['S2_HS300_MA60_up'] = False

        # S3: 北向资金近5日累计净流入为正
        nf_5d = sum(f.get("net_flow", 0) for f in self.north_flow[-5:]) if self.north_flow else 0
        signals['S3_NorthFlow_5d_positive'] = nf_5d > 0

        # S4: 全市场成交额大于20日均值
        # (简化：用3万亿作为A股活跃阈值)
        signals['S4_Volume_active'] = self.total_vol > 20000  # 2万亿

        # S5: 跌停家数小于20家（数据缺失时默认不通过）
        ld = self.breadth.get("limit_down")
        signals['S5_LimitDown_low'] = ld < 20 if ld is not None else False

        # S6: 融资余额5日斜率（用净买入方向判断）
        margin_change = self.margin.get("change", 0)
        signals['S6_Margin_increasing'] = margin_change > 0

        return signals, nf_5d

    def position_advice(self):
        """根据择时信号计算建议仓位"""
        signals, nf_5d = self.calc_signals()
        bull_count = sum(1 for v in signals.values() if v)

        # 强制限制：HS300低于MA60且MA60向下 → 仓位上限30%
        hs300_below_ma60 = not signals.get('S1_HS300_above_MA20', True)
        ma60_down = not signals.get('S2_HS300_MA60_up', True)
        force_cap = hs300_below_ma60 and ma60_down

        # 仓位映射
        if bull_count >= 5: base_position = 1.0
        elif bull_count >= 4: base_position = 0.85
        elif bull_count >= 3: base_position = 0.65
        elif bull_count >= 2: base_position = 0.40
        elif bull_count >= 1: base_position = 0.20
        else: base_position = 0.05

        if force_cap:
            base_position = min(base_position, 0.30)

        return {
            "bull_signals": bull_count,
            "total_signals": len(signals),
            "signal_detail": signals,
            "base_position": round(base_position, 2),
            "force_capped": force_cap,
            "north_flow_5d": round(nf_5d, 1),
            "advice": self._position_text(base_position, bull_count, force_cap)
        }

    def _position_text(self, pos, bulls, capped):
        if pos >= 0.85: return f"进攻(仓位{pos*100:.0f}%) — {bulls}个看多信号，市场强势"
        elif pos >= 0.60: return f"偏多(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.35: return f"中性(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.15: return f"偏防御(仓位{pos*100:.0f}%) — 仅{bulls}个看多信号"
        else: return f"防御(仓位{pos*100:.0f}%) — 几乎无看多信号" + ("，强制限制生效" if capped else "")

# ============================================================
# 模块5: 交易决策生成器
# ============================================================
class TradeDecider:
    """根据因子得分+择时+持仓生成操作计划"""

    def __init__(self, etf_scores, timing_result, portfolio=None):
        """
        etf_scores: [{code, name, score, grade, price, indicators, returns, vs_ma, factors}]
        timing_result: MarketTiming.position_advice() output
        portfolio: {code: {shares, cost, name}} 当前持仓
        """
        self.scores = sorted(etf_scores, key=lambda x: x["score"], reverse=True)
        self.timing = timing_result
        self.portfolio = portfolio or {}

    def generate_plan(self, total_capital=4000, max_single=0.25, max_industry=0.40):
        """
        生成明日操作计划
        total_capital: 总资金
        max_single: 单品种最大仓位
        max_industry: 单行业最大仓位
        """
        target_pos = self.timing["base_position"]
        target_amount = total_capital * target_pos
        current_invested = sum(
            p.get("shares", 0) * p.get("current_price", p.get("cost", 0))
            for p in self.portfolio.values()
        )

        # 排名前40%的ETF
        top_n = max(5, len(self.scores)//2)
        top_etfs = self.scores[:top_n]

        buy_list = []
        sell_list = []
        hold_list = []

        # 卖出判断
        for code, pos in self.portfolio.items():
            score_data = next((s for s in self.scores if s["code"] == code), None)
            if not score_data:
                continue

            current_price = score_data["price"]
            cost = pos.get("cost", current_price)
            pnl_pct = (current_price/cost - 1)*100 if cost > 0 else 0

            reasons = []
            # 条件1: 得分跌出前40%
            if score_data["score"] < (self.scores[len(self.scores)*4//10]["score"] if len(self.scores)>=10 else 50):
                reasons.append("得分排名下滑")
            # 条件2: 从最高点回撤-8%
            if pnl_pct <= -8:
                reasons.append(f"止损触发(浮亏{pnl_pct:.1f}%)")
            # 条件3: 大盘择时转为防御且仓位需要削减
            if target_pos < 0.3 and current_invested > target_amount:
                reasons.append("大盘防御模式，需减仓")

            if reasons:
                sell_list.append({
                    "code": code, "name": pos.get("name", code),
                    "action": "卖出", "shares": pos.get("shares", 0),
                    "current_price": current_price, "pnl_pct": round(pnl_pct, 1),
                    "reasons": reasons
                })
            else:
                hold_list.append({
                    "code": code, "name": pos.get("name", code),
                    "action": "持有", "shares": pos.get("shares", 0),
                    "current_price": current_price, "pnl_pct": round(pnl_pct, 1)
                })

        # 买入建议
        remaining = target_amount - current_invested
        if remaining > 0 and target_pos >= 0.2:
            for etf in top_etfs:
                if etf["code"] in self.portfolio: continue  # 已持有
                if etf["grade"] in ["D_谨慎", "E_回避"]: continue

                # 单只仓位上限
                allocation = min(remaining * 0.3, total_capital * max_single)
                shares = int(allocation / etf["price"])
                if shares < 100: continue

                amount = shares * etf["price"]

                # 生成买入理由
                reasons = []
                if etf["indicators"]["rsi"] <= 55:
                    reasons.append(f"RSI={etf['indicators']['rsi']} 中性偏低")
                if etf["returns"]["r5d"] <= 0:
                    reasons.append("短期回调充分")
                if etf["indicators"]["consecutive_up"] <= 1:
                    reasons.append("非追高(连涨≤1天)")
                if etf["factors"].get("F1_趋势强度", 0) >= 7:
                    reasons.append("趋势强")
                if not reasons:
                    reasons.append(f"综合得分{etf['score']}分，排名前{top_n}")

                buy_list.append({
                    "code": etf["code"], "name": etf["name"],
                    "action": "买入", "shares": shares,
                    "price": etf["price"], "amount": round(amount, 0),
                    "score": etf["score"], "grade": etf["grade"],
                    "reasons": reasons[:3]
                })
                remaining -= amount

                if len(buy_list) >= 5: break

        return {
            "target_position": round(target_pos*100),
            "target_amount": round(target_amount, 0),
            "current_invested": round(current_invested, 0),
            "remaining": round(remaining, 0),
            "buy_list": buy_list,
            "sell_list": sell_list,
            "hold_list": hold_list,
            "timing_advice": self.timing["advice"]
        }

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 简易测试
    print("Quant Engine v1.0 — Ready", file=sys.stderr)
    print("Usage: import quant_engine and call score_etf_comprehensive()")
