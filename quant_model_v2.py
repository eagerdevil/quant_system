#!/usr/bin/env python
"""
Multi-Factor Quantitative Scoring Model v2.0
- 250 trading days (~1 year) of historical data
- 10-factor scoring system
- Risk/Return decomposition
- Output: JSON for easy parsing
"""
import json, urllib.request, math, sys
from datetime import datetime

# ============ CONFIG ============
RISK_FREE = 0.025  # Risk-free rate (1-year China bond)
DAYS = 250  # ~1 year of trading data
END_DATE = "20260623"

# ============ INDICATOR FUNCTIONS ============

def sma(data, n):
    if len(data) < n: return [None]*len(data)
    return [None]*(n-1) + [sum(data[i-n+1:i+1])/n for i in range(n-1, len(data))]

def ema(data, n):
    if len(data) < 2: return [None]*len(data)
    out = [data[0]]
    alpha = 2/(n+1)
    for v in data[1:]:
        out.append(alpha*v + (1-alpha)*out[-1])
    return out

def rsi(closes, n=14):
    if len(closes) < n+1: return [None]*len(closes)
    out = [None]*n
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(d if d>0 else 0)
        losses.append(-d if d<0 else 0)
    avg_gain = sum(gains[:n])/n
    avg_loss = sum(losses[:n])/n
    for i in range(n, len(closes)):
        if avg_loss == 0: out.append(100)
        else:
            rs = avg_gain/avg_loss
            out.append(100 - 100/(1+rs))
        if i < len(closes)-1:
            avg_gain = (avg_gain*(n-1) + gains[i])/n
            avg_loss = (avg_loss*(n-1) + losses[i])/n
    return out

def macd_calc(closes):
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    dif = [e12-e26 if (e12 and e26) else 0 for e12,e26 in zip(ema12, ema26)]
    dea = ema(dif, 9)
    hist = [(d-dea[i])*2 for i,d in enumerate(dif)]
    return dif, dea, hist

def bollinger(closes, n=20):
    ma = sma(closes, n)
    upper, lower = [None]*len(closes), [None]*len(closes)
    bandwidth = [None]*len(closes)
    for i in range(n-1, len(closes)):
        window = closes[i-n+1:i+1]
        avg = ma[i]
        std = math.sqrt(sum((x-avg)**2 for x in window)/n)
        upper[i] = avg + 2*std
        lower[i] = avg - 2*std
        bandwidth[i] = (upper[i]-lower[i])/avg*100 if avg>0 else 0
    return ma, upper, lower, bandwidth

def atr(highs, lows, closes, n=14):
    if len(closes) < n+1: return [None]*len(closes)
    tr = [None]
    for i in range(1, len(closes)):
        h_l = highs[i] - lows[i]
        h_pc = abs(highs[i] - closes[i-1])
        l_pc = abs(lows[i] - closes[i-1])
        tr.append(max(h_l, h_pc, l_pc))
    out = [None]*n
    avg = sum(tr[1:n+1])/n
    out.append(avg)
    for i in range(n+1, len(closes)):
        avg = (avg*(n-1) + tr[i])/n
        out.append(avg)
    return out

def keltner(highs, lows, closes, n=20, mult=1.5):
    ma = ema(closes, n)
    atr_vals = atr(highs, lows, closes, n)
    upper = [ma[i]+mult*atr_vals[i] if (ma[i] and atr_vals[i]) else None for i in range(len(closes))]
    lower = [ma[i]-mult*atr_vals[i] if (ma[i] and atr_vals[i]) else None for i in range(len(closes))]
    return ma, upper, lower

def max_drawdown(closes):
    peak, md = closes[0], 0
    for c in closes[1:]:
        if c > peak: peak = c
        dd = (peak-c)/peak
        if dd > md: md = dd
    return md

def sharpe(closes, rf=RISK_FREE):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    std_ret = math.sqrt(sum((r-avg_ret)**2 for r in rets)/(len(rets)-1)) if len(rets)>1 else 0.01
    if std_ret == 0: return 0
    return (avg_ret - rf/252)/std_ret * math.sqrt(252)

def sortino(closes, rf=RISK_FREE):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    down_rets = [r for r in rets if r < 0]
    if len(down_rets) < 2: return 5.0 if avg_ret>0 else -5.0
    down_std = math.sqrt(sum((r)**2 for r in down_rets)/(len(down_rets)-1))
    if down_std == 0: return 0
    return (avg_ret - rf/252)/down_std * math.sqrt(252)

def volatility(closes):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg = sum(rets)/len(rets)
    return math.sqrt(sum((r-avg)**2 for r in rets)/(len(rets)-1)) * math.sqrt(252)

def beta(closes, benchmark_closes):
    """Calculate beta relative to benchmark (CSI 300 proxy)"""
    rets_a = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    rets_b = [(benchmark_closes[i]-benchmark_closes[i-1])/benchmark_closes[i-1] for i in range(1,len(benchmark_closes))]
    n = min(len(rets_a), len(rets_b))
    rets_a, rets_b = rets_a[-n:], rets_b[-n:]
    avg_a = sum(rets_a)/n; avg_b = sum(rets_b)/n
    cov = sum((rets_a[i]-avg_a)*(rets_b[i]-avg_b) for i in range(n))/n
    var_b = sum((r-avg_b)**2 for r in rets_b)/n
    return cov/var_b if var_b>0 else 1.0

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
    if ma[-1] is None: return 0
    return (closes[-1]/ma[-1] - 1)*100

def ma_trend(closes):
    """Check MA alignment: MA5>MA10>MA20>MA60 = bullish"""
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    if any(x[-1] is None for x in [ma5, ma10, ma20, ma60]): return 0
    bullish = 0
    if ma5[-1] > ma10[-1]: bullish += 1
    if ma10[-1] > ma20[-1]: bullish += 1
    if ma20[-1] > ma60[-1]: bullish += 1
    return bullish

def mom_score(closes, n=20):
    """Momentum: rate of change over n days, normalized"""
    if len(closes) < n+1: return 0
    roc = (closes[-1]/closes[-n-1] - 1)*100
    # Penalize extreme moves either direction
    if 0 <= roc <= 10: return 5
    elif -5 <= roc < 0: return 4
    elif 10 < roc <= 20: return 2
    elif -10 <= roc < -5: return 3
    else: return 1

# ============ FETCH DATA ============

def fetch_kline(code, days=DAYS):
    market = "1" if code.startswith(("5","1")) else "0"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end={END_DATE}&lmt={days}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("data", {}).get("klines", [])
    except:
        return []

def parse_klines(klines):
    opens, closes, highs, lows, volumes = [], [], [], [], []
    for k in klines:
        parts = k.split(",")
        opens.append(float(parts[1]))
        closes.append(float(parts[2]))
        highs.append(float(parts[3]))
        lows.append(float(parts[4]))
        volumes.append(float(parts[5]))
    return opens, closes, highs, lows, volumes

# ============ SCORING ENGINE ============

def score_etf(code, name, closes, highs, lows, volumes, bench_closes=None):
    """10-factor quantitative scoring model"""
    cur = closes[-1]
    n = len(closes)

    # Calculate all indicators
    rsi14 = rsi(closes, 14)
    dif, dea, hist = macd_calc(closes)
    bb_ma, bb_up, bb_low, bb_bw = bollinger(closes, 20)
    atr_vals = atr(highs, lows, closes, 14)
    kelt_ma, kelt_up, kelt_low = keltner(highs, lows, closes, 20)

    # Current values
    rsi_now = rsi14[-1] if rsi14[-1] else 50
    dif_now = dif[-1] if dif[-1] else 0
    hist_now = hist[-1] if hist[-1] else 0
    atr_now = atr_vals[-1] if atr_vals[-1] else 0
    atr_pct = (atr_now/cur*100) if cur>0 else 0

    bb_pos = 50
    if bb_up[-1] and bb_low[-1] and bb_up[-1]!=bb_low[-1]:
        bb_pos = (cur - bb_low[-1])/(bb_up[-1] - bb_low[-1])*100

    bb_width = bb_bw[-1] if bb_bw[-1] else 10

    # Risk metrics
    vol = volatility(closes)
    max_dd = max_drawdown(closes)
    shp = sharpe(closes)
    srt = sortino(closes)
    b = beta(closes, bench_closes) if bench_closes else 1.0

    # Trend metrics
    cons_up = consecutive_up(closes)
    cons_down = consecutive_down(closes)
    ma_align = ma_trend(closes)
    mom = mom_score(closes, 20)

    r5 = ret_n(closes, 5)
    r10 = ret_n(closes, 10)
    r20 = ret_n(closes, 20)
    r60 = ret_n(closes, 60)
    r120 = ret_n(closes, 120) if len(closes)>=121 else 0

    p5 = pma(closes, 5)
    p10 = pma(closes, 10)
    p20 = pma(closes, 20)
    p60 = pma(closes, 60)

    vr = vol_ratio(volumes, 5)
    vr10 = vol_ratio(volumes, 10)

    # ===== 10-FACTOR SCORING =====
    scores = {}

    # F1: Trend Strength (0-12)
    f1 = 0
    if dif_now > 0 and hist_now > 0: f1 = 10  # Strong bullish
    elif dif_now > 0 and hist_now > -0.005: f1 = 8  # Bullish, slight pullback
    elif dif_now > 0: f1 = 5  # Weakening bullish
    elif dif_now > -0.01 and hist_now > 0: f1 = 7  # Potential reversal
    elif dif_now > -0.01: f1 = 4  # Near bottom
    else: f1 = 2
    # MA alignment bonus
    if ma_align >= 2: f1 = min(12, f1+2)
    scores['F1_Trend'] = f1

    # F2: RSI Position (0-12)
    f2 = 0
    if 40 <= rsi_now <= 50: f2 = 12  # Sweet spot
    elif 35 <= rsi_now < 40: f2 = 10  # Near oversold
    elif 50 < rsi_now <= 58: f2 = 9  # Moderate
    elif 30 <= rsi_now < 35: f2 = 8  # Oversold approaching
    elif 58 < rsi_now <= 65: f2 = 6  # Getting warm
    elif 25 <= rsi_now < 30: f2 = 5  # Very oversold
    elif 65 < rsi_now <= 72: f2 = 4  # Overbought approaching
    elif rsi_now < 25: f2 = 3
    elif rsi_now > 72: f2 = 2  # Overbought
    scores['F2_RSI'] = f2

    # F3: Pullback Freshness (0-12)
    f3 = 0
    if cons_down >= 2: f3 = 12  # Multi-day pullback = fresh entry
    elif cons_down == 1: f3 = 10
    elif cons_up == 0: f3 = 10  # Fell today
    elif cons_up == 1: f3 = 8
    elif cons_up == 2: f3 = 6
    elif cons_up <= 4: f3 = 4
    else: f3 = 1  # Extended rally
    scores['F3_Pullback'] = f3

    # F4: Volatility Quality (0-10)
    f4 = 0
    if 15 <= vol*100 <= 32: f4 = 10  # Moderate volatility
    elif 32 < vol*100 <= 45: f4 = 7
    elif 10 <= vol*100 < 15: f4 = 6
    elif 45 < vol*100 <= 60: f4 = 4
    else: f4 = 2
    scores['F4_Volatility'] = f4

    # F5: Volume Health (0-10)
    f5 = 0
    if 0.85 <= vr <= 1.20: f5 = 10  # Healthy volume
    elif 0.70 <= vr <= 1.40: f5 = 7
    elif 0.50 <= vr < 0.70: f5 = 5  # Shrinking volume
    elif 1.40 < vr <= 2.0: f5 = 5  # Expanding volume
    else: f5 = 3
    scores['F5_Volume'] = f5

    # F6: Risk-Adjusted Return (0-10)
    f6 = 0
    if srt > 2.0: f6 = 10
    elif srt > 1.2: f6 = 8
    elif srt > 0.6: f6 = 7
    elif srt > 0.2: f6 = 5
    elif srt > -0.5: f6 = 4
    else: f6 = 2
    scores['F6_Sortino'] = f6

    # F7: Drawdown Recovery (0-10)
    f7 = 0
    if max_dd < 0.08: f7 = 10  # Very low drawdown
    elif max_dd < 0.13: f7 = 8
    elif max_dd < 0.20: f7 = 6
    elif max_dd < 0.30: f7 = 4
    else: f7 = 2
    scores['F7_MaxDD'] = f7

    # F8: Multi-Timeframe Return (0-10)
    f8 = 0
    # Rewards: short-term flat/slightly positive, medium-term positive, long-term strong
    if -2 <= r5 <= 4: f8 += 3
    elif 4 < r5 <= 8: f8 += 2
    if 0 <= r20 <= 10: f8 += 4
    elif -5 <= r20 < 0: f8 += 3
    if r60 > 5: f8 += 3
    elif r60 > 0: f8 += 2
    scores['F8_Returns'] = f8

    # F9: Bollinger/Keltner Position (0-8)
    f9 = 0
    if 20 <= bb_pos <= 55: f9 = 8  # Lower half to mid, room to run
    elif 55 < bb_pos <= 70: f9 = 6
    elif 5 <= bb_pos < 20: f9 = 5  # Near lower band
    elif 70 < bb_pos <= 85: f9 = 4
    elif bb_pos < 5: f9 = 3
    else: f9 = 2  # Above upper band
    scores['F9_BB'] = f9

    # F10: Beta/Correlation Context (0-6)
    f10 = 0
    if 0.6 <= b <= 1.3: f10 = 6  # Moderate beta
    elif 0.3 <= b < 0.6: f10 = 4  # Low beta defensive
    elif 1.3 < b <= 1.8: f10 = 3  # Higher beta
    else: f10 = 2
    scores['F10_Beta'] = f10

    total = sum(scores.values())

    # Grade
    if total >= 80: grade = "STRONG_BUY"
    elif total >= 65: grade = "BUY"
    elif total >= 52: grade = "WATCH"
    elif total >= 40: grade = "HOLD_OFF"
    else: grade = "AVOID"

    return {
        "code": code,
        "name": name,
        "score": total,
        "grade": grade,
        "price": round(cur, 4),
        "indicators": {
            "rsi": round(rsi_now, 1),
            "consecutive_up": cons_up,
            "consecutive_down": cons_down,
            "volatility_pct": round(vol*100, 1),
            "sharpe": round(shp, 2),
            "sortino": round(srt, 2),
            "max_dd_pct": round(max_dd*100, 1),
            "beta": round(b, 2),
            "ma_alignment": ma_align,
            "bb_position_pct": round(bb_pos, 1),
            "bb_width_pct": round(bb_width, 1),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio_5d": round(vr, 2),
            "vol_ratio_10d": round(vr10, 2),
            "macd_dif": round(dif_now, 4),
            "macd_hist": round(hist_now, 4)
        },
        "returns": {
            "r5d": round(r5, 1),
            "r10d": round(r10, 1),
            "r20d": round(r20, 1),
            "r60d": round(r60, 1),
            "r120d": round(r120, 1)
        },
        "vs_ma": {
            "pct_ma5": round(p5, 1),
            "pct_ma10": round(p10, 1),
            "pct_ma20": round(p20, 1),
            "pct_ma60": round(p60, 1)
        },
        "factor_scores": scores
    }

# ============ MAIN ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quant_model_v2.py <code1,code2,...>")
        sys.exit(1)

    codes_arg = sys.argv[1]
    # Also accept optional name mapping
    name_map = {}
    if len(sys.argv) >= 3:
        name_map = json.loads(sys.argv[2])

    default_names = {
        "512760": "芯片ETF国泰", "159995": "芯片ETF华夏", "562500": "机器人ETF华夏",
        "159819": "AIETF易方达", "512880": "证券ETF国泰", "588000": "科创50ETF华夏",
        "159516": "半导体设备ETF", "159992": "创新药ETF", "159659": "纳指ETF招商",
        "518850": "黄金ETF华夏", "159183": "新能源车ETF", "512400": "有色ETF南方",
        "588200": "科创芯片ETF", "512170": "医疗ETF华宝", "159227": "航空航天ETF华夏",
        "512670": "国防ETF鹏华", "512810": "军工ETF华宝", "512680": "军工ETF广发",
        "159870": "化工ETF鹏华", "516020": "化工ETF华宝", "159980": "有色ETF大成",
        "515070": "AIETF华夏", "159770": "机器人ETF天弘", "159915": "创业板ETF",
        "512890": "红利低波ETF", "159928": "消费ETF", "159865": "养殖ETF",
        "512200": "房地产ETF", "159869": "游戏ETF", "159825": "农业ETF",
        "516970": "基建ETF", "512100": "中证1000ETF", "510300": "沪深300ETF",
        "513100": "纳指ETF国泰", "513500": "标普500ETF", "159611": "电力ETF"
    }

    codes = [c.strip() for c in codes_arg.split(",")]

    # Fetch benchmark (CSI 300) first
    bench_klines = fetch_kline("510300", DAYS)
    _, bench_c, _, _, _ = parse_klines(bench_klines) if bench_klines else ([], [], [], [], [])

    results = []
    for code in codes:
        name = name_map.get(code, default_names.get(code, code))
        klines = fetch_kline(code, DAYS)
        if not klines:
            print(f"WARN: {code} {name} - no data", file=sys.stderr)
            continue

        opens, closes, highs, lows, volumes = parse_klines(klines)
        if len(closes) < 30:
            print(f"WARN: {code} {name} - only {len(closes)} days", file=sys.stderr)
            continue

        try:
            result = score_etf(code, name, closes, highs, lows, volumes, bench_c if bench_c else None)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {code} {name} - {e}", file=sys.stderr)

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    print(json.dumps(results, ensure_ascii=False, indent=2))
