import json, math

raw = {
    "512760":{"name":"芯片ETF国泰","closes":[1.628,1.588,0.798,0.796,0.768,0.792,0.769,0.767,0.779,0.829,0.83,0.846,0.849,0.869,0.865,0.876,0.881,0.897,0.884,0.908,0.897,0.91,0.953,0.941,0.94,0.983,1.042,1.054,1.026,1.092,1.097,1.127,1.1,1.092,1.107,1.149,1.201,1.159,1.186,1.269,1.256,1.227,1.237,1.164,1.102,1.119,1.158,1.18,1.121,1.068,1.123,1.114,1.122,1.114,1.179,1.187,1.261,1.317,1.355,1.373]},
    "562500":{"name":"机器人ETF华夏","closes":[0.938,0.925,0.93,0.926,0.918,0.943,0.922,0.912,0.909,0.967,0.956,0.971,0.971,0.993,0.982,1.002,1.009,1.017,1.014,1.022,1.013,1.006,1.033,1.012,1.021,1.032,1.056,1.097,1.122,1.141,1.13,1.156,1.128,1.164,1.183,1.198,1.193,1.187,1.208,1.216,1.205,1.163,1.165,1.118,1.102,1.122,1.127,1.129,1.163,1.168,1.179,1.132,1.09,1.087,1.121,1.142,1.144,1.17,1.154,1.155]},
    "512880":{"name":"证券ETF国泰","closes":[1.059,1.035,1.041,1.037,1.029,1.045,1.025,1.016,1.015,1.054,1.031,1.068,1.075,1.081,1.073,1.081,1.076,1.075,1.066,1.075,1.066,1.056,1.057,1.07,1.076,1.078,1.094,1.089,1.08,1.097,1.102,1.095,1.071,1.048,1.047,1.056,1.049,1.054,1.037,1.045,1.056,1.043,1.021,1.033,1.035,1.037,1.035,1.022,1.016,1.001,1.005,1.018,1.008,1.045,1.072,1.082,1.081,1.05,1.131,1.148]},
    "588000":{"name":"科创50ETF华夏","closes":[1.385,1.357,1.37,1.359,1.324,1.369,1.33,1.324,1.342,1.425,1.415,1.437,1.448,1.479,1.482,1.498,1.499,1.527,1.504,1.529,1.508,1.531,1.591,1.568,1.575,1.654,1.744,1.769,1.727,1.811,1.816,1.865,1.815,1.787,1.803,1.871,1.931,1.86,1.887,1.998,1.968,1.911,1.942,1.844,1.754,1.781,1.823,1.832,1.763,1.683,1.755,1.744,1.756,1.756,1.844,1.855,1.939,2.017,2.056,2.075]},
    "518850":{"name":"黄金ETF华夏","closes":[9.718,9.532,9.576,9.736,9.784,10.101,9.825,9.915,9.921,10.197,9.993,10.06,10.012,10.098,10.152,10.182,10.12,10.12,10.097,10.096,9.988,9.91,9.982,9.819,9.726,9.762,9.88,10.001,9.974,9.861,9.892,9.894,9.883,9.642,9.602,9.6,9.451,9.534,9.546,9.6,9.541,9.429,9.215,9.483,9.432,9.498,9.366,9.381,9.337,9.04,9.089,8.819,8.586,8.742,9.011,9.033,9.043,8.992,8.8,8.7]},
    "512400":{"name":"有色ETF南方","closes":[1.952,1.919,1.976,2.008,1.983,2.026,1.991,1.971,1.992,2.119,2.103,2.096,2.096,2.136,2.104,2.17,2.167,2.179,2.18,2.179,2.098,2.118,2.095,2.043,2.14,2.119,2.199,2.207,2.206,2.208,2.188,2.209,2.13,2.039,2.009,1.981,1.985,1.947,2.019,2.025,2.095,2.013,1.991,1.965,1.949,2.015,2.032,1.97,1.926,1.82,1.869,1.843,1.88,1.976,2.063,2.049,2.045,2.05,2.155,2.034]},
    "588200":{"name":"科创芯片ETF","closes":[2.413,2.363,2.383,2.372,2.294,2.377,2.307,2.311,2.36,2.514,2.519,2.565,2.589,2.65,2.648,2.679,2.699,2.745,2.71,2.778,2.735,2.806,2.938,2.893,2.899,3.088,3.26,3.332,3.241,3.428,3.467,3.566,3.483,3.422,3.458,3.594,3.766,3.569,3.641,3.904,3.817,3.707,3.798,3.586,3.37,3.445,3.571,3.622,3.456,3.299,3.483,3.476,3.539,3.496,3.708,3.742,3.949,4.126,4.229,4.289]},
    "512170":{"name":"医疗ETF华宝","closes":[0.324,0.319,0.327,0.33,0.33,0.341,0.339,0.33,0.332,0.339,0.336,0.339,0.335,0.337,0.337,0.337,0.331,0.334,0.332,0.336,0.335,0.335,0.332,0.34,0.338,0.336,0.335,0.338,0.337,0.34,0.337,0.336,0.329,0.327,0.322,0.325,0.321,0.324,0.321,0.322,0.32,0.317,0.313,0.315,0.313,0.309,0.305,0.301,0.3,0.293,0.292,0.291,0.293,0.298,0.297,0.293,0.294,0.296,0.297,0.302]}
}

def sma(data, n):
    if len(data) < n: return [None]*len(data)
    out = [None]*(n-1)
    for i in range(n-1, len(data)):
        out.append(sum(data[i-n+1:i+1])/n)
    return out

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

def macd(closes):
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    dif = [e12-e26 if (e12 and e26) else 0 for e12,e26 in zip(ema12, ema26)]
    dea = ema(dif, 9)
    hist = [(d-dea[i])*2 for i,d in enumerate(dif)]
    return dif, dea, hist

def bollinger(closes, n=20):
    ma = sma(closes, n)
    upper, lower = [None]*len(closes), [None]*len(closes)
    for i in range(n-1, len(closes)):
        window = closes[i-n+1:i+1]
        avg = ma[i]
        std = math.sqrt(sum((x-avg)**2 for x in window)/n)
        upper[i] = avg + 2*std
        lower[i] = avg - 2*std
    return ma, upper, lower

def max_drawdown(closes):
    peak, md = closes[0], 0
    for c in closes[1:]:
        if c > peak: peak = c
        dd = (peak-c)/peak
        if dd > md: md = dd
    return md

def sharpe(closes, rf=0.025):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    std_ret = math.sqrt(sum((r-avg_ret)**2 for r in rets)/(len(rets)-1)) if len(rets)>1 else 0.01
    if std_ret == 0: return 0
    return (avg_ret - rf/252)/std_ret * math.sqrt(252)

def volatility(closes):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg = sum(rets)/len(rets)
    return math.sqrt(sum((r-avg)**2 for r in rets)/(len(rets)-1)) * math.sqrt(252)

def consecutive_up(closes):
    cnt = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] > closes[i-1]: cnt += 1
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

def pct_above_ma(closes, ma_n):
    ma = sma(closes, ma_n)
    if ma[-1] is None: return 0
    return (closes[-1]/ma[-1] - 1)*100

print("="*75)
print("  QUANTITATIVE MULTI-FACTOR MODEL — 60-Day Analysis")
print("="*75)

results = []
for code, data in raw.items():
    name = data["name"]
    c = data["closes"]
    v = data.get("volumes", [1]*len(c))

    rsi14 = rsi(c, 14)
    dif, dea, hist = macd(c)
    bb_ma, bb_up, bb_low = bollinger(c, 20)

    cur = c[-1]
    cons_up = consecutive_up(c)
    r5 = ret_n(c, 5)
    r10 = ret_n(c, 10)
    r20 = ret_n(c, 20)
    tvol = volatility(c)
    maxdd = max_drawdown(c)
    shrp = sharpe(c)
    vrat = vol_ratio(v, 5)
    pma5 = pct_above_ma(c, 5)
    pma10 = pct_above_ma(c, 10)
    pma20 = pct_above_ma(c, 20)

    rsi_now = rsi14[-1] if rsi14[-1] else 50
    dif_now = dif[-1]
    hist_now = hist[-1]

    bb_pos = 50
    if bb_up[-1] and bb_low[-1] and bb_up[-1]!=bb_low[-1]:
        bb_pos = (cur - bb_low[-1])/(bb_up[-1] - bb_low[-1])*100

    # ===== SCORING =====
    score = 0
    details = []

    # Trend (20)
    trend = 0
    if dif_now > 0 and hist_now > 0: trend += 9
    elif dif_now > 0: trend += 5
    elif dif_now > -0.01: trend += 8  # near golden cross
    if 0 < pma5 < 8: trend += 6
    elif -3 < pma5 <= 0: trend += 8  # slightly below MA5 = pullback
    if pma20 > 0: trend += 5
    details.append(f"趋势:{trend}/20")
    score += trend

    # RSI Position (20)
    rsi_score = 0
    if 35 <= rsi_now <= 45: rsi_score = 18
    elif 45 < rsi_now <= 55: rsi_score = 14
    elif 30 <= rsi_now < 35: rsi_score = 15
    elif 55 < rsi_now <= 65: rsi_score = 10
    elif rsi_now < 30: rsi_score = 8
    else: rsi_score = 5
    details.append(f"RSI:{rsi_score}/20 (RSI={rsi_now:.1f})")
    score += rsi_score

    # Pullback freshness (20)
    pull_score = 0
    if cons_up == 0: pull_score = 18
    elif cons_up == 1: pull_score = 16
    elif cons_up == 2: pull_score = 12
    elif cons_up == 3: pull_score = 8
    elif cons_up <= 5: pull_score = 5
    else: pull_score = 0
    details.append(f"回调:{pull_score}/20 (连涨{cons_up}天)")
    score += pull_score

    # Momentum health (15)
    mom_score = 0
    if 0.9 <= vrat <= 1.3: mom_score += 8
    elif 0.7 <= vrat <= 1.5: mom_score += 5
    else: mom_score += 2
    if maxdd < 0.12: mom_score += 7
    elif maxdd < 0.20: mom_score += 5
    elif maxdd < 0.30: mom_score += 3
    details.append(f"动量:{mom_score}/15 (量比={vrat:.1f} MDD={maxdd*100:.0f}%)")
    score += mom_score

    # Risk/Reward (15)
    rr_score = 0
    if shrp > 1.5: rr_score += 8
    elif shrp > 0.8: rr_score += 6
    elif shrp > 0.3: rr_score += 4
    else: rr_score += 2
    if 25 <= bb_pos <= 75: rr_score += 7
    elif 10 <= bb_pos < 25: rr_score += 5
    else: rr_score += 3
    details.append(f"风险回报:{rr_score}/15 (Sharpe={shrp:.2f} BB%={bb_pos:.0f})")
    score += rr_score

    # Recent return (10)
    ret_score = 0
    if -3 <= r5 <= 0: ret_score = 10
    elif 0 < r5 <= 3: ret_score = 8
    elif -5 <= r5 < -3: ret_score = 7
    elif 3 < r5 <= 8: ret_score = 4
    else: ret_score = 2
    details.append(f"近期:{ret_score}/10 (5d={r5:+.1f}%)")
    score += ret_score

    if score >= 75: grade = "STRONG BUY"
    elif score >= 62: grade = "BUY"
    elif score >= 50: grade = "WATCH"
    elif score >= 38: grade = "HOLD OFF"
    else: grade = "AVOID"

    results.append((score, grade, name, code, cur, cons_up, rsi_now, dif_now, hist_now, r5, r10, r20, tvol, maxdd, shrp, vrat, pma5, pma20, bb_pos))

    print(f"\n  [{grade}] {name} ({code}) — Score: {score}/100")
    print(f"    现价:{cur:.4f} | RSI:{rsi_now:.1f} | 连涨:{cons_up}天 | 5日:{r5:+.1f}%")
    print(f"    MA5:{pma5:+.1f}% | MA20:{pma20:+.1f}% | BB位置:{bb_pos:.0f}%")
    print(f"    MACD:{dif_now:+.4f} Hist:{hist_now:+.4f} | Sharpe:{shrp:.2f}")
    print(f"    波动率:{tvol*100:.1f}% | 最大回撤:{maxdd*100:.1f}% | 量比:{vrat:.2f}")
    print(f"    {' | '.join(details)}")

# Sorted ranking
results.sort(key=lambda x: x[0], reverse=True)
print(f"\n{'='*75}")
print("  FINAL RANKING (Quant Score)")
print(f"{'='*75}")
for i, (score, grade, name, code, cur, cons_up, rsi_now, _, _, r5, _, _, _, _, _, _, _, _, _) in enumerate(results):
    print(f"  {i+1}. [{grade}] {name} ({code}) — {score}/100 | {cur:.4f} | RSI:{rsi_now:.0f} | 连涨{cons_up}天 | 5日:{r5:+.1f}%")
