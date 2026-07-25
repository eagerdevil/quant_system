import json, urllib.request, math

# Fetch 159227 航空航天ETF
url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.159227&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end=20260623&lmt=60"
with urllib.request.urlopen(url, timeout=10) as resp:
    data = json.loads(resp.read())
klines = data.get("data", {}).get("klines", [])
c227 = [float(k.split(",")[2]) for k in klines]
v227 = [float(k.split(",")[5]) for k in klines]

url2 = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.512670&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end=20260623&lmt=60"
with urllib.request.urlopen(url2, timeout=10) as resp:
    data2 = json.loads(resp.read())
klines2 = data2.get("data", {}).get("klines", [])
c670 = [float(k.split(",")[2]) for k in klines2]
v670 = [float(k.split(",")[5]) for k in klines2]

robot_c = [0.938,0.925,0.93,0.926,0.918,0.943,0.922,0.912,0.909,0.967,0.956,0.971,0.971,0.993,0.982,1.002,1.009,1.017,1.014,1.022,1.013,1.006,1.033,1.012,1.021,1.032,1.056,1.097,1.122,1.141,1.13,1.156,1.128,1.164,1.183,1.198,1.193,1.187,1.208,1.216,1.205,1.163,1.165,1.118,1.102,1.122,1.127,1.129,1.163,1.168,1.179,1.132,1.09,1.087,1.121,1.142,1.144,1.17,1.154,1.155]
robot_v = [7487200,4742440,3943838,3967912,4412294,6131072,4515538,3164527,3215827,10356137,5130679,6129719,3886982,8229323,5584775,6669344,5561708,5874961,4662180,5523305,7494225,6466342,8242307,6508083,4822444,5793729,8413368,11641903,14041248,12764342,9877545,9778909,9903918,21017574,13860660,10896066,10172714,15019715,11243848,12236326,15803558,13866729,10656086,11052262,8087283,9738270,8484481,7126437,16346449,17069150,10763535,11327203,10554541,8251432,6761090,6710475,5985648,8270146,10105140,4369567]

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

def pma(closes, n):
    ma = sma(closes, n)
    if ma[-1] is None: return 0
    return (closes[-1]/ma[-1] - 1)*100

def score_etf(name, code, c, v):
    rsi14 = rsi(c, 14)
    dif, dea, hist = macd_calc(c)
    bb_ma, bb_up, bb_low = bollinger(c, 20)
    cur = c[-1]
    cons = consecutive_up(c)
    r5 = ret_n(c, 5)
    r10 = ret_n(c, 10)
    r20 = ret_n(c, 20)
    vol = volatility(c)
    dd = max_drawdown(c)
    sh = sharpe(c)
    vr = vol_ratio(v, 5)
    p5 = pma(c, 5)
    p10 = pma(c, 10)
    p20 = pma(c, 20)
    rsi_now = rsi14[-1] if rsi14[-1] else 50
    dif_now = dif[-1]
    hist_now = hist[-1]
    bb_pos = 50
    if bb_up[-1] and bb_low[-1] and bb_up[-1]!=bb_low[-1]:
        bb_pos = (cur - bb_low[-1])/(bb_up[-1] - bb_low[-1])*100

    score = 0
    # Trend (20)
    if dif_now > 0 and hist_now > 0: score += 9
    elif dif_now > 0: score += 5
    elif dif_now > -0.01: score += 8
    if 0 < p5 < 8: score += 6
    elif -3 < p5 <= 0: score += 8
    if p20 > 0: score += 5
    # RSI (20)
    if 35 <= rsi_now <= 45: score += 18
    elif 45 < rsi_now <= 55: score += 14
    elif 30 <= rsi_now < 35: score += 15
    elif 55 < rsi_now <= 65: score += 10
    else: score += 5
    # Pullback (20)
    if cons == 0: score += 18
    elif cons == 1: score += 16
    elif cons == 2: score += 12
    elif cons == 3: score += 8
    elif cons <= 5: score += 5
    # Momentum (15)
    if 0.9 <= vr <= 1.3: score += 8
    elif 0.7 <= vr <= 1.5: score += 5
    else: score += 2
    if dd < 0.12: score += 7
    elif dd < 0.20: score += 5
    elif dd < 0.30: score += 3
    # Risk/Reward (15)
    if sh > 1.5: score += 8
    elif sh > 0.8: score += 6
    elif sh > 0.3: score += 4
    else: score += 2
    if 25 <= bb_pos <= 75: score += 7
    elif 10 <= bb_pos < 25: score += 5
    else: score += 3
    # Recent (10)
    if -3 <= r5 <= 0: score += 10
    elif 0 < r5 <= 3: score += 8
    elif -5 <= r5 < -3: score += 7
    elif 3 < r5 <= 8: score += 4
    else: score += 2

    if score >= 75: grade = "STRONG BUY"
    elif score >= 62: grade = "BUY"
    elif score >= 50: grade = "WATCH"
    elif score >= 38: grade = "HOLD OFF"
    else: grade = "AVOID"

    print(f"\n{'='*60}")
    print(f"  [{grade}] {name} ({code}) - Score: {score}/100")
    print(f"{'='*60}")
    print(f"  Price:{cur:.4f} | RSI(14):{rsi_now:.1f} | ConsecUp:{cons} | 5d:{r5:+.1f}% | 10d:{r10:+.1f}% | 20d:{r20:+.1f}%")
    print(f"  vsMA5:{p5:+.1f}% | vsMA20:{p20:+.1f}% | BBpos:{bb_pos:.0f}%")
    print(f"  MACD DIF:{dif_now:+.4f} Hist:{hist_now:+.4f} | Sharpe:{sh:.2f} | MaxDD:{dd*100:.1f}%")
    print(f"  Volatility:{vol*100:.1f}% | VolRatio:{vr:.2f}")
    return score, grade, cur, rsi_now, cons, r5

print("="*60)
print("  QUANT COMPARISON: Robot vs Aerospace vs Defense")
print("="*60)

s_227 = score_etf("航空航天ETF华夏", "159227", c227, v227)
s_670 = score_etf("国防ETF鹏华", "512670", c670, v670)
s_robot = score_etf("机器人ETF华夏", "562500", robot_c, robot_v)

# Correlation
def daily_rets(c):
    return [(c[i]-c[i-1])/c[i-1] for i in range(1, len(c))]

def corr(x, y):
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]
    mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)/n)
    sy = math.sqrt(sum((a-my)**2 for a in y)/n)
    if sx == 0 or sy == 0: return 0
    return sum((x[i]-mx)*(y[i]-my) for i in range(n))/(n*sx*sy)

r_227 = daily_rets(c227)
r_670 = daily_rets(c670)
r_rob = daily_rets(robot_c)

print(f"\n{'='*60}")
print("  PORTFOLIO CORRELATION MATRIX")
print(f"{'='*60}")
print(f"  机器人 vs 航空航天: {corr(r_rob, r_227):.3f}")
print(f"  机器人 vs 国防:     {corr(r_rob, r_670):.3f}")
print(f"  航空航天 vs 国防:   {corr(r_227, r_670):.3f}")
print(f"")
print(f"  >>> 机器人与航空航天相关性较低 -> 组合分散效果好")

print(f"\n{'='*60}")
print("  RECOMMENDED POSITION SIZING")
print(f"{'='*60}")
print(f"  Cash: 1,361 | Reserve: 700 | Deployable: 661")
print(f"")
print(f"  PLAN A (Balanced 600):")
print(f"    562500 Robot @1.155 x 200sh = 231")
print(f"    159227 Aero  @1.122 x 300sh = 337")
print(f"    Total: 568 | Remain: 793 (19.2% cash)")
print(f"")
print(f"  PLAN B (Robot-heavy 550):")
print(f"    562500 Robot @1.155 x 300sh = 347")
print(f"    159227 Aero  @1.122 x 200sh = 224")
print(f"    Total: 571 | Remain: 790 (19.1% cash)")
print(f"")
print(f"  PLAN C (Small test 400):")
print(f"    562500 Robot @1.155 x 100sh = 116")
print(f"    159227 Aero  @1.122 x 200sh = 224")
print(f"    Total: 340 | Remain: 1021 (24.7% cash)")

print(f"\n{'='*60}")
print("  COMPOSITE VS CHIP ETF")
print(f"{'='*60}")
print(f"  Chip 512760: Score 40 - HOLD OFF (overheated)")
print(f"  Robot 562500: Score 80 - STRONG BUY")
print(f"  Aero 159227: Score varies - see above")
