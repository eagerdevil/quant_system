import urllib.request, json

# Fetch K-line from East Money
def fetch_kline(code, days=60):
    if code.startswith('5') or code.startswith('6'):
        secid = f"1.{code}"
    else:
        secid = f"0.{code}"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt={days}"
    req = urllib.request.Request(url, headers={"Referer": "https://quote.eastmoney.com"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        if data and data.get("data") and data["data"].get("klines"):
            klines = []
            for line in data["data"]["klines"]:
                parts = line.split(",")
                klines.append({
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": float(parts[5]),
                    "amount": float(parts[6])
                })
            return klines
    except Exception as e:
        print(f"Error: {e}")
    return []

# SMA
def sma(values, period):
    if len(values) < period:
        return 0
    return sum(values[-period:]) / period

# Analysis for each holding
codes = {
    "518850": "黄金ETF华夏",
    "159183": "新能源车ETF招商",
    "159659": "纳斯达克100ETF招商",
    "562500": "机器人ETF华夏"
}

for code, name in codes.items():
    kline = fetch_kline(code, 60)
    if not kline:
        print(f"{code} {name}: NO DATA")
        continue

    closes = [k["close"] for k in kline]
    current = closes[-1]
    prev_close = closes[-2] if len(closes) > 2 else current

    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)

    # Returns
    r5d = (current / closes[-6] - 1) * 100 if len(closes) > 6 else 0
    r10d = (current / closes[-11] - 1) * 100 if len(closes) > 11 else 0
    r20d = (current / closes[-21] - 1) * 100 if len(closes) > 21 else 0

    # Consecutive up/down days
    up_days = 0
    down_days = 0
    for i in range(len(closes)-1, max(0, len(closes)-8), -1):
        if closes[i] > closes[i-1]:
            up_days += 1
        else:
            break
    for i in range(len(closes)-1, max(0, len(closes)-8), -1):
        if closes[i] < closes[i-1]:
            down_days += 1
        else:
            break

    # Volume avg (last 5d vs 20d)
    vols = [k["volume"] for k in kline]
    vol_5d = sum(vols[-5:]) / 5
    vol_20d = sum(vols[-20:]) / 20
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

    above_ma = sum([1 for m in [ma5, ma10, ma20] if current > m])

    print(f"\n{'='*60}")
    print(f"{code} {name}  |  {current:.3f}")
    print(f"  均线: MA5={ma5:.3f} MA10={ma10:.3f} MA20={ma20:.3f} | 站上{above_ma}/3条")
    print(f"  涨幅: 5d={r5d:+.1f}%  10d={r10d:+.1f}%  20d={r20d:+.1f}%")
    print(f"  连涨{up_days}天 / 连跌{down_days}天  |  量比(5d/20d)={vol_ratio:.1f}x")
    print(f"  60日最高={max(closes):.3f}  60日最低={min(closes):.3f}")
