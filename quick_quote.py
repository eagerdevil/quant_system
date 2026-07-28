import sys, urllib.request

def fetch_sina(codes):
    results = {}
    for code in codes:
        if code.startswith('5') or code.startswith('6'):
            full = f"sh{code}"
        else:
            full = f"sz{code}"
        url = f"https://hq.sinajs.cn/list={full}"
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            text = resp.read().decode("gbk", errors="replace")
            if '="' in text:
                parts = text.split('="')
                if len(parts) > 1:
                    data = parts[1].split(",")
                    if len(data) > 10:
                        results[code] = {
                            "name": data[0],
                            "open": float(data[1]) if data[1] else 0,
                            "prev_close": float(data[2]) if data[2] else 0,
                            "price": float(data[3]) if data[3] else 0,
                            "high": float(data[4]) if data[4] else 0,
                            "low": float(data[5]) if data[5] else 0,
                            "volume": float(data[8]) if len(data) > 8 and data[8] else 0,
                            "amount": float(data[9]) if len(data) > 9 and data[9] else 0,
                        }
        except Exception as e:
            results[code] = {"error": str(e)}
    return results

holdings = ["518850", "159183", "159659", "562500"]
etf_data = fetch_sina(holdings)
for code, d in etf_data.items():
    if "error" not in d:
        chg = d["price"] - d["prev_close"]
        chg_pct = (chg / d["prev_close"] * 100) if d["prev_close"] else 0
        print(f"{code} {d['name']:20s} {d['price']:>8.3f}  {chg:+.3f} ({chg_pct:+.2f}%)  H:{d['high']:.3f} L:{d['low']:.3f}  vol:{d['volume']/1e4:.0f}W")
    else:
        print(f"{code} ERROR: {d['error']}")

print()
indices = ["sh000001", "sz399001", "sh000300"]
idx_data = fetch_sina(indices)
for code, d in idx_data.items():
    if "error" not in d:
        chg = d["price"] - d["prev_close"]
        chg_pct = (chg / d["prev_close"] * 100) if d["prev_close"] else 0
        print(f"{code} {d['name']:20s} {d['price']:>10.2f}  {chg:+.2f} ({chg_pct:+.2f}%)")
