import sys
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')
from data_engine import fetch_total_volume, fetch_json, _simple_get

# Step 1: Test primary URL
url = "https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f6"
data = fetch_json(url)
print(f"Primary data: {data}")

# Step 2: Test Sina fallback directly
import urllib.request
sh_url = "https://hq.sinajs.cn/list=sh000001"
req = urllib.request.Request(sh_url, headers={"Referer": "https://finance.sina.com.cn"})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    sh_text = resp.read().decode("gbk", errors="replace")
    sh_parts = sh_text.split(",")
    print(f"\nSH direct: parts count={len(sh_parts)}")
    if len(sh_parts) > 9:
        print(f"  parts[9] = {sh_parts[9]}")
        sh_amount = float(sh_parts[9]) / 1e8
        print(f"  SH amount = {sh_amount:.0f}亿")
except Exception as e:
    print(f"SH error: {e}")

sz_url = "https://hq.sinajs.cn/list=sz399001"
req2 = urllib.request.Request(sz_url, headers={"Referer": "https://finance.sina.com.cn"})
try:
    resp2 = urllib.request.urlopen(req2, timeout=10)
    sz_text = resp2.read().decode("gbk", errors="replace")
    sz_parts = sz_text.split(",")
    print(f"\nSZ direct: parts count={len(sz_parts)}")
    if len(sz_parts) > 9:
        print(f"  parts[9] = {sz_parts[9]}")
        sz_amount = float(sz_parts[9]) / 1e8
        print(f"  SZ amount = {sz_amount:.0f}亿")
except Exception as e:
    print(f"SZ error: {e}")

# Step 3: Now test _simple_get
print("\n=== Testing _simple_get ===")
try:
    resp3 = _simple_get(sh_url, timeout=10)
    print(f"_simple_get resp type: {type(resp3)}")
    if hasattr(resp3, 'read'):
        text3 = resp3.read().decode("gbk", errors="replace")
        print(f"_simple_get text[:100]: {text3[:100]}")
except Exception as e:
    print(f"_simple_get error: {e}")
