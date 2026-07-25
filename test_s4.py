import sys
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')

# Test the Sina fallback directly 
import urllib.request

# Test sh000001
sh_url = "https://hq.sinajs.cn/list=sh000001"
req = urllib.request.Request(sh_url, headers={"Referer": "https://finance.sina.com.cn"})
resp = urllib.request.urlopen(req, timeout=10)
sh_text = resp.read().decode("gbk", errors="replace")
print(f"SH text: {sh_text[:200]}")
sh_parts = sh_text.split(",")
print(f"SH parts count: {len(sh_parts)}")
if len(sh_parts) > 10:
    print(f"SH parts[9] (amount?): {sh_parts[9]}")
    print(f"SH parts[10] (amount?): {sh_parts[10]}")

# Test sz399001  
sz_url = "https://hq.sinajs.cn/list=sz399001"
req2 = urllib.request.Request(sz_url, headers={"Referer": "https://finance.sina.com.cn"})
resp2 = urllib.request.urlopen(req2, timeout=10)
sz_text = resp2.read().decode("gbk", errors="replace")
print(f"\nSZ text: {sz_text[:200]}")
sz_parts = sz_text.split(",")
print(f"SZ parts count: {len(sz_parts)}")
if len(sz_parts) > 10:
    print(f"SZ parts[9] (amount?): {sz_parts[9]}")
    print(f"SZ parts[10] (amount?): {sz_parts[10]}")
