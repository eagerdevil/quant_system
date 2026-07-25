import sys
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')
from data_engine import fetch_index_daily

# Test what fetch_index_daily returns
sh = fetch_index_daily("000001", 1)
print(f"SH data: {sh}")
if sh:
    d = sh[0]
    print(f"  open={d.get('open')} close={d.get('close')}")
    change = (d['close'] / d['open'] - 1) if d.get('open') else 0
    print(f"  change={change:.4f}")

sz = fetch_index_daily("399001", 1)
print(f"\nSZ data: {sz}")
if sz:
    d = sz[0]
    print(f"  open={d.get('open')} close={d.get('close')}")
    change = (d['close'] / d['open'] - 1) if d.get('open') else 0
    print(f"  change={change:.4f}")
