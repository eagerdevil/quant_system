import sys
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')
from data_engine import fetch_north_bound_flow, fetch_total_volume, fetch_market_breadth, fetch_margin_balance

print("=== S3: 北向资金 ===")
nf = fetch_north_bound_flow(5)
print(f"Result: {nf}")
if nf:
    total = sum(f['net_flow'] for f in nf)
    print(f"5日净流入合计: {total:.1f}亿 → S3={'PASS' if total > 0 else 'FAIL'}")

print("\n=== S4: 全市场成交额 ===")
tv = fetch_total_volume()
print(f"成交额: {tv:.0f}亿 → S4={'PASS' if tv > 7000 else 'FAIL'} (阈值7000亿)")

print("\n=== S5: 涨跌停家数 ===")
mb = fetch_market_breadth()
print(f"涨停: {mb.get('limit_up')}, 跌停: {mb.get('limit_down')}")
print(f"上涨: {mb.get('up_count')}, 下跌: {mb.get('down_count')}")
ld = mb.get('limit_down')
print(f"S5={'PASS' if ld is not None and ld < 20 else 'FAIL'} (阈值20)")

print("\n=== S6: 融资余额 ===")
mg = fetch_margin_balance()
print(f"Margin data: {mg}")
if mg:
    print(f"融资变化: {mg.get('change', 0):.0f}亿 → S6={'PASS' if mg.get('change',0) > 0 else 'FAIL'}")
else:
    print("S6: FAIL (no data)")
