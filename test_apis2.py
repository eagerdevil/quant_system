import sys
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')

# Test S4 with fixed code
from data_engine import fetch_total_volume
tv = fetch_total_volume()
print(f"S4: 全市场成交额 = {tv:.0f}亿")
if tv > 7000:
    print("  → S4=PASS (超过7000亿阈值)")
else:
    print(f"  → S4=FAIL (低于7000亿阈值)")

# Test S6 via collect_all_data path  
from data_engine import _estimate_margin_from_index, get_all_index_data
indices = get_all_index_data(60)
print(f"\nIndices keys: {list(indices.keys())[:3]}...")
if "000300" in indices:
    data = indices["000300"].get("data", [])
    print(f"HS300 data points: {len(data)}")
estimate = _estimate_margin_from_index(indices)
print(f"S6 estimated margin: {estimate}")
if estimate:
    print(f"  融资变化: {estimate.get('change', 0):.1f}亿")
    print(f"  S6={'PASS' if estimate.get('change', 0) > 0 else 'FAIL'}")

# Full test
from data_engine import fetch_north_bound_flow, fetch_market_breadth
nf = fetch_north_bound_flow(5)
total_nf = sum(f['net_flow'] for f in nf) if nf else 0
print(f"\nS3: 北向5日 = {total_nf:.0f}亿 → {'PASS' if total_nf > 0 else 'FAIL'}")

mb = fetch_market_breadth()
ld = mb.get('limit_down')
print(f"S5: 跌停 = {ld} → {'PASS' if ld is not None and ld < 20 else 'FAIL'}")
