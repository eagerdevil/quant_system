import sys, json
sys.path.insert(0, 'E:/Claudecode/quant/quant_system')
from data_engine import fetch_json

# Test what clist returns
url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:110+t:1&fields=f12"
data = fetch_json(url)
print(f"clist up_count raw: {json.dumps(data, ensure_ascii=False)[:300]}")
if data:
    print(f"  data key: {data.get('data')}")
    print(f"  data type: {type(data.get('data'))}")
    print(f"  data is None? {data.get('data') is None}")
    
# What about the data['data']['total']?
if data and data.get('data'):
    print(f"  total: {data['data'].get('total')}")
elif data and data.get('data') is None:
    print(f"  data is explicitly null")
else:
    print(f"  no data at all")

# Now test the actual condition
up_count = data["data"]["total"] if (data and data.get("data")) else None
print(f"\nup_count = {up_count}")
print(f"up_count is None? {up_count is None}")
