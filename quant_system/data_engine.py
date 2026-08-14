#!/usr/bin/env python
"""
量化系统 模块1: 数据采集引擎
============================
覆盖：行情/指数/宏观/资金/情绪/基本面/事件
数据源：东方财富 API（主）+ 腾讯/新浪 API（备用）
"""
import json, urllib.request, time, sys, re, os, logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)
TODAY = datetime.now().strftime("%Y%m%d")  # 保持向后兼容

def get_today():
    """返回当前日期字符串 YYYYMMDD（每次调用实时计算）"""
    return datetime.now().strftime("%Y%m%d")

def _to_float(v, default=0.0):
    """安全转换数值；东财接口缺失值常返回 '-'/''/None，避免 TypeError/ValueError 击穿采集流程"""
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default
MAX_RETRY = 3

# K线字段索引（东方财富 push2his API 返回格式: date,open,close,high,low,volume,amount[,...]）
_K_DATE, _K_OPEN, _K_CLOSE, _K_HIGH, _K_LOW, _K_VOL, _K_AMT = range(7)


def _parse_kline(k_str):
    """安全解析单根K线字符串，返回 {date, open, close, high, low, volume, amount, turnover}"""
    parts = k_str.split(",")
    if len(parts) < 7:
        return None
    try:
        return {
            "date": parts[_K_DATE],
            "open": float(parts[_K_OPEN]),
            "close": float(parts[_K_CLOSE]),
            "high": float(parts[_K_HIGH]),
            "low": float(parts[_K_LOW]),
            "volume": float(parts[_K_VOL]),
            "amount": float(parts[_K_AMT]) if len(parts) > _K_AMT else 0.0,
            # v7.6: 换手率(%) — f61字段(索引10), 仅ETF K线请求该字段, 缺失时0.0
            "turnover": float(parts[10]) if len(parts) > 10 else 0.0,
        }
    except (ValueError, IndexError):
        return None


# GitHub Actions 环境检测（东方财富API可能封禁美国IP，GitHub Actions Runner在美国）
_ON_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"

# 东方财富API可用性标志（启动时预检，避免逐个超时）
_EASTMONEY_OK = True

def _probe_eastmoney():
    """GitHub Actions启动时快速探测东方财富API连通性，失败则全局跳过"""
    global _EASTMONEY_OK
    if not _ON_GITHUB:
        return  # 本地环境不需要探测
    try:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1&fields2=f51&klt=101&fqt=0&end=20260101&lmt=1"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data and data.get("data"):
                logger.info("  [PROBE] 东方财富API连通正常 ✓")
                return
    except Exception as e:
        pass
    _EASTMONEY_OK = False
    logger.info("  [PROBE] 东方财富API不可达，全局跳过(改用腾讯/新浪备用源)")

# 备用数据源开关（东方财富API被限时自动使用）
_USE_FALLBACK = False

# ============================================================
# 1. 指数日线数据
# ============================================================
INDEX_CODES = {
    "000001": "上证指数", "399001": "深证成指", "399006": "创业板指",
    "000300": "沪深300", "000905": "中证500", "000688": "科创50",
    "399673": "创业板50"
}

def fetch_json(url, timeout=10, retries=MAX_RETRY):
    # 东方财富API预检失败 → 直接跳过快失败，节省时间
    # 8/14修复: datacenter-web(北向资金真实源)与push2系分域判断, push2被限不连坐北向
    if not _EASTMONEY_OK and "eastmoney.com" in url and "datacenter" not in url:
        return None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Referer": "https://quote.eastmoney.com/",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.info(f"  [API ERROR] 第{attempt+1}次尝试失败: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(1.2 * (2 ** attempt))  # 8/14: 指数退避1.2/2.4/4.8s, 防WAF 707风暴
    return None

# ============================================================
# v7.6: K线增量缓存 — 每天全量重拉10分钟 → 增量1-2天
# 缓存按代码存储, 含数据最后日期; 限流时用缓存降级(滞后≤1-2天)
# ============================================================
KLINE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kline_cache")

def _kline_cache_path(code, ver=""):
    # ver: 复权版本标记(如"_qf1"), 防止fqt=0/1数据混用同文件(8/14)
    return os.path.join(KLINE_CACHE_DIR, f"{code}{ver}.json")

def _load_kline_cache(code, ver=""):
    """读取K线缓存, 返回 (klines, last_date) 或 (None, None)"""
    try:
        with open(_kline_cache_path(code, ver), 'r', encoding='utf-8') as f:
            d = json.load(f)
        klines = d.get("klines", [])
        return klines, (klines[-1]["date"] if klines else "")
    except (OSError, json.JSONDecodeError, KeyError, IndexError):
        return None, None

def _save_kline_cache(code, klines, ver=""):
    """保存K线缓存(仅K线, 不覆盖实时数据)"""
    if not klines:
        return
    try:
        os.makedirs(KLINE_CACHE_DIR, exist_ok=True)
        with open(_kline_cache_path(code, ver), 'w', encoding='utf-8') as f:
            json.dump({"code": code, "klines": klines,
                       "last_date": klines[-1]["date"], "updated": get_today()}, f)
    except OSError:
        pass

def _merge_klines(cached, fresh):
    """按日期合并去重, 新数据覆盖旧数据, 保持时间升序"""
    if not cached:
        return fresh
    by_date = {k["date"]: k for k in cached}
    for k in fresh:
        by_date[k["date"]] = k
    return [by_date[d] for d in sorted(by_date)]

def _fetch_kline_with_cache(code, url, days, fallback_fn, ver=""):
    """
    通用增量缓存逻辑 (v7.6):
    1. 有缓存且最后日期==今天 → 直接用缓存(0请求)
    2. 有缓存 → 增量拉最近10根合并(1次请求); 失败→用缓存降级(标记stale)
    3. 无缓存 → 全量拉取并落缓存
    ver: 复权版本标记, 不同fqt的数据存不同缓存文件(8/14)
    """
    cached, last_date = _load_kline_cache(code, ver)
    today = get_today()
    # 周末必休市: 直接用缓存(零请求)
    if cached and datetime.now().weekday() >= 5:
        return cached
    # 8/14修复: 缓存日期可能是"2026-08-13"(横杠)而today是"20260814"(无横杠),
    # 原比较永不相等→每天全量重拉浪费请求; 统一8位数字再比
    # 当日命中必须同时满足长度要求(回测等长历史场景days=1400+即使当日缓存也要全量补)
    if cached and str(last_date).replace("-", "") == today and len(cached) >= days:
        return cached  # 当日数据已缓存且长度足够, 零请求

    if cached and len(cached) >= days:
        # 增量: 只拉最近10根, 单次重试(限流时快速放弃用缓存降级)
        inc_url = url.replace(f"lmt={days}", "lmt=10")
        data = fetch_json(inc_url, timeout=8, retries=1)
        fresh = []
        if data and data.get("data") and data["data"].get("klines"):
            for k in data["data"]["klines"]:
                p = _parse_kline(k)
                if p:
                    fresh.append(p)
        if fresh:
            merged = _merge_klines(cached, fresh)
            _save_kline_cache(code, merged, ver)
            return merged
        # 增量失败: 缓存降级(数据滞后, 记录日志)
        logger.info(f"  [缓存降级] {code} 增量更新失败, 用缓存(截至{last_date})")
        return cached

    # 无缓存 或 缓存长度不足days(8/14: 回测等长历史场景days=1400+) — 全量拉取
    if cached:
        logger.info(f"  [缓存补全] {code} 缓存{len(cached)}根 < 需求{days}根, 全量重拉")
    data = fetch_json(url)
    klines = []
    if data and data.get("data") and data["data"].get("klines"):
        for k in data["data"]["klines"]:
            p = _parse_kline(k)
            if p:
                klines.append(p)
    if not klines and fallback_fn:
        klines = fallback_fn(code, days) or []
    if klines:
        _save_kline_cache(code, klines, ver)
    return klines

def fetch_index_daily(code, days=120):
    """获取指数日K线（东方财富主 + 腾讯备用 + v7.6增量缓存）"""
    market = "1" if code.startswith(("0","5","1")) else "0"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end={get_today()}&lmt={days}"
    return _fetch_kline_with_cache(code, url, days, _fetch_sina_index)

def get_all_index_data(days=120):
    """获取所有主要指数日线"""
    result = {}
    for code, name in INDEX_CODES.items():
        data = fetch_index_daily(code, days)
        if data:
            result[code] = {"name": name, "data": data}
    return result

# ============================================================
# 2. 申万一级行业指数
# ============================================================
SW_INDUSTRY_CODES = {
    "801010": "农林牧渔", "801020": "采掘", "801030": "化工",
    "801040": "钢铁", "801050": "有色金属", "801080": "电子",
    "801110": "家用电器", "801120": "食品饮料", "801130": "纺织服装",
    "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产", "801200": "商业贸易",
    "801210": "休闲服务", "801230": "综合", "801710": "建筑材料",
    "801720": "建筑装饰", "801730": "电气设备", "801740": "国防军工",
    "801750": "计算机", "801760": "传媒", "801770": "通信",
    "801780": "银行", "801790": "非银金融", "801880": "汽车",
    "801890": "机械设备"
}

def fetch_sw_industry_returns(days=60, max_fail=6):
    """
    获取申万行业指数收益率
    v7.6: 连续失败max_fail次即放弃返回部分结果 — 修复东财限流时28个串行请求
    每个失败3次重试, 全失败曾让每日任务卡死10+分钟
    """
    result = {}
    fail_streak = 0
    for code, name in SW_INDUSTRY_CODES.items():
        market = "0" if code.startswith("8") else "1"
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end={get_today()}&lmt={days}"
        data = fetch_json(url)
        if not data or not data.get("data"):
            fail_streak += 1
            if fail_streak >= max_fail:
                logger.info(f"  [降级] 申万行业接口连续失败{fail_streak}次, 跳过行业轮动(行业加成本轮不生效)")
                break
            continue
        fail_streak = 0
        closes = []
        for k in data["data"]["klines"]:
            try:
                closes.append(float(k.split(",")[2]))
            except (ValueError, IndexError):
                continue
        if len(closes) < 30:  # 8/14: 数据不完整跳过该行业(防单行解析崩溃拖垮整个行业轮动)
            fail_streak += 1
            continue
        result[code] = {"name": name, "closes": closes}
    return result

# ============================================================
# 3. 宏观数据
# ============================================================
def fetch_bond_yield():
    """获取十年期国债收益率"""
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=131.CNY10YR&fields=f43,f44,f45,f57,f58"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        return {"yield": d.get("f43", 0)/100, "change": d.get("f170", 0)/100, "name": "10Y国债"}
    return None

def fetch_shibor():
    """获取Shibor隔夜利率"""
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=101.SHIBORON&fields=f43,f44,f57,f58"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        return {"rate": d.get("f43", 0)/100, "name": "Shibor隔夜"}
    return None

# ============================================================
# 4. 资金流向数据
# ============================================================
def fetch_north_bound_flow(days=5):
    """
    获取沪深股通（北向）资金数据（v7.6, 8/5重写 — 移除假数据）

    重要背景：自2024-08-19起，沪深交易所停止披露北向资金当日/历史净买额，
    市场上任何声称"北向净买XX亿"的实时数据均为伪造或估算。
    本函数不再造假（旧版用上证指数K线冒充5日历史、腾讯板块资金流冒充北向），
    改为返回仍真实披露的「沪深股通成交额」（东财数据中心 RPT_MUTUAL_DEAL_HISTORY）。

    Returns:
        [{date, deal_amt(亿), net_flow: None, north_active: bool}] 或 None
        net_flow 恒为 None（停披露），north_active 供S3信号替代净买额使用。
    """
    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get?"
        "reportName=RPT_MUTUAL_DEAL_HISTORY&columns=ALL"
        f"&pageNumber=1&pageSize={max(days + 2, 12)}&sortColumns=TRADE_DATE&sortTypes=-1"
        "&filter=(MUTUAL_TYPE%3D%22003%22)"
    )
    data = fetch_json(url, timeout=15)
    if not data or not data.get("result") or not data["result"].get("data"):
        return None

    result = []
    for row in data["result"]["data"]:
        deal_amt = _to_float(row.get("DEAL_AMT"), 0.0)  # 单位: 百万元
        if deal_amt <= 0:
            continue
        result.append({
            "date": str(row.get("TRADE_DATE", ""))[:10],
            "deal_amt": round(deal_amt / 100.0, 1),  # 百万元 → 亿元
            "net_flow": None,  # 2024-08-19起官方停披露净买额，不再伪造
        })
    result = result[:days]
    if not result:
        return None

    # 标注成交活跃度：最新5日均成交 vs 前5日均（供S3信号替代净买额）
    recent = [r["deal_amt"] for r in result]
    recent_avg = sum(recent) / len(recent)
    if len(result) >= 10:
        prev_avg = sum(r["deal_amt"] for r in result[5:10]) / 5.0
        result[-1]["north_active"] = recent_avg > prev_avg
    else:
        result[-1]["north_active"] = None  # 历史不足，中性
    return result

def fetch_market_fund_flow():
    """获取全市场主力资金流向"""
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f62,f64,f66,f68,f70,f72"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    d = data["data"]
    return {
        "super_large": d.get("f62", 0)/1e8,
        "large": d.get("f64", 0)/1e8,
        "medium": d.get("f66", 0)/1e8,
        "small": d.get("f68", 0)/1e8,
        "main_net": d.get("f70", 0)/1e8
    }

def fetch_margin_balance():
    """
    获取融资余额 — 8/14修复: secid=128.RZYL为死亡接口(恒None)
    失败直接返回None(信号层按中性处理), 删除指数估算"伪造信号"
    """
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=128.RZYL&fields=f43,f44,f170"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        # f170为涨跌幅基点(123=1.23%), 原÷1e8≈0; 修正为÷100取百分比方向判断
        return {"balance": d.get("f43", 0)/1e8, "change": d.get("f170", 0)/100}
    return None

# ============================================================
# 4b. 机构资金追踪（新增）
# ============================================================
def fetch_sector_fund_flow():
    """获取行业主力资金流向（申万一级）"""
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=15&po=1&np=1&fltt=2&fid=f62&fs=m:90+t:2&fields=f12,f14,f62,f64,f66"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    result = []
    for d in data["data"].get("diff", [])[:10]:
        result.append({
            "code": d.get("f12",""), "name": d.get("f14",""),
            "main_net": d.get("f62", 0)/1e8,  # 主力净流入(亿)
            "super_large": d.get("f64", 0)/1e8,
            "large": d.get("f66", 0)/1e8
        })
    return result

def fetch_dragon_tiger():
    """获取龙虎榜机构净买入汇总"""
    # v7.1: m:0=全部A股, t:5=龙虎榜, fields=股票代码+名称+机构净买+总净买
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10&po=1&np=1&fltt=2&fid=f184&fs=m:0+t:5&fields=f12,f14,f184,f186"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    result = []
    for d in data["data"].get("diff", [])[:10]:
        result.append({
            "code": d.get("f12",""), "name": d.get("f14",""),
            "inst_net": d.get("f184", 0)/1e4,  # 机构净买(万)
            "total_net": d.get("f186", 0)/1e4
        })
    return result

def fetch_etf_flow_top():
    """
    获取ETF份额申赎动向（v7.6改造: 同时返回净流入TOP/净流出TOP, 供F18因子）
    f146=份额变化(排序字段), f147=资金流向
    Returns: {"inflow": [{code,name,share_change,flow}...10], "outflow": [...]} 或 None
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=40&po=1&np=1&fltt=2&fid=f146&fs=b:MK0021&fields=f12,f14,f146,f147"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    inflow, outflow = [], []
    for d in data["data"].get("diff", []):
        chg = _to_float(d.get("f146"))
        item = {"code": d.get("f12",""), "name": d.get("f14",""),
                "share_change": chg, "flow": _to_float(d.get("f147"))}
        if chg > 0 and len(inflow) < 10:
            inflow.append(item)
        elif chg < 0 and len(outflow) < 10:
            outflow.append(item)
        if len(inflow) >= 10 and len(outflow) >= 10:
            break
    return {"inflow": inflow, "outflow": outflow}

# fetch_north_bound_top 已于2026/8/14审查中删除：
# fs=b:MK0354 是东财【可转债板块】而非北向资金板块，返回"炬申转债"等可转债列表
# 冒充"北向资金偏好行业"属假数据(P0)，北向真实数据仅剩 datacenter 沪深股通成交额(north_flow)

# ============================================================
# 5. 市场情绪数据
# ============================================================
def fetch_market_breadth():
    """获取涨跌停家数、炸板率等（东方财富主 + 指数数据备用估算, 估算时标记estimated=True）"""
    estimated = False  # 8/14: 走估算分支时置True, 日报标注"估算"防假数据
    # v7.1: m:0=全部A股, t:3=涨停, t:4=跌停, t:1=上涨, t:0=下跌
    # 使用pz=1只取total字段（不需要具体股票列表）
    url_zt = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:0+t:3&fields=f12"
    zt_data = fetch_json(url_zt)
    limit_up = zt_data["data"]["total"] if (zt_data and zt_data.get("data")) else None

    # 跌停家数
    url_dt = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:0+t:4&fields=f12"
    dt_data = fetch_json(url_dt)
    limit_down = dt_data["data"]["total"] if (dt_data and dt_data.get("data")) else None

    # 上涨/下跌家数
    url_up = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:0+t:1&fields=f12"
    up_data = fetch_json(url_up)
    up_count = up_data["data"]["total"] if (up_data and up_data.get("data")) else None

    url_down = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:0+t:0&fields=f12"
    down_data = fetch_json(url_down)
    down_count = down_data["data"]["total"] if (down_data and down_data.get("data")) else None

    # v7.0: GitHub Actions备用 — 从指数涨跌推算涨跌比（近似）
    if up_count is None or down_count is None:
        try:
            sh_data = fetch_index_daily("000001", 1)
            sz_data = fetch_index_daily("399001", 1)
            if sh_data and sz_data:
                sh_change = (sh_data[0]["close"] / sh_data[0]["open"] - 1) if sh_data else 0
                sz_change = (sz_data[0]["close"] / sz_data[0]["open"] - 1) if sz_data else 0
                # 粗略估算：指数涨→多数上涨，指数跌→多数下跌
                avg_change = (sh_change + sz_change) / 2
                total_est = 5000  # A股总数约5000只
                if avg_change > 0.005:
                    up_ratio = 0.6 + avg_change * 5
                elif avg_change < -0.005:
                    up_ratio = max(0.2, 0.5 + avg_change * 5)
                else:
                    up_ratio = 0.5
                up_count = up_count or int(total_est * up_ratio)
                down_count = down_count or int(total_est * (1 - up_ratio))
                estimated = True
                logger.info(f"  [FALLBACK] 涨跌家数估算: 涨{up_count}/跌{down_count} (指数变化{avg_change*100:.1f}%)")
        except Exception as e:
            logger.info(f"  [FALLBACK] 涨跌家数备用源失败: {e}")

    # v7.1: 当limit_up/limit_down为None时（clist API失败），从涨跌比估算
    if limit_up is None or limit_down is None:
        estimated = True  # 8/14: 涨跌停家数为估算值, 日报需标注
        # 根据涨跌比估算涨跌停家数（全部源失败时用中性默认，避免荒谬值污染情绪评分）
        if up_count is None and down_count is None:
            up_ratio = 0.5
        else:
            up_ratio = (up_count or 2500) / max((up_count or 0) + (down_count or 0), 1)
        if up_ratio > 0.65:
            limit_up = limit_up or 80 + int((up_ratio - 0.65) * 200)
            limit_down = limit_down or max(0, 15 - int((up_ratio - 0.65) * 30))
        elif up_ratio < 0.35:
            limit_down = limit_down or 80 + int((0.35 - up_ratio) * 200)
            limit_up = limit_up or max(0, 15 - int((0.35 - up_ratio) * 30))
        else:
            limit_up = limit_up or 30
            limit_down = limit_down or 30

    return {
        "limit_up": limit_up, "limit_down": limit_down,
        "up_count": up_count, "down_count": down_count,
        "estimated": estimated,  # 8/14: 日报标注"估算"
        "total": (up_count or 0) + (down_count or 0)
    }

def fetch_total_volume():
    """
    获取全市场成交额(亿) — 沪深两市K线求和(与fetch_total_volume_history同口径) + Sina备用
    8/14修复: 原secid=1.000001只含沪市(约9,900亿 vs 真实全市场2.1万亿), 与S4动态分位口径错配
    """
    merged = {}
    for secid in ("1.000001", "0.399001"):
        url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
               f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f57&klt=101&fqt=0&end={get_today()}&lmt=2")
        data = fetch_json(url, timeout=12)
        if not data or not data.get("data") or not data["data"].get("klines"):
            continue
        try:
            merged[secid] = float(data["data"]["klines"][-1].split(",")[1]) / 1e8  # f57成交额(元)→亿
        except (ValueError, IndexError):
            continue
    if len(merged) == 2:
        return round(merged["1.000001"] + merged["0.399001"], 1)
    # v7.0: Sina备用（沪市+深市成交额之和）
    try:
        sh_url = "https://hq.sinajs.cn/list=sh000001"  # v7.1: s_前缀返回6字段compact格式，去掉s_获取完整32字段
        sh_resp = _simple_get(sh_url, timeout=10)
        if sh_resp:
            # v7.1: _simple_get可能返回requests.Response(无.read())或urllib响应(有.read())
            if hasattr(sh_resp, "read"):
                sh_text = sh_resp.read().decode("gbk", errors="replace")
            elif hasattr(sh_resp, "content"):
                sh_text = sh_resp.content.decode("gbk", errors="replace")
            elif hasattr(sh_resp, "text"):
                sh_text = sh_resp.text
            else:
                sh_text = str(sh_resp)
            sh_parts = sh_text.split(",")
            if len(sh_parts) > 9:
                sh_amount = float(sh_parts[9]) / 1e8  # v7.1: index=9, 元→亿
            else:
                sh_amount = 0
        else:
            sh_amount = 0

        sz_url = "https://hq.sinajs.cn/list=sz399001"  # v7.1: 同上
        sz_resp = _simple_get(sz_url, timeout=10)
        if sz_resp:
            # v7.1: 同上, 兼容requests.Response和urllib响应
            if hasattr(sz_resp, "read"):
                sz_text = sz_resp.read().decode("gbk", errors="replace")
            elif hasattr(sz_resp, "content"):
                sz_text = sz_resp.content.decode("gbk", errors="replace")
            elif hasattr(sz_resp, "text"):
                sz_text = sz_resp.text
            else:
                sz_text = str(sz_resp)
            sz_parts = sz_text.split(",")
            if len(sz_parts) > 9:
                sz_amount = float(sz_parts[9]) / 1e8  # v7.1: index=9, 元→亿
            else:
                sz_amount = 0
        else:
            sz_amount = 0

        total = sh_amount + sz_amount
        if total > 0:
            logger.info(f"  [FALLBACK] 全市场成交额(Sina): {total:.0f}亿")
            return total
    except Exception as e:
        logger.info(f"  [FALLBACK] 成交额备用源失败: {e}")
    return None  # 8/14: 全部失败返回None(信号层中性), 原返回0会让情绪评分误判恐慌

def fetch_total_volume_history(days=60):
    """
    获取全市场历史成交额序列(亿) — 上证指数+深证成指K线 amount(f57) 之和
    v7.6 (8/5): 供S4信号动态分位阈值使用（替代固定2万亿死线）
    失败/限流时返回 []，调用方降级为固定阈值。
    """
    merged = {}
    for secid in ("1.000001", "0.399001"):
        url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
               f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
               f"&klt=101&fqt=0&lmt={days}")
        data = fetch_json(url, timeout=12)
        if not data or not data.get("data") or not data["data"].get("klines"):
            continue
        for k in data["data"]["klines"]:
            parts = k.split(",")
            if len(parts) > 7:
                try:
                    merged[parts[0]] = merged.get(parts[0], 0.0) + float(parts[6]) / 1e8  # f57成交额(元)→亿
                except ValueError:
                    continue
    if not merged:
        return []
    seq = [{"date": d, "amount": round(merged[d], 1)} for d in sorted(merged)]
    return seq[-days:]

# ============================================================
# 5b. 市场情绪综合指标 (v5.0)
# ============================================================
def compute_market_sentiment(breadth, total_volume, north_flow_5d=None):
    """
    从涨跌比/成交量/资金流计算综合情绪评分 (0-100)。
    50=中性, >60=偏乐观, <40=偏恐慌。

    输入:
        breadth: {limit_up, limit_down, up_count, down_count}
        total_volume: 全市场成交额(亿)
        north_flow_5d: 北向5日累计(亿), 可选
    返回: {
        score: 0-100,
        level: "贪婪"|"偏乐观"|"中性"|"偏恐慌"|"恐慌",
        signals: {涨跌比, 量能, 涨停热度, ...},
        estimated: bool (广度数据含估算时True, 8/14新增)
    }
    """
    score = 50  # 中性基准
    signals = {}

    # 1. 涨跌比 (权重: 30%)
    up = breadth.get("up_count") or 0
    down = breadth.get("down_count") or 1
    if up + down > 0:
        ratio = up / (up + down)
        signals["涨跌比"] = round(ratio, 2)
        # ratio: 0.3恐慌 → -15, 0.5中性 → 0, 0.7亢奋 → +10
        adj = (ratio - 0.5) * 60
        score += adj * 0.30

    # 2. 成交活跃度 (权重: 25%)
    signals["成交额(亿)"] = round(total_volume, 0) if total_volume else 0
    if not total_volume:
        adj = 0  # 8/14: 数据缺失(0/None)中性处理, 原None>20000抛TypeError
    elif total_volume > 20000:
        adj = 15  # 活跃
    elif total_volume > 15000:
        adj = 8
    elif total_volume > 10000:
        adj = 0
    elif total_volume > 7000:
        adj = -5
    else:
        adj = -12  # 极度缩量
    score += adj * 0.25

    # 3. 涨停热度 (权重: 20%)
    lu = breadth.get("limit_up") or 0
    ld = breadth.get("limit_down") or 0
    signals["涨停家数"] = lu
    signals["跌停家数"] = ld
    if lu > 100:
        adj = 18
    elif lu > 60:
        adj = 10
    elif lu > 30:
        adj = 3
    else:
        adj = -5
    if ld > 50:
        adj -= 15  # 恐慌
    elif ld > 20:
        adj -= 7
    score += adj * 0.20

    # 4. 沪深股通成交活跃度 (权重: 15%) — v7.6: 北向净买额2024/8停披露，以成交额替代
    # 阈值按2026年常态(1500-2200亿/日)标定
    if north_flow_5d is not None:
        signals["沪深股通5日均成交(亿)"] = round(north_flow_5d, 1)
        if north_flow_5d > 2200:
            adj = 8   # 放量活跃，资金参与度高
        elif north_flow_5d > 1500:
            adj = 4   # 正常活跃
        elif north_flow_5d > 1000:
            adj = 0   # 中性
        else:
            adj = -6  # 低迷，资金观望
    else:
        adj = 0
    score += adj * 0.15

    # 5. 连板效应 (权重: 10%)
    if lu > 5 and ld < 10:
        score += 5 * 0.10

    score = max(0, min(100, round(score)))

    # 等级
    if score >= 70:
        level = "贪婪"
    elif score >= 58:
        level = "偏乐观"
    elif score >= 42:
        level = "中性"
    elif score >= 30:
        level = "偏恐慌"
    else:
        level = "恐慌"

    return {"score": score, "level": level, "signals": signals,
            "estimated": breadth.get("estimated", False)}  # 8/14: 广度估算标记透传

# ============================================================
# 6. ETF 基本面与因子数据
# ============================================================
KEY_ETFS = {
    # === 全球宽基 — A股 ===
    "510300": "沪深300ETF华泰柏瑞",
    "510500": "中证500ETF",
    "512100": "中证1000ETF南方",
    "159845": "中证1000ETF华夏",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "588050": "科创50ETF易方达",

    # === 全球宽基 — 美股 ===
    "513100": "纳指ETF国泰",
    "159659": "纳斯达克100ETF招商",
    "513500": "标普500ETF",

    # === 全球宽基 — 日股 ===
    "513520": "日经ETF",
    "513800": "日本东证指数ETF南方",

    # === 港股 ===
    "513180": "恒生科技ETF",

    # === 实盘持仓（必须保留：日报持仓价格更新依赖采集范围）===
    "159750": "港股科技50ETF招商",
    "159529": "标普消费ETF景顺",
}

# 用户重点关注列表（优先分析）
# 全球宽基ETF重点关注名单（用户决策 8/13: 只关注全球ETF，长期持有，寻求全球发展机会）
USER_WATCHLIST = [
    "510300",   # 沪深300ETF华泰柏瑞（A股宽基）
    "510500",   # 中证500ETF（A股宽基）
    "512100",   # 中证1000ETF南方（A股宽基）
    "159915",   # 创业板ETF（A股宽基）
    "588000",   # 科创50ETF（A股宽基）
    "513100",   # 纳指ETF国泰（美股）
    "159659",   # 纳斯达克100ETF招商（美股）
    "513500",   # 标普500ETF（美股）
    "513520",   # 日经ETF（日股）
    "513800",   # 日本东证指数ETF南方（日股）
    "513180",   # 恒生科技ETF（港股）
    "159750",   # 港股科技50ETF招商（港股，实盘持仓）
    "159529",   # 标普消费ETF景顺（美股，实盘持仓）
]

# 个股关注（独立处理）
USER_STOCKS = {
    "000995": "皇台酒业",
}

# ============================================================
# 备用数据源：腾讯/新浪 API（东方财富被限时使用）
# ============================================================
_HTTP_SESSION = None

def _get_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        import urllib.request as _urllib
        # 尝试用 requests，失败则用 urllib
        try:
            import requests as _requests
            _HTTP_SESSION = _requests.Session()
            _HTTP_SESSION.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn/",
            })
        except ImportError:
            _HTTP_SESSION = None
    return _HTTP_SESSION

def _simple_get(url, timeout=10):
    """简易HTTP GET，先尝试requests，回退urllib（均使用TLS验证）"""
    session = _get_session()
    if session:
        try:
            r = session.get(url, timeout=timeout)  # verify=True 默认
            if r.status_code == 200:
                return r
        except Exception:
            pass
    # 回退urllib（默认SSL上下文，验证证书）
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn/",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp
    except Exception:
        return None

def _code_to_market_prefix(code):
    """ETF/股票代码 -> 行情前缀 (shXXXXXX / szXXXXXX)

    5xxxxx/6xxxxx/58xxxx → 沪市(sh)，其余 → 深市(sz)
    适用于新浪、腾讯等API。
    """
    return f"sh{code}" if code.startswith(("5", "6", "58")) else f"sz{code}"

# 向后兼容别名（新浪和腾讯用同样的前缀规则）
_code_to_sina_prefix = _code_to_market_prefix
_code_to_tencent_prefix = _code_to_market_prefix

def _fetch_tencent_kline(code, days=250):
    """从腾讯API获取日K线（备用）"""
    prefix = _code_to_tencent_prefix(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,{days},qfq"
    resp = _simple_get(url, timeout=15)
    if not resp:
        return None
    try:
        if hasattr(resp, 'json'):
            data = resp.json()
        else:
            data = json.loads(resp.read())
    except Exception:
        return None
    if data.get("code") != 0:
        return None
    stock_data = data.get("data", {}).get(prefix, {})
    klines = stock_data.get("qfqday") or stock_data.get("day", [])
    if not klines:
        return None
    result = []
    for k in klines:
        if len(k) < 5:
            continue
        # 腾讯K线: [date, open, close, high, low, volume, amount?]
        # k[6] 有时为dict(嵌套数据), 需判断类型
        amt = k[6] if len(k) > 6 and isinstance(k[6], str) else "0"
        result.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]) if len(k) > 5 else 0.0,
            "amount": float(amt),
        })
    return result

def _fetch_sina_realtime(code):
    """从新浪API获取实时行情（备用）"""
    prefix = _code_to_sina_prefix(code)
    url = f"https://hq.sinajs.cn/list={prefix}"
    resp = _simple_get(url, timeout=10)
    if not resp:
        return None
    try:
        if hasattr(resp, 'text'):
            text = resp.text
        else:
            text = resp.read().decode('gbk', errors='ignore')
    except Exception:
        return None

    # 解析 var hq_str_sh510300="名称,今开,昨收,当前,最高,最低,..."
    match = re.search(r'"([^"]*)"', text)
    if not match:
        return None
    fields = match.group(1).split(",")
    if len(fields) < 10:
        return None

    # 新浪ETF/股票字段 (通用格式)
    # 0:名称, 1:今开, 2:昨收, 3:现价, 4:最高, 5:最低, 8:成交量, 9:成交额
    name = fields[0]
    try:
        price = float(fields[3])
        prev_close = float(fields[2])
        change_pct = (price / prev_close - 1) * 100 if prev_close > 0 else 0
        return {
            "price": price,
            "open": float(fields[1]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "volume": float(fields[8]) if len(fields) > 8 else 0,
            "amount": float(fields[9]) if len(fields) > 9 else 0,
            "change_pct": round(change_pct, 2),
            "prev_close": prev_close,
            "name": name,
            "code": code,
        }
    except (ValueError, IndexError):
        return None

def _fetch_sina_index(code, days=120):
    """从腾讯API获取指数日K线（备用）
    指数前缀: 000xxx→sh, 399xxx→sz
    """
    prefix = f"sh{code}" if code.startswith("0") else f"sz{code}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,{days},qfq"
    resp = _simple_get(url, timeout=15)
    if not resp:
        return None
    try:
        if hasattr(resp, 'json'):
            data = resp.json()
        else:
            data = json.loads(resp.read())
    except Exception:
        return None
    if data.get("code") != 0:
        return None
    stock_data = data.get("data", {}).get(prefix, {})
    klines = stock_data.get("qfqday") or stock_data.get("day", [])
    if not klines:
        return None
    result = []
    for k in klines:
        if len(k) < 5:
            continue
        # 腾讯K线格式: [date, open, close, high, low, volume]
        # k[6] 有时是dict(不复权数据), 跳过
        amt = k[6] if len(k) > 6 and isinstance(k[6], str) else "0"
        result.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]) if len(k) > 5 else 0.0,
            "amount": float(amt),
        })
    return result

def fetch_etf_kline(code, days=250):
    """获取ETF日K线（东方财富主 + 腾讯备用 + v7.6增量缓存）"""
    # 沪市ETF: 5xxxxx, 58xxxx; 深市ETF: 15xxxx, 16xxxx
    market = "1" if code.startswith(("5","58")) else "0"
    # v7.6: fields2加f60/f61(涨跌额/换手率), 供F17换手率分位因子
    # 8/14: fqt=0→1前复权(与fetch_stock_kline一致), ETF分红除息日价格不再跳空,
    #       收益率/趋势因子计算才正确; 缓存加_qf1版本隔离, 旧未复权缓存作废
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end={get_today()}&lmt={days}"
    return _fetch_kline_with_cache(code, url, days, _fetch_tencent_kline, ver="_qf1")

def fetch_stock_kline(code, days=250):
    """获取个股日K线（支持股票和ETF，东方财富主 + 腾讯备用）"""
    if code.startswith(("5","6","58")):
        market = "1"
    elif code.startswith(("0","3","2")):
        market = "0"
    else:
        market = "0"
    # v7.6: fields2加f60/f61(换手率, 供股票F17因子) + 增量缓存
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end={get_today()}&lmt={days}"
    return _fetch_kline_with_cache(code, url, days, _fetch_tencent_kline)

def fetch_stock_realtime(code):
    """获取个股/ETF实时行情（东方财富主 + 新浪备用）"""
    if code.startswith(("5","6","58")):
        market = "1"
    elif code.startswith(("0","3","2")):
        market = "0"
    else:
        market = "0"
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f169,f170,f171,f2,f3,f4,f15,f16,f17,f18,f20,f21"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        # 东财 push2: 价格/昨收为"分"单位(×100)，涨跌幅f170为"基点"(×100，如+5%→500)
        # 统一 /100 并做 "-"/缺失 防护（此前 |涨跌幅|<0.1% 时直接返回原值，错100倍）
        price = _to_float(d.get("f43")) / 100.0
        high = _to_float(d.get("f44")) / 100.0
        low = _to_float(d.get("f45")) / 100.0
        open_ = _to_float(d.get("f46")) / 100.0
        prev_close = _to_float(d.get("f60")) / 100.0
        change_pct = _to_float(d.get("f170")) / 100.0
        return {
            "price": round(price, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "open": round(open_, 4),
            "volume": _to_float(d.get("f47")), "amount": _to_float(d.get("f48")),
            "change_pct": round(change_pct, 2),
            "prev_close": round(prev_close, 4),
            "name": d.get("f58", ""), "code": d.get("f57", ""),
            "market_cap": _to_float(d.get("f20")), "float_cap": _to_float(d.get("f21"))
        }
    # 备用: 新浪API
    return _fetch_sina_realtime(code)

def fetch_etf_realtime(code):
    """获取ETF实时行情（东方财富主 + 新浪备用）"""
    # 沪市ETF: 5xxxxx, 58xxxx; 深市ETF: 15xxxx, 16xxxx
    market = "1" if code.startswith(("5","58")) else "0"
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f169,f170,f171,f2,f3,f4,f15,f16,f17,f18"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        # 东方财富API：ETF价格统一以厘(1/1000元)返回
        # v7.6: 原判断(f60>50→厘)在f60缺失时_scale=1 → 价格放大1000倍。
        # 改为多条件投票确定单位:
        #   1) 恒等式投票: 现价≈昨收×(1+涨跌幅)，选误差最小的尺度
        #   2) 高低价一致性: 现价>最高价说明尺度错误
        #   3) 价格区间: A股ETF现价合理区间0.1~100元
        _raw_prev = _to_float(d.get("f60"))
        _raw_price = _to_float(d.get("f43"))
        raw_high = _to_float(d.get("f44"))
        raw_chg = _to_float(d.get("f170"))

        def _pick_scale():
            if _raw_prev > 0 and _raw_price > 0:
                best_s, best_err = None, None
                for s in (1000.0, 100.0, 1.0):
                    if raw_high > 0 and _raw_price / s > raw_high / s:
                        continue  # 现价>最高价 → 尺度错误
                    est = _raw_prev / s * (1 + raw_chg / 10000.0)  # 由昨收反推现价
                    if est <= 0:
                        continue
                    err = abs(_raw_price / s - est) / est
                    if best_err is None or err < best_err:
                        best_s, best_err = s, err
                if best_s is not None and best_err < 0.15:  # 误差<15%才可信
                    return best_s
            if _raw_price > 0:
                # 无昨收/投票不信任: 用价格区间判断（A股ETF现价0.1~100元）
                for s in (1000.0, 100.0, 1.0):
                    p = _raw_price / s
                    if 0.1 <= p <= 100:
                        return s
            return 1000.0  # 兜底: 东财push2对ETF以厘返回
        _scale = _pick_scale()

        price = _raw_price / _scale

        # IOPV（实时估值，f169；部分QDII ETF返回无效值如-61/9）
        raw_iopv = _to_float(d.get("f169"))
        iopv = None
        if raw_iopv > 0:
            cand_iopv = raw_iopv / _scale
            # 合理性校验：IOPV 必须接近现价（偏差>30%视为垃圾值，避免10622%假溢价）
            if price > 0 and abs(cand_iopv / price - 1) < 0.30:
                iopv = cand_iopv

        # 涨跌幅（f170 基点单位 ×100，如+1.2%→120）
        raw_chg = _to_float(d.get("f170"))
        change_pct = raw_chg / 100.0

        return {
            "price": round(price, 4),
            "high": round(_to_float(d.get("f44")) / _scale, 4),
            "low": round(_to_float(d.get("f45")) / _scale, 4),
            "open": round(_to_float(d.get("f46")) / _scale, 4),
            "volume": _to_float(d.get("f47")), "amount": _to_float(d.get("f48")),
            "change_pct": round(change_pct, 2),
            "prev_close": round(_raw_prev / _scale, 4),
            "name": d.get("f58", ""), "code": d.get("f57", ""),
            "iopv": iopv  # 实时估值（QDII ETF可能为None）
        }
    # 备用: 新浪API
    return _fetch_sina_realtime(code)

# ============================================================
# 6b. ETF净值与溢价率（新增 — 修正量化模型对QDII溢价的盲区）
# ============================================================

# QDII/跨境ETF列表（含港股通，存在溢价风险）
# 以15、16、51、52、56开头且跟踪海外指数的ETF
QDII_ETF_CODES = {
    # 纳斯达克/标普
    "159659", "513100", "513500",
    # 恒生/中概/港股通
    "513180", "513130", "513050", "159750",
    # 日经/东证
    "513520", "513800",
    # 标普消费
    "159529",
    # 其他跨境（历史）
    "159660", "513060", "513880", "513220",
}

def fetch_etf_fund_nav(code):
    time.sleep(0.12)  # 8/14: 批量NAV拉取(15-47只)节流, 防fund.eastmoney限流封IP
    """
    获取ETF最新官方净值（基金公司T+1公布）
    数据源: 天天基金API
    返回: {"nav": float, "date": str, "name": str} 或 None
    """
    try:
        url = f"https://api.fund.eastmoney.com/f10/lsjz?callback=&fundCode={code}&pageIndex=1&pageSize=2"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://fundf10.eastmoney.com/",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        lsjz_list = data.get("Data", {}).get("LSJZList", [])
        if lsjz_list:
            latest = lsjz_list[0]
            return {
                "nav": float(latest.get("DWJZ", 0)),
                "date": latest.get("FSRQ", ""),
                "name": data.get("Data", {}).get("FundName", "")
            }
    except Exception as e:
        logger.info(f"  [NAV ERROR] {code}: {e}")
    return None

def compute_etf_premium(code, current_price, nav_data=None):
    """
    计算ETF溢价率
    返回: {
        "premium_pct": float (正=溢价, 负=折价),
        "nav": float,
        "nav_date": str,
        "data_source": "official_nav" | "iopv" | "estimated" | "unavailable",
        "is_qdii": bool
    }
    """
    is_qdii = code in QDII_ETF_CODES

    # 优先级1: 官方净值（最可靠）
    if nav_data and nav_data.get("nav", 0) > 0:
        premium = (current_price / nav_data["nav"] - 1) * 100
        return {
            "premium_pct": round(premium, 2),
            "nav": round(nav_data["nav"], 4),
            "nav_date": nav_data.get("date", ""),
            "data_source": "official_nav",
            "is_qdii": is_qdii
        }

    # 优先级2: 尝试实时IOPV
    rt = fetch_etf_realtime(code)
    if rt and rt.get("iopv") and rt["iopv"] > 0:
        premium = (current_price / rt["iopv"] - 1) * 100
        return {
            "premium_pct": round(premium, 2),
            "nav": round(rt["iopv"], 4),
            "nav_date": get_today(),
            "data_source": "iopv",
            "is_qdii": is_qdii
        }

    # 优先级3: QDII ETF无数据时标记为数据缺失
    if is_qdii:
        return {
            "premium_pct": None,
            "nav": None,
            "nav_date": None,
            "data_source": "unavailable",
            "is_qdii": True
        }

    # 非QDII ETF: 溢价通常可忽略
    return {
        "premium_pct": 0.0,
        "nav": None,
        "nav_date": None,
        "data_source": "assumed_zero",
        "is_qdii": False
    }


def fetch_etf_nav_history(code, days=80):
    """
    获取ETF最近N个交易日历史净值（天天基金API，T+1公布）
    注意: 接口单页最多返回20条，需分页拉取
    返回: {date: nav} 字典（date格式 YYYY-MM-DD）或 None
    """
    try:
        result = {}
        page = 1
        remaining = days
        while remaining > 0 and page <= 10:
            url = (f"https://api.fund.eastmoney.com/f10/lsjz?callback=&fundCode={code}"
                   f"&pageIndex={page}&pageSize=20")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://fundf10.eastmoney.com/",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            lsjz_list = data.get("Data", {}).get("LSJZList", [])
            if not lsjz_list:
                break
            for item in lsjz_list:
                nav = float(item.get("DWJZ", 0))
                if nav > 0:
                    result[item.get("FSRQ", "")] = nav
            page += 1
            remaining -= len(lsjz_list)
        return result or None
    except Exception as e:
        logger.info(f"  [NAV HIST ERROR] {code}: {e}")
        return None


def compute_etf_premium_history(code, kline, days=60):
    """
    计算历史溢价序列及统计特征（价格K线 vs 历史官方净值，按日期对齐）
    溢价 = 收盘价/当日净值 - 1（历史净值T+1公布，为估算值；IOPV更准但无历史）

    返回:
    {
        "has_history": bool,
        "series": [{"date":..., "premium":...}, ...],   # 升序，最多days日
        "current": float,      # 最新溢价%
        "median": float,       # 历史溢价中位数%（中枢）
        "mean": float,         # 历史均值%
        "max": float,          # 历史最大溢价%
        "percentile": float,   # 当前溢价在历史样本中的百分位 0-100
        "trend_10d": float,    # 近10日平均溢价%
        "trend_60d": float,    # 全窗口平均溢价%
        "trend_gap": float,    # 近10日-全窗口（>0溢价扩大, <0收敛）
        "n": int,
    }
    """
    nav_hist = fetch_etf_nav_history(code, days=days + 20)
    if not nav_hist:
        return {"has_history": False}

    series = []
    for k in kline:
        d = k.get("date", "")
        if len(d) == 8 and d.isdigit():
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"  # K线 YYYYMMDD → 净值 YYYY-MM-DD
        nav = nav_hist.get(d)
        close = k.get("close")
        if nav and close:
            series.append({"date": d, "premium": round((close / nav - 1) * 100, 2)})

    if len(series) < 10:
        return {"has_history": False, "n": len(series)}

    series = series[-days:]
    prem_values = [s["premium"] for s in series]
    current = prem_values[-1]
    n = len(prem_values)

    # 当前溢价在历史中的百分位（< current 的样本占比）
    percentile = round(sum(1 for p in prem_values if p < current) / n * 100, 1)

    trend_10d = round(sum(prem_values[-10:]) / min(10, n), 2)
    trend_60d = round(sum(prem_values) / n, 2)

    import statistics
    return {
        "has_history": True,
        "series": series,
        "current": current,
        "median": round(statistics.median(prem_values), 2),
        "mean": round(sum(prem_values) / n, 2),
        "max": round(max(prem_values), 2),
        "percentile": percentile,
        "trend_10d": trend_10d,
        "trend_60d": trend_60d,
        "trend_gap": round(trend_10d - trend_60d, 2),
        "n": n,
    }


def calc_fear_index():
    """计算恐慌指数：今日下跌家数 / 总家数"""
    breadth = fetch_market_breadth()
    if breadth:
        total = breadth.get("total") or 0
        down = breadth.get("down_count") or 0
        if total > 0:
            return down / total * 100
    return 50


# ============================================================
# 8. 综合数据采集
# ============================================================
def collect_all_data(etf_codes=None, stock_codes=None, sequential=True):
    """一次性采集所有数据

    Args:
        etf_codes: ETF代码列表
        stock_codes: 个股代码列表
        sequential: True=逐只拉取(防限流), False=快速模式
    """
    logger.info("[DATA ENGINE] 开始采集数据...")

    # GitHub Actions: 快速探测东方财富API连通性，不通则全局跳过
    _probe_eastmoney()

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": get_today()
    }

    # 指数数据
    # P1修复: 60根不够quant_engine S2信号(需>=61根算MA60今昨对比), 60根时S2恒缺失→force_cap压仓死代码
    logger.info("  -> 指数日线...")
    result["indices"] = get_all_index_data(70)

    # 宏观数据
    logger.info("  -> 宏观数据...")
    result["bond_yield"] = fetch_bond_yield()
    result["shibor"] = fetch_shibor()

    # 资金数据
    logger.info("  -> 资金流向...")
    result["north_bound"] = fetch_north_bound_flow(10)
    result["fund_flow"] = fetch_market_fund_flow()
    # 8/14修复: 删除指数估算"伪造信号"(硬编码1.5万亿基准), 接口失败即None, S6信号层按中性处理
    result["margin"] = fetch_margin_balance()

    # 情绪数据
    logger.info("  -> 市场情绪...")
    result["breadth"] = fetch_market_breadth()
    result["total_volume"] = fetch_total_volume()
    result["volume_history"] = fetch_total_volume_history(60)  # v7.6: S4动态分位
    result["fear_index"] = calc_fear_index()

    # 机构资金追踪（新增）
    logger.info("  -> 机构资金追踪...")
    result["sector_flow"] = fetch_sector_fund_flow()
    result["dragon_tiger"] = fetch_dragon_tiger()
    result["etf_flow"] = fetch_etf_flow_top()
    # 北向偏好行业已删除(原接口实为可转债板块, 假数据P0)

    # ETF数据 - 逐只拉取
    all_etf_codes = list(dict.fromkeys((etf_codes or []) + USER_WATCHLIST))
    logger.info(f"  -> ETF数据 ({len(all_etf_codes)}只, {'逐只' if sequential else '快速'}模式)...")
    etf_data = {}
    fail_count = 0

    for i, code in enumerate(all_etf_codes):
        if sequential and i > 0:
            time.sleep(1.2)  # 逐只间隔1.2秒，防盘中API拥堵

        name = KEY_ETFS.get(code, code)

        # GitHub Actions: 跳过实时行情(hq.sinajs.cn封禁US IP)，用K线收盘价
        # 本地环境: 正常拉取实时行情
        kline, realtime = None, None
        if _ON_GITHUB:
            # 仅拉K线，不重试（K线API web.ifzq.gtimg.cn 对US IP友好）
            kline = fetch_etf_kline(code, 250)
            realtime = None  # 用K线收盘价作为现价
        else:
            for attempt in range(3):
                kline = fetch_etf_kline(code, 250) if not kline else kline
                realtime = fetch_etf_realtime(code) if not realtime else realtime
                if kline and realtime:
                    break
                if attempt < 2:
                    time.sleep(1.0)

        if kline:
            etf_data[code] = {"name": name, "kline": kline, "realtime": realtime}
            rt_tag = "K线" if (_ON_GITHUB or not realtime) else "OK"
            if sequential:
                logger.info(f"    [{i+1}/{len(all_etf_codes)}] {code} {name} - {rt_tag} ({len(kline)}d)")
        else:
            fail_count += 1
            logger.info(f"    [{i+1}/{len(all_etf_codes)}] {code} {name} - FAIL (3次重试后仍失败)")

    result["etfs"] = etf_data
    if fail_count > 0:
        logger.info(f"  !! {fail_count}只ETF数据获取失败")

    # 个股数据 - 逐只拉取
    if stock_codes:
        stock_codes_list = list(stock_codes)
        logger.info(f"  -> 个股数据 ({len(stock_codes_list)}只)...")
        stock_data = {}
        for i, code in enumerate(stock_codes_list):
            if sequential and i > 0:
                time.sleep(1.2)  # 逐只间隔1.2秒

            name = USER_STOCKS.get(code, code)
            kline, realtime = None, None
            if _ON_GITHUB:
                kline = fetch_stock_kline(code, 250)
                realtime = None
            else:
                for attempt in range(3):
                    kline = fetch_stock_kline(code, 250) if not kline else kline
                    realtime = fetch_stock_realtime(code) if not realtime else realtime
                    if kline and realtime:
                        break
                    if attempt < 2:
                        time.sleep(1.0)

            if kline:
                stock_data[code] = {"name": name, "kline": kline, "realtime": realtime}
                rt_tag = "K线" if (_ON_GITHUB or not realtime) else "OK"
                logger.info(f"    [{i+1}/{len(stock_codes_list)}] {code} {name} - {rt_tag} ({len(kline)}d)")
            else:
                logger.info(f"    [{i+1}/{len(stock_codes_list)}] {code} {name} - FAIL")

        result["stocks"] = stock_data

    logger.info("[DATA ENGINE] 采集完成")
    return result

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys, logging
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    etfs = sys.argv[1:] if len(sys.argv) > 1 else list(KEY_ETFS.keys())[:10]
    data = collect_all_data(etfs)
    # Save to file
    with open(os.path.join(SCRIPT_DIR, f"data_{get_today()}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[DATA ENGINE] 数据已保存到 data_{get_today()}.json")
