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
TODAY = datetime.now().strftime("%Y%m%d")
MAX_RETRY = 3

# GitHub Actions 环境检测（东方财富API屏蔽美国IP，GitHub Actions Runner在美国）
_ON_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"

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

def fetch_json(url, timeout=10):
    # GitHub Actions 在美国，东方财富API封禁美国IP → 直接走备用接口
    if _ON_GITHUB and "eastmoney.com" in url:
        logger.info(f"  [SKIP] GitHub环境跳过东方财富，直接走备用源")
        return None

    for attempt in range(MAX_RETRY):
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
            if attempt < MAX_RETRY - 1:
                time.sleep(1.2)  # 逐只间隔1.2秒
    return None

def fetch_index_daily(code, days=120):
    """获取指数日K线（东方财富主 + 腾讯备用）"""
    market = "1" if code.startswith(("0","5","1")) else "0"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end={TODAY}&lmt={days}"
    data = fetch_json(url)
    if data and data.get("data"):
        klines = data["data"]["klines"]
        result = []
        for k in klines:
            parts = k.split(",")
            result.append({
                "date": parts[0], "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]), "volume": float(parts[5]),
                "amount": float(parts[6])
            })
        return result
    # 备用: 腾讯API
    return _fetch_sina_index(code, days)

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

def fetch_sw_industry_returns(days=60):
    """获取申万行业指数收益率"""
    result = {}
    for code, name in SW_INDUSTRY_CODES.items():
        market = "0" if code.startswith("8") else "1"
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&end={TODAY}&lmt={days}"
        data = fetch_json(url)
        if not data or not data.get("data"):
            continue
        closes = [float(k.split(",")[2]) for k in data["data"]["klines"]]
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
    """获取北向资金净买入（东方财富主 + Tencent备用）"""
    url = f"https://push2.eastmoney.com/api/qt/kamt.kline/get?fields1=f1,f3&fields2=f2,f3,f4,f6,f8,f10&klt=101&lmt={days}"
    data = fetch_json(url)
    if data and data.get("data"):
        result = []
        for k in data["data"]["klines"]:
            parts = k.split(",")
            result.append({"date": parts[0], "net_flow": float(parts[1])/1e8})  # 亿元
        return result
    # v7.0: Tencent备用（北向资金实时数据）
    try:
        t_url = "https://qt.gtimg.cn/q=ff_bk0479"
        t_resp = _simple_get(t_url, timeout=10)
        if t_resp:
            t_text = t_resp.read().decode("gbk", errors="replace") if hasattr(t_resp, "read") else t_resp
            # 提取北向净流入（格式: ~字段间用~分隔）
            if "~" in t_text:
                t_parts = t_text.split("~")
                # 尝试找净流入字段（通常在位置4-6之间）
                for idx in [5, 4, 6]:
                    if idx < len(t_parts):
                        try:
                            flow = float(t_parts[idx]) / 1e8
                            if abs(flow) < 1000:  # 合理范围
                                logger.info(f"  [FALLBACK] 北向资金(Tencent): {flow:.1f}亿")
                                today = datetime.now().strftime("%Y-%m-%d")
                                return [{"date": today, "net_flow": flow}]
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logger.info(f"  [FALLBACK] 北向资金备用源失败: {e}")
    return None

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
    """获取融资余额"""
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=128.RZYL&fields=f43,f44,f170"
    data = fetch_json(url)
    if data and data.get("data"):
        d = data["data"]
        return {"balance": d.get("f43", 0)/1e8, "change": d.get("f170", 0)/1e8}
    return None

# ============================================================
# 4b. 机构资金追踪（新增）
# ============================================================
def fetch_sector_fund_flow():
    """获取行业主力资金流向（申万一级）"""
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=15&po=1&np=1&fltt=2&fid=f62&fs=m:90+t2&fields=f12,f14,f62,f64,f66"
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
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10&po=1&np=1&fltt=2&fid=f184&fs=m:110+t:5&fields=f12,f14,f184,f186"
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
    """获取ETF份额变化TOP（机构申购赎回动向）"""
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10&po=1&np=1&fltt=2&fid=f146&fs=b:MK0021&fields=f12,f14,f146,f147"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    result = []
    for d in data["data"].get("diff", [])[:10]:
        result.append({
            "code": d.get("f12",""), "name": d.get("f14",""),
            "share_change": d.get("f146", 0),  # 份额变化
            "flow": d.get("f147", 0)  # 资金流向
        })
    return result

def fetch_north_bound_top():
    """获取北向资金重仓行业流向"""
    url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=8&po=1&np=1&fltt=2&fid=f184&fs=b:MK0354&fields=f12,f14,f184"
    data = fetch_json(url)
    if not data or not data.get("data"):
        return None
    result = []
    for d in data["data"].get("diff", [])[:8]:
        result.append({
            "code": d.get("f12",""), "name": d.get("f14",""),
            "net_flow": d.get("f184", 0)
        })
    return result

# ============================================================
# 5. 市场情绪数据
# ============================================================
def fetch_market_breadth():
    """获取涨跌停家数、炸板率等（东方财富主 + 指数数据备用估算）"""
    # 涨停家数
    url_zt = f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:110+t:3&fields=f12"
    zt_data = fetch_json(url_zt)
    limit_up = zt_data["data"]["total"] if (zt_data and zt_data.get("data")) else None

    # 跌停家数
    url_dt = f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:110+t:4&fields=f12"
    dt_data = fetch_json(url_dt)
    limit_down = dt_data["data"]["total"] if (dt_data and dt_data.get("data")) else None

    # 上涨/下跌家数
    url_up = f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:110+t:1&fields=f12"
    up_data = fetch_json(url_up)
    up_count = up_data["data"]["total"] if (up_data and up_data.get("data")) else None

    url_down = f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1&fltt=2&fid=f3&fs=m:110+t:0&fields=f12"
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
                logger.info(f"  [FALLBACK] 涨跌家数估算: 涨{up_count}/跌{down_count} (指数变化{avg_change*100:.1f}%)")
        except Exception as e:
            logger.info(f"  [FALLBACK] 涨跌家数备用源失败: {e}")

    return {
        "limit_up": limit_up, "limit_down": limit_down,
        "up_count": up_count, "down_count": down_count,
        "total": (up_count or 0) + (down_count or 0)
    }

def fetch_total_volume():
    """获取全市场成交额（东方财富主 + Sina备用）"""
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f6"
    data = fetch_json(url)
    if data and data.get("data"):
        return data["data"].get("f6", 0)/1e8  # 亿元
    # v7.0: Sina备用（沪市+深市成交额之和）
    try:
        sh_url = "https://hq.sinajs.cn/list=s_sh000001"
        sh_resp = _simple_get(sh_url, timeout=10)
        if sh_resp:
            sh_text = sh_resp.read().decode("gbk", errors="replace") if hasattr(sh_resp, "read") else sh_resp
            sh_parts = sh_text.split(",")
            if len(sh_parts) > 10:
                sh_amount = float(sh_parts[10]) / 1e4  # 万→亿
            else:
                sh_amount = 0
        else:
            sh_amount = 0

        sz_url = "https://hq.sinajs.cn/list=s_sz399001"
        sz_resp = _simple_get(sz_url, timeout=10)
        if sz_resp:
            sz_text = sz_resp.read().decode("gbk", errors="replace") if hasattr(sz_resp, "read") else sz_resp
            sz_parts = sz_text.split(",")
            if len(sz_parts) > 10:
                sz_amount = float(sz_parts[10]) / 1e4
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
    return 0

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
        signals: {涨跌比, 量能, 涨停热度, ...}
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
    if total_volume > 20000:
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

    # 4. 北向资金 (权重: 15%)
    if north_flow_5d is not None:
        signals["北向5日(亿)"] = round(north_flow_5d, 1)
        if north_flow_5d > 100:
            adj = 15
        elif north_flow_5d > 30:
            adj = 8
        elif north_flow_5d > -30:
            adj = 0
        elif north_flow_5d > -100:
            adj = -8
        else:
            adj = -15
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

    return {"score": score, "level": level, "signals": signals}

# ============================================================
# 6. ETF 基本面与因子数据
# ============================================================
KEY_ETFS = {
    # === 用户重点关注 ===
    "562500": "机器人ETF华夏",       # 用户指定
    "512760": "芯片ETF国泰",         # 用户指定
    "518850": "黄金ETF华夏",         # 用户持仓
    "159326": "电网设备ETF华夏",     # 用户指定
    "159227": "航空航天ETF华夏",     # 用户指定（之前交易过）
    "515070": "人工智能ETF华夏",     # 用户指定
    "159183": "新能源车ETF招商",     # 用户持仓
    "159659": "纳斯达克100ETF招商",  # 用户指定

    # === 宽基指数 ===
    "510300": "沪深300ETF", "510500": "中证500ETF", "159915": "创业板ETF",
    "588000": "科创50ETF", "512100": "中证1000ETF",

    # === 行业/主题 ===
    "512880": "证券ETF", "159995": "芯片ETF华夏", "588200": "科创芯片ETF",
    "159819": "AIETF易方达", "159516": "半导体设备ETF",
    "512670": "国防ETF", "512810": "军工ETF",
    "512400": "有色ETF", "512170": "医疗ETF", "159992": "创新药ETF",
    "512890": "红利低波ETF", "159928": "消费ETF", "159865": "养殖ETF",
    "159611": "电力ETF", "512200": "房地产ETF", "159869": "游戏ETF",
    "159870": "化工ETF", "516020": "化工ETF华宝",
    "513100": "纳指ETF国泰", "513500": "标普500ETF",

    # === 补充 ===
    "159320": "电网设备ETF广发", "560390": "电网设备ETF易方达",
}

# 用户重点关注列表（优先分析）
USER_WATCHLIST = [
    "562500",   # 机器人ETF华夏
    "512760",   # 芯片ETF国泰
    "518850",   # 黄金ETF华夏（持仓）
    "159326",   # 电网设备ETF华夏
    "159227",   # 航空航天ETF华夏
    "515070",   # 人工智能ETF华夏
    "159183",   # 新能源车ETF招商（持仓）
    "159659",   # 纳斯达克100ETF招商
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
    """简易HTTP GET，先尝试requests，回退urllib"""
    session = _get_session()
    if session:
        try:
            r = session.get(url, timeout=timeout, verify=False)
            if r.status_code == 200:
                return r
        except Exception:
            pass
    # 回退urllib
    ctx = __import__('ssl').create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = __import__('ssl').CERT_NONE
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn/",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        return resp
    except Exception:
        return None

def _code_to_sina_prefix(code):
    """ETF/股票代码 -> 新浪行情前缀 (shXXXXXX / szXXXXXX)"""
    if code.startswith(("5", "6", "58")):
        return f"sh{code}"
    else:
        return f"sz{code}"

def _code_to_tencent_prefix(code):
    """ETF/股票代码 -> 腾讯行情前缀 (shXXXXXX / szXXXXXX)"""
    if code.startswith(("5", "6", "58")):
        return f"sh{code}"
    else:
        return f"sz{code}"

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
    """获取ETF日K线（东方财富主 + 腾讯备用）"""
    # 沪市ETF: 5xxxxx, 58xxxx; 深市ETF: 15xxxx, 16xxxx
    market = "1" if code.startswith(("5","58")) else "0"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59&klt=101&fqt=0&end={TODAY}&lmt={days}"
    data = fetch_json(url)
    if data and data.get("data"):
        result = []
        for k in data["data"]["klines"]:
            parts = k.split(",")
            result.append({
                "date": parts[0], "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]), "volume": float(parts[5]),
                "amount": float(parts[6])
            })
        return result
    # 备用: 腾讯API
    return _fetch_tencent_kline(code, days)

def fetch_stock_kline(code, days=250):
    """获取个股日K线（支持股票和ETF，东方财富主 + 腾讯备用）"""
    if code.startswith(("5","6","58")):
        market = "1"
    elif code.startswith(("0","3","2")):
        market = "0"
    else:
        market = "0"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1&end={TODAY}&lmt={days}"
    data = fetch_json(url)
    if data and data.get("data"):
        result = []
        for k in data["data"]["klines"]:
            parts = k.split(",")
            result.append({
                "date": parts[0], "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]), "volume": float(parts[5]),
                "amount": float(parts[6])
            })
        return result
    # 备用: 腾讯API
    return _fetch_tencent_kline(code, days)

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
        price = d.get("f43", 0)
        if price > 100:
            price = price / 100
        return {
            "price": price,
            "high": d.get("f44", 0)/100 if d.get("f44", 0) > 100 else d.get("f44", 0),
            "low": d.get("f45", 0)/100 if d.get("f45", 0) > 100 else d.get("f45", 0),
            "open": d.get("f46", 0)/100 if d.get("f46", 0) > 100 else d.get("f46", 0),
            "volume": d.get("f47", 0), "amount": d.get("f48", 0),
            "change_pct": d.get("f170", 0)/100 if abs(d.get("f170", 0)) > 10 else d.get("f170", 0),
            "prev_close": d.get("f60", 0)/100 if d.get("f60", 0) > 100 else d.get("f60", 0),
            "name": d.get("f58", ""), "code": d.get("f57", ""),
            "market_cap": d.get("f20", 0), "float_cap": d.get("f21", 0)
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
        # 解析IOPV（实时估值，f169；部分QDII ETF返回无效值如-61）
        raw_iopv = d.get("f169", 0)
        iopv = None
        if raw_iopv and raw_iopv > 0:
            iopv = raw_iopv / 1000 if raw_iopv > 100 else raw_iopv
        return {
            "price": d.get("f43", 0)/1000 if d.get("f43", 0) > 100 else d.get("f43", 0),
            "high": d.get("f44", 0)/1000 if d.get("f44", 0) > 100 else d.get("f44", 0),
            "low": d.get("f45", 0)/1000 if d.get("f45", 0) > 100 else d.get("f45", 0),
            "open": d.get("f46", 0)/1000 if d.get("f46", 0) > 100 else d.get("f46", 0),
            "volume": d.get("f47", 0), "amount": d.get("f48", 0),
            "change_pct": d.get("f170", 0)/100 if abs(d.get("f170", 0)) > 1 else d.get("f170", 0),
            "prev_close": d.get("f60", 0)/1000 if d.get("f60", 0) > 100 else d.get("f60", 0),
            "name": d.get("f58", ""), "code": d.get("f57", ""),
            "iopv": iopv  # 实时估值（QDII ETF可能为None）
        }
    # 备用: 新浪API
    return _fetch_sina_realtime(code)

# ============================================================
# 6b. ETF净值与溢价率（新增 — 修正量化模型对QDII溢价的盲区）
# ============================================================

# QDII ETF列表（跨境ETF，存在溢价风险）
# 以15、16、51、52、56开头且跟踪海外指数的ETF
QDII_ETF_CODES = {
    "159659", "513100", "513500",  # 纳斯达克ETF招商、纳指ETF国泰、标普500ETF
    "159660", "513050", "513060",  # 纳指相关
    "513520", "513880", "513220",  # 日经、恒生科技等
}

def fetch_etf_fund_nav(code):
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
            "nav_date": TODAY,
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
def calc_fear_index():
    """计算恐慌指数：今日下跌家数 / 总家数"""
    breadth = fetch_market_breadth()
    if breadth:
        total = breadth.get("total") or 0
        down = breadth.get("down_count") or 0
        if total > 0:
            return down / total * 100
    return 50


def _estimate_margin_from_index(indices):
    """
    v7.1 CI fallback: 从沪深300指数近5日趋势估算融资余额变化
    当东方财富API不可用时（如GitHub Actions CI环境），
    用指数涨跌粗略推断融资趋势：
    - 指数涨>2% → 融资大概率增加（投资者加杠杆追涨）
    - 指数跌>2% → 融资大概率减少（去杠杆/止损）
    - 否则 → 融资持平

    返回: {"balance": 估计值(亿), "change": 估计变化(亿)} 或 None
    """
    if "000300" not in indices:
        return None

    data = indices["000300"].get("data", [])
    if len(data) < 6:
        return None

    closes = [d["close"] for d in data]
    current = closes[-1]
    avg_5d_ago = sum(closes[-6:-1]) / 5

    pct_change = (current - avg_5d_ago) / avg_5d_ago * 100

    # A股融资余额通常在1.4-1.6万亿之间
    base_balance = 15000  # 亿
    if pct_change > 2.0:
        change = base_balance * 0.008  # 约+120亿
    elif pct_change > 1.0:
        change = base_balance * 0.004  # 约+60亿
    elif pct_change < -2.0:
        change = -base_balance * 0.008  # 约-120亿
    elif pct_change < -1.0:
        change = -base_balance * 0.004  # 约-60亿
    else:
        change = 0

    return {"balance": base_balance, "change": round(change, 1)}


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

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": TODAY
    }

    # 指数数据
    logger.info("  -> 指数日线...")
    result["indices"] = get_all_index_data(60)

    # 宏观数据
    logger.info("  -> 宏观数据...")
    result["bond_yield"] = fetch_bond_yield()
    result["shibor"] = fetch_shibor()

    # 资金数据
    logger.info("  -> 资金流向...")
    result["north_bound"] = fetch_north_bound_flow(10)
    result["fund_flow"] = fetch_market_fund_flow()
    margin_data = fetch_margin_balance()
    if not margin_data or margin_data.get("balance", 0) == 0:
        # v7.1 CI fallback: 从指数趋势估算融资变化
        margin_data = _estimate_margin_from_index(result.get("indices", {}))
        if margin_data:
            logger.info("  [FALLBACK] 融资余额(指数估算): balance=%.0f亿 change=%+.1f亿",
                       margin_data.get("balance", 0), margin_data.get("change", 0))
    result["margin"] = margin_data

    # 情绪数据
    logger.info("  -> 市场情绪...")
    result["breadth"] = fetch_market_breadth()
    result["total_volume"] = fetch_total_volume()
    result["fear_index"] = calc_fear_index()

    # 机构资金追踪（新增）
    logger.info("  -> 机构资金追踪...")
    result["sector_flow"] = fetch_sector_fund_flow()
    result["dragon_tiger"] = fetch_dragon_tiger()
    result["etf_flow"] = fetch_etf_flow_top()
    result["north_top"] = fetch_north_bound_top()

    # ETF数据 - 逐只拉取
    all_etf_codes = list(dict.fromkeys((etf_codes or []) + USER_WATCHLIST))
    logger.info(f"  -> ETF数据 ({len(all_etf_codes)}只, {'逐只' if sequential else '快速'}模式)...")
    etf_data = {}
    fail_count = 0

    for i, code in enumerate(all_etf_codes):
        if sequential and i > 0:
            time.sleep(1.2)  # 逐只间隔1.2秒，防盘中API拥堵

        name = KEY_ETFS.get(code, code)

        # 最多重试3次
        kline, realtime = None, None
        for attempt in range(3):
            kline = fetch_etf_kline(code, 250) if not kline else kline
            realtime = fetch_etf_realtime(code) if not realtime else realtime
            if kline and realtime:
                break
            if attempt < 2:
                time.sleep(1.0)

        if kline:
            etf_data[code] = {"name": name, "kline": kline, "realtime": realtime}
            if sequential:
                logger.info(f"    [{i+1}/{len(all_etf_codes)}] {code} {name} - OK ({len(kline)}d)")
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
            for attempt in range(3):
                kline = fetch_stock_kline(code, 250) if not kline else kline
                realtime = fetch_stock_realtime(code) if not realtime else realtime
                if kline and realtime:
                    break
                if attempt < 2:
                    time.sleep(1.0)

            if kline:
                stock_data[code] = {"name": name, "kline": kline, "realtime": realtime}
                logger.info(f"    [{i+1}/{len(stock_codes_list)}] {code} {name} - OK ({len(kline)}d)")
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
    with open(os.path.join(SCRIPT_DIR, f"data_{TODAY}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[DATA ENGINE] 数据已保存到 data_{TODAY}.json")
