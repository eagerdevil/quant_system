#!/usr/bin/env python
"""
量化系统 核心引擎 v2.0
=====================
模块2: 数据预处理（去极值+标准化+中性化）
模块3: 多因子模型（动量/波动/基本面/资金/情绪）
模块4: 大盘择时与仓位管理
模块5: 交易决策生成

v2.0 新增:
- 因子权重从 factor_weights.json 自动加载
- compute_indicators() 和 score_factors() 导出供 optimizer 使用
- 每周日 optimizer.py 自动更新权重 → 周一自动生效
"""
import json, math, sys, os
from datetime import datetime

TODAY = datetime.now().strftime("%Y%m%d")

# ============================================================
# 因子权重系统（v2.0 新增）
# ============================================================
FACTOR_MAX = {
    "F1_趋势强度": 10, "F2_动量": 8, "F3_反转": 8,
    "F4_RSI": 8, "F5_均线偏离": 6, "F6_低波动": 6,
    "F7_成交量": 6, "F8_回调": 10, "F9_Sortino": 6,
    "F10_MaxDD": 6, "F11_布林带": 6, "F12_多周期": 6,
    "F13_均线排列": 4, "F14_长期": 4, "F15_夏普": 6,
    "F16_量价关系": 8  # v3.0 新增：量价背离检测
}
FACTOR_NAMES = list(FACTOR_MAX.keys())
DEFAULT_WEIGHTS = {k: 1.0 for k in FACTOR_NAMES}

def _load_weights():
    """加载因子权重配置，失败时返回默认等权"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "factor_weights.json")
    try:
        with open(weights_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        w = config.get("factor_weights", DEFAULT_WEIGHTS)
        # 确保所有因子都有权重
        for k in FACTOR_NAMES:
            if k not in w:
                w[k] = 1.0
        return w, config
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return dict(DEFAULT_WEIGHTS), {"meta": {"version": 0, "ic_score": 0}}

CURRENT_WEIGHTS, WEIGHT_CONFIG = _load_weights()

# ============================================================
# 模块2: 数据预处理
# ============================================================
def mad_outlier_filter(series, n=5.0):
    """MAD法去极值：5倍绝对中位差"""
    if len(series) < 3: return series
    sorted_s = sorted(series)
    median = sorted_s[len(sorted_s)//2]
    abs_dev = sorted(abs(x - median) for x in series)
    mad = abs_dev[len(abs_dev)//2] * 1.4826  # 一致估计量
    if mad == 0: return series
    upper = median + n * mad
    lower = median - n * mad
    return [min(max(x, lower), upper) for x in series]

def zscore_normalize(series):
    """Z-score标准化"""
    if len(series) < 2: return [0]*len(series)
    mean = sum(series)/len(series)
    std = math.sqrt(sum((x-mean)**2 for x in series)/(len(series)-1))
    if std == 0: return [0]*len(series)
    return [(x-mean)/std for x in series]

# ============================================================
# 模块3: 多因子计算 — 指标函数
# ============================================================
def sma(data, n):
    if len(data) < n: return data[-1] if data else 0
    return sum(data[-n:])/n

def ema(data, n):
    if len(data) < n: return data[-1] if data else 0
    alpha = 2/(n+1)
    val = data[0]
    for v in data[1:]: val = alpha*v + (1-alpha)*val
    return val

def rsi(closes, n=14):
    if len(closes) < n+1: return 50
    gains, losses = [], []
    for i in range(-n, 0):
        d = closes[i] - closes[i-1]
        gains.append(d if d>0 else 0)
        losses.append(-d if d<0 else 0)
    avg_gain = sum(gains)/n
    avg_loss = sum(losses)/n
    if avg_loss == 0: return 100
    return 100 - 100/(1 + avg_gain/avg_loss)

def macd_calc(closes):
    """返回 (DIF, DEA, Hist) 当前值"""
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    dif = e12 - e26
    # 简化DEA
    if len(closes) >= 35:
        difs = []
        for i in range(26, len(closes)):
            e12_i = ema(closes[:i+1], 12)
            e26_i = ema(closes[:i+1], 26)
            difs.append(e12_i - e26_i)
        dea = ema(difs, 9) if difs else dif
    else:
        dea = dif
    return dif, dea, (dif-dea)*2

def max_drawdown(closes):
    peak, md = closes[0], 0
    for c in closes:
        if c > peak: peak = c
        dd = (peak-c)/peak
        if dd > md: md = dd
    return md

def volatility(closes, n=20):
    if len(closes) < n+1: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(-n, 0)]
    avg = sum(rets)/n
    return math.sqrt(sum((r-avg)**2 for r in rets)/n) * math.sqrt(252)

def sharpe(closes, rf=0.025):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    std_ret = math.sqrt(sum((r-avg_ret)**2 for r in rets)/(len(rets)-1)) if len(rets)>1 else 0.01
    if std_ret == 0: return 0
    return (avg_ret - rf/252)/std_ret * math.sqrt(252)

def sortino(closes, rf=0.025):
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    avg_ret = sum(rets)/len(rets)
    down_rets = [r for r in rets if r < 0]
    if len(down_rets) < 2: return 3.0 if avg_ret>0 else -3.0
    down_std = math.sqrt(sum(r**2 for r in down_rets)/(len(down_rets)-1))
    if down_std == 0: return 0
    return (avg_ret - rf/252)/down_std * math.sqrt(252)

def consecutive_up(closes):
    cnt = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] > closes[i-1]: cnt += 1
        else: break
    return cnt

def consecutive_down(closes):
    cnt = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] < closes[i-1]: cnt += 1
        else: break
    return cnt

def ret_n(closes, n):
    if len(closes) < n+1: return 0
    return (closes[-1]/closes[-n-1] - 1)*100

def vol_ratio(volumes, n=5):
    if len(volumes) < n*2: return 1.0
    recent = sum(volumes[-n:])/n
    prior = sum(volumes[-n*2:-n])/n
    return recent/prior if prior>0 else 1

def pma(closes, n):
    ma = sma(closes, n)
    if ma == 0: return 0
    return (closes[-1]/ma - 1)*100

def ma_alignment(closes):
    """MA多头排列程度：MA5>MA10>MA20>MA60"""
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60) if len(closes)>=60 else sma(closes, len(closes))
    score = 0
    if ma5 > ma10: score += 1
    if ma10 > ma20: score += 1
    if ma20 > ma60: score += 1
    return score

def bollinger_position(closes, n=20):
    """布林带位置 0-100"""
    if len(closes) < n: return 50
    window = closes[-n:]
    avg = sum(window)/n
    std = math.sqrt(sum((x-avg)**2 for x in window)/n)
    if std == 0: return 50
    upper, lower = avg + 2*std, avg - 2*std
    if upper == lower: return 50
    return (closes[-1] - lower)/(upper - lower)*100

def adx(closes, highs, lows, n=14):
    """
    平均趋向指数 (Average Directional Index)
    返回: adx值 (0-100, >25趋势市, <20震荡市)
    用于市场状态分类
    """
    if len(closes) < n + 1:
        return 20  # 数据不足默认震荡
    # True Range
    tr_list = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr_list.append(max(hl, hc, lc))
    atr_val = sum(tr_list[-n:]) / n
    if atr_val == 0:
        return 20

    # +DM / -DM
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        if up > down and up > 0:
            plus_dm.append(up)
        else:
            plus_dm.append(0)
        if down > up and down > 0:
            minus_dm.append(down)
        else:
            minus_dm.append(0)

    # 平滑
    def _smooth_adx(series, n):
        if len(series) < n:
            return sum(series) / len(series) if series else 0
        smoothed = sum(series[:n])
        for i in range(n, len(series)):
            smoothed = smoothed - smoothed / n + series[i]
        return smoothed / n

    atr_smooth = _smooth_adx(tr_list, n)
    plus_di = (_smooth_adx(plus_dm, n) / atr_smooth * 100) if atr_smooth > 0 else 0
    minus_di = (_smooth_adx(minus_dm, n) / atr_smooth * 100) if atr_smooth > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

    # ADX = smoothed DX
    if len(closes) >= n * 2:
        dx_values = []
        for i in range(n, len(closes)):
            # 简化：用最后2n天近似
            pass
        return _smooth_adx([dx], min(n, 5))
    return dx

def bollinger_bandwidth(closes, n=20):
    """布林带宽度 — 衡量波动率压缩/扩张"""
    if len(closes) < n:
        return 0.05  # 默认中等宽度
    window = closes[-n:]
    avg = sum(window) / n
    std = math.sqrt(sum((x-avg)**2 for x in window) / n)
    if avg == 0:
        return 0
    return (4 * std) / avg  # (upper - lower) / middle

def calc_choppiness(closes, highs, lows, n=14):
    """
    震荡指数 (Choppiness Index)
    值越高越震荡 (>61.8 = 震荡), 越低越趋势 (<38.2 = 趋势)
    """
    if len(closes) < n:
        return 50
    window_high = max(highs[-n:])
    window_low = min(lows[-n:])
    tr_sum = sum(
        max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        for i in range(-n+1, 0)
    ) + (max(highs[-1]-lows[-1], abs(highs[-1]-closes[-2]), abs(lows[-1]-closes[-2])) if len(closes) > n else 0)
    if tr_sum == 0 or window_high == window_low:
        return 50
    chop = 100 * math.log10(tr_sum / (window_high - window_low)) / math.log10(n)
    return max(0, min(100, chop))


def detect_volume_price_divergence(closes, volumes):
    """
    量价背离检测 (v3.0 新增)

    检测四种关键背离模式：
    1. 看跌背离: 价格创近期新高 + 成交量递减 → 上涨动力不足
    2. 看涨背离: 价格创近期新低 + 成交量递增 → 下跌衰竭
    3. 缩量反弹: 连续上涨 + 量能萎缩 → 虚假反弹
    4. 放量下跌: 价格下跌 + 量暴增(非极端RSI) → 恐慌抛售

    返回: {
        "divergence_type": "bearish"|"bullish"|"weak_rally"|"panic_sell"|None,
        "strength": 0-10 (越高越强),
        "description": str
    }
    """
    if len(closes) < 20 or len(volumes) < 20:
        return {"divergence_type": None, "strength": 0, "description": ""}

    # 分两段：前10日 vs 近10日
    mid = len(closes) // 2
    closes_recent = closes[-mid:]
    closes_prior = closes[:mid]
    volumes_recent = volumes[-mid:]
    volumes_prior = volumes[:mid]

    avg_vol_recent = sum(volumes_recent) / len(volumes_recent)
    avg_vol_prior = sum(volumes_prior) / len(volumes_prior)
    if avg_vol_prior == 0:
        return {"divergence_type": None, "strength": 0, "description": ""}

    vol_change = avg_vol_recent / avg_vol_prior

    max_close_recent = max(closes_recent)
    max_close_prior = max(closes_prior)
    min_close_recent = min(closes_recent)
    min_close_prior = min(closes_prior)

    price_up = closes_recent[-1] > closes_prior[-1]
    price_down = closes_recent[-1] < closes_prior[-1]

    # 1. 看跌背离: 价格创新高 + 量缩 (>10%)
    if max_close_recent > max_close_prior and vol_change < 0.90:
        strength = min(10, round((1 - vol_change) * 10 + (max_close_recent/max_close_prior - 1) * 50))
        return {
            "divergence_type": "bearish",
            "strength": strength,
            "description": f"看跌背离: 价格创新高(+{(max_close_recent/max_close_prior-1)*100:.1f}%)但量缩{(1-vol_change)*100:.0f}%",
            "vol_change": round(vol_change, 2),
            "price_action": "新高量缩"
        }

    # 2. 看涨背离: 价格创新低 + 量增 (>10%)
    if min_close_recent < min_close_prior and vol_change > 1.10:
        strength = min(10, round((vol_change - 1) * 8 + (1 - min_close_recent/min_close_prior) * 50))
        return {
            "divergence_type": "bullish",
            "strength": strength,
            "description": f"看涨背离: 价格创新低(-{(1-min_close_recent/min_close_prior)*100:.1f}%)但量增{(vol_change-1)*100:.0f}%",
            "vol_change": round(vol_change, 2),
            "price_action": "新低量增"
        }

    # 3. 缩量反弹: 连涨3天+ + 近3日量<前10日均量80%
    cons_up = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] > closes[i-1]:
            cons_up += 1
        else:
            break
    recent_3vol = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else avg_vol_recent
    if cons_up >= 3 and recent_3vol < avg_vol_prior * 0.80:
        strength = min(10, cons_up * 2 + round((1 - recent_3vol/avg_vol_prior) * 10))
        return {
            "divergence_type": "weak_rally",
            "strength": strength,
            "description": f"缩量反弹: 连涨{cons_up}天但近3日量仅为前均{(recent_3vol/avg_vol_prior)*100:.0f}%",
            "vol_change": round(recent_3vol/avg_vol_prior, 2),
            "price_action": "缩量连涨"
        }

    # 4. 放量下跌: 价格下跌 + 量暴增>1.3倍
    rsi_now = rsi(closes, 14)
    if price_down and vol_change > 1.30 and rsi_now > 30:
        strength = min(10, round((vol_change - 1) * 7 + 3))
        return {
            "divergence_type": "panic_sell",
            "strength": strength,
            "description": f"放量下跌: 价格下跌+量暴增至{(vol_change)*100:.0f}%，恐慌抛售中(RSI={rsi_now:.0f})",
            "vol_change": round(vol_change, 2),
            "price_action": "放量下跌"
        }

    return {"divergence_type": None, "strength": 0, "description": ""}

# ============================================================
# 市场状态分类 (v3.0 新增)
# ============================================================
MARKET_REGIME = {
    "TREND_UP": "趋势上涨",
    "CHOPPY": "震荡整理",
    "TREND_DOWN": "趋势下跌",
    "CRISIS": "危机模式"
}

REGIME_STRATEGY = {
    "TREND_UP": {
        "base_position": (0.60, 1.00),  # 仓位范围
        "stop_loss": -0.08,            # 止损
        "buy_grade_min": "B_买入",     # 最低买入等级
        "description": "趋势向好，正常操作"
    },
    "CHOPPY": {
        "base_position": (0.20, 0.40),
        "stop_loss": -0.05,            # 收紧止损
        "buy_grade_min": "A_强烈买入",
        "description": "方向不明，缩小头寸+收紧止损"
    },
    "TREND_DOWN": {
        "base_position": (0.05, 0.20),
        "stop_loss": -0.05,            # 硬止损
        "buy_grade_min": None,         # 禁止买入
        "description": "趋势向下，防御为主"
    },
    "CRISIS": {
        "base_position": (0.0, 0.05),
        "stop_loss": -0.05,
        "buy_grade_min": None,
        "description": "危机模式，现金为王"
    }
}

def classify_market_regime(index_closes, index_highs=None, index_lows=None,
                           bond_yield=None, vix=None):
    """
    综合判断市场状态
    输入: 沪深300的日线数据
    输出: {regime, confidence, signals, description}
    """
    if len(index_closes) < 60:
        return {"regime": "CHOPPY", "confidence": 0.3,
                "signals": {}, "description": "数据不足，默认震荡"}

    # 用收盘价的复制简化highs/lows
    if index_highs is None:
        index_highs = [c * 1.005 for c in index_closes]  # 近似
    if index_lows is None:
        index_lows = [c * 0.995 for c in index_closes]

    signals = {}

    # Signal 1: ADX趋势强度
    adx_val = adx(index_closes, index_highs, index_lows, 14)
    signals["adx"] = round(adx_val, 1)
    is_trending = adx_val > 22
    is_strong_trend = adx_val > 28

    # Signal 2: 均线排列
    ma_align = ma_alignment(index_closes)
    signals["ma_alignment"] = ma_align
    ma_bullish = ma_align >= 2  # MA5>MA10>MA20 or MA10>MA20>MA60
    ma_bearish = ma_align == 0  # 全空头

    # Signal 3: 布林带宽度（波动率状态）
    bbw = bollinger_bandwidth(index_closes, 20)
    signals["bb_width"] = round(bbw, 4)
    # 布林带宽<3% = 低波动（可能酝酿突破），>8% = 高波动
    high_vol = bbw > 0.08

    # Signal 4: 震荡指数
    chop = calc_choppiness(index_closes, index_highs, index_lows, 14)
    signals["choppiness"] = round(chop, 1)
    is_choppy = chop > 55

    # Signal 5: 近期收益率
    r5 = ret_n(index_closes, 5)
    r20 = ret_n(index_closes, 20)
    signals["r5d"] = round(r5, 1)
    signals["r20d"] = round(r20, 1)
    sharp_decline = r5 < -6 or r20 < -12

    # Signal 6: 价格位置 vs MA60
    ma60 = sma(index_closes, 60)
    below_ma60 = index_closes[-1] < ma60 * 0.95  # 低于MA60超过5%
    signals["below_ma60_pct"] = round((index_closes[-1]/ma60 - 1)*100, 1) if ma60 > 0 else 0

    # === 状态判定 ===
    regime_signals = []
    confidence = 0.5

    # 危机判定：急速下跌 + 跌破关键均线
    if sharp_decline and (below_ma60 or r20 < -15):
        regime = "CRISIS"
        confidence = 0.85
        regime_signals.append(f"急速下跌(5日{r5:.1f}%, 20日{r20:.1f}%)")
        if below_ma60:
            regime_signals.append("跌破MA60超过5%")
    # 趋势下跌：空头排列 + 有趋势
    elif ma_bearish and is_trending:
        regime = "TREND_DOWN"
        confidence = 0.80
        regime_signals.append("均线空头排列")
        regime_signals.append(f"ADX={adx_val:.1f}(趋势明确)")
    # 震荡：无明确方向
    elif is_choppy or not is_trending:
        regime = "CHOPPY"
        if is_choppy and not is_trending:
            confidence = 0.75
            regime_signals.append(f"震荡指数={chop:.1f}(>55震荡)")
            regime_signals.append(f"ADX={adx_val:.1f}(<22无趋势)")
        else:
            confidence = 0.55
            regime_signals.append("方向不明确")
    # 趋势上涨：多头排列 + 有趋势
    elif ma_bullish and is_trending:
        regime = "TREND_UP"
        confidence = 0.80
        regime_signals.append("均线多头排列")
        regime_signals.append(f"ADX={adx_val:.1f}(趋势明确)")
    else:
        regime = "CHOPPY"
        confidence = 0.40
        regime_signals.append("综合判定为震荡")

    # 高波动叠加
    if high_vol:
        regime_signals.append(f"高波动(布林带宽={bbw*100:.1f}%)")
        if regime == "TREND_DOWN":
            regime = "CRISIS"
            confidence = min(1.0, confidence + 0.1)
            regime_signals.append("高波动+下跌趋势 → 升级为危机模式")

    strategy = REGIME_STRATEGY.get(regime, REGIME_STRATEGY["CHOPPY"])

    return {
        "regime": regime,
        "regime_name": MARKET_REGIME.get(regime, regime),
        "confidence": round(confidence, 2),
        "signals": signals,
        "regime_signals": regime_signals,
        "strategy": strategy,
        "description": strategy["description"]
    }
# 模块3b: 因子计算（导出供 optimizer 使用）
# ============================================================
def compute_indicators(closes, highs, lows, volumes):
    """
    计算所有原始指标。
    输入: ETF的价格序列（按时间升序）
    输出: 指标字典，可直接传给 score_factors()

    此函数被 optimizer.py 导入用于回测。
    """
    cur = closes[-1]

    rsi_now = rsi(closes, 14)
    dif, dea, hist = macd_calc(closes)
    vol = volatility(closes, 20)
    maxdd = max_drawdown(closes)
    shp = sharpe(closes)
    srt = sortino(closes)
    cons_up = consecutive_up(closes)
    cons_down = consecutive_down(closes)
    ma_align = ma_alignment(closes)
    bb_pos = bollinger_position(closes, 20)
    atr_val = sum(abs(closes[i]-closes[i-1]) for i in range(-14, 0))/14 if len(closes)>=15 else 0
    atr_pct = atr_val/cur*100 if cur>0 else 0

    r5 = ret_n(closes, 5)
    r10 = ret_n(closes, 10)
    r20 = ret_n(closes, 20)
    r60 = ret_n(closes, 60)
    r120 = ret_n(closes, 120) if len(closes)>=121 else 0

    v_ratio = vol_ratio(volumes, 5)
    v_ratio_10 = vol_ratio(volumes, 10)

    pm5 = pma(closes, 5)
    pm10 = pma(closes, 10)
    pm20 = pma(closes, 20)
    pm60 = pma(closes, 60) if len(closes)>=60 else pm20

    return {
        "rsi": rsi_now, "dif": dif, "dea": dea, "hist": hist,
        "volatility": vol, "max_drawdown": maxdd, "sharpe": shp,
        "sortino": srt, "consecutive_up": cons_up,
        "consecutive_down": cons_down, "ma_alignment": ma_align,
        "bb_position": bb_pos, "atr_pct": atr_pct,
        "r5d": r5, "r10d": r10, "r20d": r20, "r60d": r60, "r120d": r120,
        "vol_ratio_5d": v_ratio, "vol_ratio_10d": v_ratio_10,
        "pma5": pm5, "pma10": pm10, "pma20": pm20, "pma60": pm60,
        "price": cur,
        # v3.0 新增：量价背离
        "volume_divergence": detect_volume_price_divergence(closes, volumes)
    }

def score_factors(indicators):
    """
    将原始指标转换为15个因子得分（0-10/0-8/0-6/0-4）。
    输入: compute_indicators() 的输出
    输出: {F1_趋势强度: 5, F2_动量: 8, ...}

    此函数被 optimizer.py 导入用于回测。
    """
    factors = {}
    ind = indicators

    # F1: 趋势强度(0-10)
    f1 = 5
    if ind["dif"] > 0 and ind["hist"] > 0: f1 = 10
    elif ind["dif"] > 0: f1 = 7
    elif ind["dif"] > -0.01 and ind["hist"] > 0: f1 = 6
    elif ind["dif"] > -0.01: f1 = 4
    else: f1 = 2
    if ind["ma_alignment"] >= 2: f1 = min(10, f1+1)
    factors['F1_趋势强度'] = f1

    # F2: 动量(0-8)
    r20 = ind["r20d"]
    if 0 <= r20 <= 10: f2 = 8
    elif -3 <= r20 < 0: f2 = 6
    elif 10 < r20 <= 20: f2 = 5
    elif r20 < -10: f2 = 3
    else: f2 = 2
    factors['F2_动量'] = f2

    # F3: 反转信号(0-8)
    r5 = ind["r5d"]
    if r5 < -5: f3 = 8
    elif -5 <= r5 < -2: f3 = 6
    elif r5 < 0: f3 = 5
    elif 0 <= r5 <= 3: f3 = 4
    elif 3 < r5 <= 8: f3 = 3
    else: f3 = 1
    factors['F3_反转'] = f3

    # F4: RSI位置(0-8)
    rsi_now = ind["rsi"]
    if 40 <= rsi_now <= 50: f4 = 8
    elif 35 <= rsi_now < 40: f4 = 7
    elif 50 < rsi_now <= 58: f4 = 6
    elif 30 <= rsi_now < 35: f4 = 5
    elif 58 < rsi_now <= 65: f4 = 4
    elif 65 < rsi_now <= 72: f4 = 2
    else: f4 = 1
    factors['F4_RSI'] = f4

    # F5: 均线偏离(0-6)
    pm5 = ind["pma5"]
    if -3 <= pm5 <= 3: f5 = 6
    elif -5 <= pm5 < -3: f5 = 5
    elif 3 < pm5 <= 8: f5 = 4
    elif -8 <= pm5 < -5: f5 = 3
    else: f5 = 2
    factors['F5_均线偏离'] = f5

    # F6: 低波动(0-6)
    vol_pct = ind["volatility"] * 100
    if 15 <= vol_pct <= 30: f6 = 6
    elif 10 <= vol_pct < 15: f6 = 5
    elif 30 < vol_pct <= 40: f6 = 4
    elif vol_pct < 10: f6 = 3
    else: f6 = 2
    factors['F6_低波动'] = f6

    # F7: 成交量健康(0-6)
    v_ratio = ind["vol_ratio_5d"]
    if 0.85 <= v_ratio <= 1.20: f7 = 6
    elif 0.70 <= v_ratio <= 1.40: f7 = 4
    else: f7 = 2
    factors['F7_成交量'] = f7

    # F8: 回调质量(0-10)
    cons_down = ind["consecutive_down"]
    cons_up = ind["consecutive_up"]
    if cons_down >= 3: f8 = 10
    elif cons_down >= 2: f8 = 8
    elif cons_down == 1: f8 = 7
    elif cons_up == 0: f8 = 7
    elif cons_up == 1: f8 = 6
    elif cons_up == 2: f8 = 4
    else: f8 = 2
    factors['F8_回调'] = f8

    # F9: Sortino(0-6)
    srt = ind["sortino"]
    if srt > 1.5: f9 = 6
    elif srt > 0.8: f9 = 5
    elif srt > 0.3: f9 = 4
    elif srt > -0.3: f9 = 3
    else: f9 = 2
    factors['F9_Sortino'] = f9

    # F10: 最大回撤(0-6)
    maxdd = ind["max_drawdown"]
    if maxdd < 0.10: f10 = 6
    elif maxdd < 0.18: f10 = 5
    elif maxdd < 0.25: f10 = 4
    elif maxdd < 0.35: f10 = 3
    else: f10 = 2
    factors['F10_MaxDD'] = f10

    # F11: 布林带位置(0-6)
    bb_pos = ind["bb_position"]
    if 15 <= bb_pos <= 55: f11 = 6
    elif 5 <= bb_pos < 15: f11 = 5
    elif 55 < bb_pos <= 75: f11 = 4
    elif 75 < bb_pos <= 90: f11 = 2
    else: f11 = 1
    factors['F11_布林带'] = f11

    # F12: 多周期收益(0-6)
    r60 = ind["r60d"]
    if r5 > 0 and r20 > 0 and r60 > 0: f12 = 6
    elif r20 > 0 and r60 > 0: f12 = 5
    elif r60 > 0: f12 = 4
    elif r20 > 0: f12 = 3
    else: f12 = 2
    factors['F12_多周期'] = f12

    # F13: 均线排列(0-4)
    factors['F13_均线排列'] = ind["ma_alignment"]

    # F14: 长期收益(0-4)
    r120 = ind["r120d"]
    if r120 > 10: f14 = 4
    elif r120 > 0: f14 = 3
    elif r120 > -10: f14 = 2
    else: f14 = 1
    factors['F14_长期'] = f14

    # F15: 夏普比率(0-6)
    shp = ind["sharpe"]
    if shp > 1.5: f15 = 6
    elif shp > 0.8: f15 = 5
    elif shp > 0.3: f15 = 4
    elif shp > -0.3: f15 = 3
    else: f15 = 2
    factors['F15_夏普'] = f15

    # F16: 量价关系 (0-8) — v3.0 新增
    vol_div = ind.get("volume_divergence", {})
    div_type = vol_div.get("divergence_type")
    if div_type == "bullish":
        f16 = 8  # 看涨背离：下跌衰竭信号，看多
    elif div_type == "bearish":
        f16 = 2  # 看跌背离：上涨动力不足，看空
    elif div_type == "weak_rally":
        f16 = 3  # 缩量反弹：虚假反弹，谨慎
    elif div_type == "panic_sell":
        f16 = 4  # 放量下跌：恐慌中，但可能接近底部
    else:
        f16 = 5  # 无量价背离，中性
    factors['F16_量价关系'] = f16

    return factors

# ============================================================
# 综合因子评分（ETF级别）— v2.0 支持自适应权重
# ============================================================
def _apply_premium_penalty(technical_score, premium_pct):
    """
    溢价惩罚函数 — 修正量化模型对QDII ETF溢价的盲区

    设计逻辑:
    - 溢价<2%: 不惩罚（正常交易区间）
    - 溢价2-5%: 线性衰减，从100%→85%
    - 溢价5-8%: 加速衰减，从85%→65%
    - 溢价>8%: 严重惩罚，最低到50%

    参数:
        technical_score: 原始15因子技术评分 (0-100)
        premium_pct: 溢价率（正=溢价，None=数据缺失）
    返回:
        (adjusted_score, penalty_multiplier, warning)
    """
    if premium_pct is None:
        # QDII ETF但无溢价数据 → 标记警告但不调整分数
        return technical_score, 1.0, "QDII溢价数据缺失，无法评估溢价风险"

    if premium_pct < 2.0:
        return technical_score, 1.0, None

    # 分段惩罚系数
    if premium_pct <= 5.0:
        # 2-5%: 每1%溢价扣6%分数
        multiplier = 1.0 - (premium_pct - 2.0) * 0.06
    elif premium_pct <= 8.0:
        # 5-8%: 前段扣18% + 每1%额外扣7%
        multiplier = 0.82 - (premium_pct - 5.0) * 0.07
    else:
        # >8%: 前段扣39% + 每1%额外扣8%，下限0.48
        multiplier = max(0.48, 0.61 - (premium_pct - 8.0) * 0.08)

    multiplier = round(multiplier, 4)

    if premium_pct > 8:
        warning = f"🚨 溢价{premium_pct:.1f}%极度危险！市价远超净值，面临停牌+溢价回归双重风险"
    elif premium_pct > 5:
        warning = f"⚠️ 溢价{premium_pct:.1f}%偏高，买入即多付{premium_pct:.1f}%成本，需等溢价回落"
    elif premium_pct > 3:
        warning = f"⚡ 溢价{premium_pct:.1f}%，略高于安全线，关注溢价收敛趋势"
    else:
        warning = None

    adjusted = round(technical_score * multiplier)
    return adjusted, multiplier, warning


def score_etf_comprehensive(code, name, closes, highs, lows, volumes,
                             north_flow_5d=None, industry_return=None,
                             weights=None, premium_pct=None):
    """
    15因子综合评分系统 + 溢价惩罚
    参数:
        weights: dict {factor_name: weight}, None时使用factor_weights.json配置
        premium_pct: ETF溢价率（%），None表示数据缺失
    返回: {score, grade, factor_details, indicators, premium_info}
    """
    # 加载权重（参数 > 配置文件 > 默认等权）
    if weights is None:
        weights = CURRENT_WEIGHTS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 计算指标和因子
    indicators = compute_indicators(closes, highs, lows, volumes)
    factors = score_factors(indicators)

    # 加权汇总（归一化到0-100）
    weighted_sum = sum(factors[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
    max_weighted = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
    if max_weighted > 0:
        technical_score = round(weighted_sum / max_weighted * 100)
    else:
        technical_score = sum(factors.values())

    # === 溢价惩罚（v2.1 新增）===
    adjusted_score, premium_multiplier, premium_warning = _apply_premium_penalty(
        technical_score, premium_pct
    )
    # 最终评分 = 技术评分经溢价调整
    final_score = adjusted_score

    # Grade（基于调整后评分，使用可配置阈值）
    grade_thresholds = WEIGHT_CONFIG.get("grade_thresholds", {
        "A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42
    })
    if final_score >= grade_thresholds.get("A_强烈买入", 78):
        grade = "A_强烈买入"
    elif final_score >= grade_thresholds.get("B_买入", 65):
        grade = "B_买入"
    elif final_score >= grade_thresholds.get("C_观察", 55):
        grade = "C_观察"
    elif final_score >= grade_thresholds.get("D_谨慎", 42):
        grade = "D_谨慎"
    else:
        grade = "E_回避"

    return {
        "code": code, "name": name,
        "score": final_score,  # 溢价调整后的最终评分
        "technical_score": technical_score,  # 原始技术评分（不含溢价）
        "max_score": 100,
        "grade": grade,
        "price": round(indicators["price"], 4),
        # 溢价信息（v2.1新增）
        "premium_info": {
            "premium_pct": premium_pct,
            "penalty_multiplier": premium_multiplier,
            "warning": premium_warning,
            "score_penalty": technical_score - final_score
        },
        "indicators": {
            "rsi": round(indicators["rsi"], 1),
            "volatility_pct": round(indicators["volatility"]*100, 1),
            "sharpe": round(indicators["sharpe"], 2),
            "sortino": round(indicators["sortino"], 2),
            "max_dd_pct": round(indicators["max_drawdown"]*100, 1),
            "consecutive_up": indicators["consecutive_up"],
            "consecutive_down": indicators["consecutive_down"],
            "ma_alignment": indicators["ma_alignment"],
            "bb_position": round(indicators["bb_position"], 1),
            "atr_pct": round(indicators["atr_pct"], 2),
            "vol_ratio_5d": round(indicators["vol_ratio_5d"], 2),
            "macd_dif": round(indicators["dif"], 4),
            "macd_hist": round(indicators["hist"], 4)
        },
        "returns": {
            "r5d": round(indicators["r5d"], 1), "r10d": round(indicators["r10d"], 1),
            "r20d": round(indicators["r20d"], 1), "r60d": round(indicators["r60d"], 1),
            "r120d": round(indicators["r120d"], 1)
        },
        "vs_ma": {
            "pct_ma5": round(indicators["pma5"], 1), "pct_ma10": round(indicators["pma10"], 1),
            "pct_ma20": round(indicators["pma20"], 1), "pct_ma60": round(indicators["pma60"], 1)
        },
        "factors": factors,
        # v3.0 新增：量价背离详情
        "volume_divergence": indicators.get("volume_divergence", {})
    }

# ============================================================
# 模块4: 大盘择时引擎
# ============================================================
class MarketTiming:
    """市场择时信号"""

    def __init__(self, index_data, north_flow, total_vol, breadth, margin):
        """
        index_data: {code: {name, data: [{date,close,high,low,volume}...]}}
        north_flow: [{date, net_flow}]
        total_vol: float (亿元)
        breadth: {limit_up, limit_down, up_count, down_count}
        margin: {balance, change}
        """
        self.hs300 = None
        self.hs300_highs = None
        self.hs300_lows = None
        if "000300" in index_data:
            data = index_data["000300"]["data"]
            self.hs300 = [d["close"] for d in data]
            self.hs300_highs = [d.get("high", d["close"]*1.005) for d in data]
            self.hs300_lows = [d.get("low", d["close"]*0.995) for d in data]

        self.north_flow = north_flow or []
        self.total_vol = total_vol or 0
        self.breadth = breadth or {}
        self.margin = margin or {}

    def calc_signals(self):
        """计算6个择时信号，返回 {signal_name: bool}"""
        signals = {}

        # S1: 沪深300在20日均线上方
        if self.hs300 and len(self.hs300) >= 20:
            ma20 = sma(self.hs300, 20)
            signals['S1_HS300_above_MA20'] = self.hs300[-1] > ma20
        else:
            signals['S1_HS300_above_MA20'] = False

        # S2: 沪深300的60日均线向上
        if self.hs300 and len(self.hs300) >= 61:
            ma60_now = sma(self.hs300, 60)
            ma60_prev = sma(self.hs300[:-1], 60)
            signals['S2_HS300_MA60_up'] = ma60_now > ma60_prev
        else:
            signals['S2_HS300_MA60_up'] = False

        # S3: 北向资金近5日累计净流入为正
        nf_5d = sum(f.get("net_flow", 0) for f in self.north_flow[-5:]) if self.north_flow else 0
        signals['S3_NorthFlow_5d_positive'] = nf_5d > 0

        # S4: 全市场成交额大于20日均值
        # (简化：用3万亿作为A股活跃阈值)
        signals['S4_Volume_active'] = self.total_vol > 20000  # 2万亿

        # S5: 跌停家数小于20家（数据缺失时默认不通过）
        ld = self.breadth.get("limit_down")
        signals['S5_LimitDown_low'] = ld < 20 if ld is not None else False

        # S6: 融资余额5日斜率（用净买入方向判断）
        margin_change = self.margin.get("change", 0)
        signals['S6_Margin_increasing'] = margin_change > 0

        return signals, nf_5d

    def position_advice(self):
        """根据择时信号+市场状态计算建议仓位（v3.0 加入状态分类）"""
        signals, nf_5d = self.calc_signals()
        bull_count = sum(1 for v in signals.values() if v)

        # === v3.0: 市场状态分类 ===
        regime_result = None
        if self.hs300 and len(self.hs300) >= 60:
            regime_result = classify_market_regime(
                self.hs300, self.hs300_highs, self.hs300_lows
            )
        else:
            regime_result = {"regime": "CHOPPY", "confidence": 0.3,
                             "signals": {}, "regime_signals": ["数据不足"],
                             "strategy": REGIME_STRATEGY["CHOPPY"],
                             "description": "数据不足，默认震荡"}

        # === 仓位映射: 结合传统信号+状态分类 ===
        # 基础仓位（传统6信号）
        if bull_count >= 5: base_position = 1.0
        elif bull_count >= 4: base_position = 0.85
        elif bull_count >= 3: base_position = 0.65
        elif bull_count >= 2: base_position = 0.40
        elif bull_count >= 1: base_position = 0.20
        else: base_position = 0.05

        # 状态覆盖（v3.0新增）：状态分类的仓位上限
        regime_strategy = regime_result.get("strategy", REGIME_STRATEGY["CHOPPY"])
        pos_range = regime_strategy.get("base_position", (0.20, 0.40))
        regime_stop_pct = regime_strategy.get("stop_loss", -0.08)
        buy_grade_min = regime_strategy.get("buy_grade_min", "B_买入")

        # 状态仓位上限约束
        base_position = min(base_position, pos_range[1])
        base_position = max(base_position, pos_range[0])

        # 强制限制：HS300低于MA60且MA60向下 → 仓位上限30%
        hs300_below_ma60 = not signals.get('S1_HS300_above_MA20', True)
        ma60_down = not signals.get('S2_HS300_MA60_up', True)
        force_cap = hs300_below_ma60 and ma60_down

        if force_cap:
            base_position = min(base_position, 0.30)

        return {
            "bull_signals": bull_count,
            "total_signals": len(signals),
            "signal_detail": signals,
            "base_position": round(base_position, 2),
            "force_capped": force_cap,
            "north_flow_5d": round(nf_5d, 1),
            "advice": self._position_text(base_position, bull_count, force_cap),
            # v3.0 新增
            "regime": regime_result["regime"],
            "regime_name": MARKET_REGIME.get(regime_result["regime"], regime_result["regime"]),
            "regime_confidence": regime_result.get("confidence", 0),
            "regime_signals": regime_result.get("regime_signals", []),
            "regime_stop_loss": regime_stop_pct,
            "regime_buy_grade_min": buy_grade_min,
            "regime_description": regime_result.get("description", "")
        }

    def _position_text(self, pos, bulls, capped):
        if pos >= 0.85: return f"进攻(仓位{pos*100:.0f}%) — {bulls}个看多信号，市场强势"
        elif pos >= 0.60: return f"偏多(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.35: return f"中性(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.15: return f"偏防御(仓位{pos*100:.0f}%) — 仅{bulls}个看多信号"
        else: return f"防御(仓位{pos*100:.0f}%) — 几乎无看多信号" + ("，强制限制生效" if capped else "")

# ============================================================
# 模块5: 交易决策生成器
# ============================================================

def compute_atr_stop_loss(closes, highs, lows, cost_price,
                          atr_period=14, atr_mult=2.5,
                          min_stop_pct=-5.0, max_stop_pct=-12.0):
    """
    ATR自适应动态止损 (v3.0 新增)

    逻辑:
    - 基于ATR（平均真实波幅）计算止损距离
    - 高波动ETF → 更宽止损（避免被正常波动震出）
    - 低波动ETF → 更窄止损（更快止损）
    - 硬约束: 止损比例在min_stop_pct~max_stop_pct之间
    - 市场危机状态下可以用更紧的止损

    参数:
        closes, highs, lows: 价格序列
        cost_price: 持仓成本价
        atr_period: ATR计算周期
        atr_mult: ATR倍数（默认2.5倍ATR）
        min_stop_pct: 最小止损百分比（如-5%）
        max_stop_pct: 最大止损百分比（如-12%）

    返回: {
        "stop_price": float,
        "stop_pct": float (负数),
        "atr_value": float,
        "atr_pct": float (ATR占价格百分比),
        "method": "ATR自适应"
    }
    """
    if len(closes) < atr_period + 1:
        # 数据不足，回退到固定-8%
        stop_pct = -0.08
        return {
            "stop_price": round(cost_price * (1 + stop_pct), 4),
            "stop_pct": round(stop_pct * 100, 1),
            "atr_value": 0,
            "atr_pct": 0,
            "method": "固定止损(数据不足)"
        }

    # 计算ATR
    tr_list = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr_list.append(max(hl, hc, lc))

    atr_val = sum(tr_list[-atr_period:]) / atr_period
    current_price = closes[-1]
    atr_pct = atr_val / current_price * 100 if current_price > 0 else 2.0

    # 止损距离 = ATR × 倍数，转换为百分比
    stop_distance_pct = -(atr_pct * atr_mult)

    # 硬约束
    if stop_distance_pct > min_stop_pct:  # 比-5%更近 → 用最小值
        stop_distance_pct = min_stop_pct
    if stop_distance_pct < max_stop_pct:  # 比-12%更远 → 用最大值
        stop_distance_pct = max_stop_pct

    stop_price = cost_price * (1 + stop_distance_pct / 100)

    return {
        "stop_price": round(stop_price, 4),
        "stop_pct": round(stop_distance_pct, 1),
        "atr_value": round(atr_val, 4),
        "atr_pct": round(atr_pct, 2),
        "method": f"ATR({atr_period}d)×{atr_mult}"
    }


class TradeDecider:
    """根据因子得分+择时+持仓生成操作计划"""

    def __init__(self, etf_scores, timing_result, portfolio=None):
        """
        etf_scores: [{code, name, score, grade, price, indicators, returns, vs_ma, factors}]
        timing_result: MarketTiming.position_advice() output
        portfolio: {code: {shares, cost, name}} 当前持仓
        """
        self.scores = sorted(etf_scores, key=lambda x: x["score"], reverse=True)
        self.timing = timing_result
        self.portfolio = portfolio or {}

    def generate_plan(self, total_capital=4000, max_single=0.25, max_industry=0.40):
        """
        生成明日操作计划
        total_capital: 总资金
        max_single: 单品种最大仓位
        max_industry: 单行业最大仓位
        """
        target_pos = self.timing["base_position"]
        target_amount = total_capital * target_pos
        current_invested = sum(
            p.get("shares", 0) * p.get("current_price", p.get("cost", 0))
            for p in self.portfolio.values() if isinstance(p, dict)
        )

        # 排名前40%的ETF
        top_n = max(5, len(self.scores)//2)
        top_etfs = self.scores[:top_n]

        buy_list = []
        sell_list = []
        hold_list = []

        # 卖出判断
        for code, pos in self.portfolio.items():
            if code.startswith("_") or not isinstance(pos, dict):
                continue  # 跳过元数据（如_available_cash）
            score_data = next((s for s in self.scores if s["code"] == code), None)
            if not score_data:
                continue

            current_price = score_data["price"]
            cost = pos.get("cost", current_price)
            pnl_pct = (current_price/cost - 1)*100 if cost > 0 else 0

            reasons = []
            # 条件1: 得分跌出前40% 且 评分低于B级(65)
            # ⚠️ 双重条件：排名下滑+分数弱才触发，避免"昨天B级买今天B级卖"
            rank_threshold = self.scores[len(self.scores)*4//10]["score"] if len(self.scores)>=10 else 50
            if score_data["score"] < rank_threshold and score_data["score"] < 65:
                reasons.append("得分排名下滑")
            # 条件2: 从最高点回撤-8%
            if pnl_pct <= -8:
                reasons.append(f"止损触发(浮亏{pnl_pct:.1f}%)")
            # 条件3: 大盘择时转为防御且仓位需要削减
            # ⚠️ B级以上（≥65分）不因防御模式强制卖出，避免"昨天买今天卖"
            if target_pos < 0.3 and current_invested > target_amount:
                if score_data["score"] < 65:
                    reasons.append("大盘防御模式，需减仓")

            if reasons:
                sell_list.append({
                    "code": code, "name": pos.get("name", code),
                    "action": "卖出", "shares": pos.get("shares", 0),
                    "current_price": current_price, "pnl_pct": round(pnl_pct, 1),
                    "reasons": reasons
                })
            else:
                hold_list.append({
                    "code": code, "name": pos.get("name", code),
                    "action": "持有", "shares": pos.get("shares", 0),
                    "current_price": current_price, "pnl_pct": round(pnl_pct, 1)
                })

        # 买入建议
        remaining = target_amount - current_invested
        if remaining > 0 and target_pos >= 0.2:
            for etf in top_etfs:
                if etf["code"] in self.portfolio: continue  # 已持有
                if etf["grade"] in ["D_谨慎", "E_回避"]: continue

                # 单只仓位上限
                allocation = min(remaining * 0.3, total_capital * max_single)
                shares = int(allocation / etf["price"])
                if shares < 100: continue

                amount = shares * etf["price"]

                # 生成买入理由
                reasons = []
                if etf["indicators"]["rsi"] <= 55:
                    reasons.append(f"RSI={etf['indicators']['rsi']} 中性偏低")
                if etf["returns"]["r5d"] <= 0:
                    reasons.append("短期回调充分")
                if etf["indicators"]["consecutive_up"] <= 1:
                    reasons.append("非追高(连涨≤1天)")
                if etf["factors"].get("F1_趋势强度", 0) >= 7:
                    reasons.append("趋势强")
                if not reasons:
                    reasons.append(f"综合得分{etf['score']}分，排名前{top_n}")

                buy_list.append({
                    "code": etf["code"], "name": etf["name"],
                    "action": "买入", "shares": shares,
                    "price": etf["price"], "amount": round(amount, 0),
                    "score": etf["score"], "grade": etf["grade"],
                    "reasons": reasons[:3]
                })
                remaining -= amount

                if len(buy_list) >= 5: break

        return {
            "target_position": round(target_pos*100),
            "target_amount": round(target_amount, 0),
            "current_invested": round(current_invested, 0),
            "remaining": round(remaining, 0),
            "buy_list": buy_list,
            "sell_list": sell_list,
            "hold_list": hold_list,
            "timing_advice": self.timing["advice"]
        }

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 简易测试
    print("Quant Engine v2.0 — Ready (自适应权重)", file=sys.stderr)
    print(f"  已加载权重: {len(CURRENT_WEIGHTS)}个因子", file=sys.stderr)
    if WEIGHT_CONFIG.get("meta", {}).get("last_optimized"):
        print(f"  上次优化: {WEIGHT_CONFIG['meta']['last_optimized']}", file=sys.stderr)
        print(f"  优化IC: {WEIGHT_CONFIG['meta'].get('ic_score', 'N/A')}", file=sys.stderr)
    print("Usage: import quant_engine and call score_etf_comprehensive()")
