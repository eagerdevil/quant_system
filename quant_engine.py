#!/usr/bin/env python
"""
量化系统 核心引擎 v6.0 (pandas+numpy 向量化)
=============================================
模块2: 数据预处理（去极值+标准化+中性化）
模块3: 多因子模型（动量/波动/基本面/资金/情绪）
模块4: 大盘择时与仓位管理
模块5: 交易决策生成

v6.0 变更:
- 全部指标计算迁移到 pandas/numpy 向量化
- 保持与 v5.0 完全相同的 API 和输出格式
- 性能提升约 10-50x（取决于数据量）
"""
import json, math, sys, os, logging
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    """加载因子权重配置，失败时返回默认等权。
    v5.0: 自动检测因子衰减，对IC为负的因子降低权重。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "factor_weights.json")
    try:
        with open(weights_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        w = config.get("factor_weights", dict(DEFAULT_WEIGHTS))
        # 确保所有因子都有权重
        for k in FACTOR_NAMES:
            if k not in w:
                w[k] = 1.0

        # v5.0: 因子衰减自动降权
        factor_health = config.get("factor_health", {})
        if factor_health:
            decayed = []
            for k, original_w in list(w.items()):
                ic = factor_health.get(k, 0.01)
                if ic < -0.02:
                    # IC严重为负: 降至原权重的30%
                    w[k] = round(original_w * 0.30, 2)
                    decayed.append(f"{k}(IC={ic:.4f},w:{original_w}->{w[k]})")
                elif ic < -0.01:
                    # IC略负: 降至原权重的60%
                    w[k] = round(original_w * 0.60, 2)
                    decayed.append(f"{k}(IC={ic:.4f},w:{original_w}->{w[k]})")
            if decayed:
                logger.info(f"[FACTOR DECAY] 衰减因子: {', '.join(decayed)}")
        return w, config
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return dict(DEFAULT_WEIGHTS), {"meta": {"version": 0, "ic_score": 0}}

CURRENT_WEIGHTS, WEIGHT_CONFIG = _load_weights()

# ============================================================
# v5.0: 系统运行参数（集中管理，避免魔法数字散落各处）
# ============================================================
SYSTEM_CONFIG = {
    # --- 大盘择时 ---
    "volume_active_threshold": 20000,    # 成交额阈值(亿)，低于此值视为缩量
    "limit_down_danger": 20,             # 跌停家数警戒线
    "margin_lookback": 5,                # 融资余额回看天数
    # --- 技术指标 ---
    "rsi_period": 14,
    "rsi_overbought": 68,               # RSI过热线
    "rsi_oversold": 30,                 # RSI超卖线
    "bollinger_period": 20,
    "bollinger_std": 2,
    "adx_default": 20,                  # 数据不足时默认ADX
    "adx_trend_threshold": 22,          # ADX趋势判定阈值
    # --- 仓位管理 ---
    "max_single_weight": 0.25,          # 单只ETF仓位上限
    "max_total_holdings": 5,            # 最大持仓数
    "default_stop_loss": -0.08,         # 默认止损线
    "default_take_profit": 0.08,        # 默认止盈线
    # --- 回测 ---
    "commission_rate": 0.0005,          # 手续费率(万5)
    "min_klines_for_backtest": 300,     # 回测最少K线条数
    "backtest_lookback_days": 250,      # 回测预加载天数
}
# 向下兼容快捷引用
CFG = SYSTEM_CONFIG

# v4.0: 从配置文件中读取自进化参数
OPTIMIZED_PARAMS = {
    "grade_thresholds": WEIGHT_CONFIG.get("grade_thresholds", {
        "A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42
    }),
    "premium_steepness": WEIGHT_CONFIG.get("premium_steepness", 0.07),
    "premium_threshold": WEIGHT_CONFIG.get("premium_threshold", 2.5),
    "adx_trend_threshold": WEIGHT_CONFIG.get("adx_trend_threshold", 22)
}


# ============================================================
# 工具函数：list <-> numpy 互转
# ============================================================
def _to_np(series):
    """将 list 转为 numpy array（float64）"""
    if isinstance(series, np.ndarray):
        return series.astype(np.float64)
    return np.array(series, dtype=np.float64)

def _to_list(arr):
    """将 numpy array 转为 Python list"""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


# ============================================================
# 模块2: 数据预处理 — numpy 向量化
# ============================================================
def mad_outlier_filter(series, n=5.0):
    """MAD法去极值：5倍绝对中位差 — numpy 向量化"""
    if len(series) < 3:
        return series
    arr = _to_np(series)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) * 1.4826  # 一致估计量
    if mad == 0:
        return series
    upper = median + n * mad
    lower = median - n * mad
    result = np.clip(arr, lower, upper)
    return _to_list(result)


def zscore_normalize(series):
    """Z-score标准化 — numpy 向量化"""
    if len(series) < 2:
        return [0] * len(series)
    arr = _to_np(series)
    std = np.std(arr, ddof=1)
    if std == 0:
        return [0] * len(arr)
    result = (arr - np.mean(arr)) / std
    return _to_list(result)


# ============================================================
# 模块3: 多因子计算 — 指标函数 (pandas/numpy 向量化)
# ============================================================

def sma(data, n):
    """简单移动平均 — 返回标量（与旧版兼容）"""
    if len(data) < n:
        return float(data[-1]) if len(data) > 0 else 0.0
    s = pd.Series(data)
    return float(s.rolling(n).mean().iloc[-1])


def ema(data, n):
    """指数移动平均 — 返回标量"""
    if len(data) < n:
        return float(data[-1]) if len(data) > 0 else 0.0
    s = pd.Series(data)
    return float(s.ewm(span=n, adjust=False).mean().iloc[-1])


def rsi(closes, n=14):
    """RSI 指标 — numpy 向量化，返回标量"""
    if len(closes) < n + 1:
        return 50.0
    arr = _to_np(closes)
    deltas = np.diff(arr[-n-1:])
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def macd_calc(closes):
    """返回 (DIF, DEA, Hist) 当前值"""
    s = pd.Series(closes)
    e12 = s.ewm(span=12, adjust=False).mean().iloc[-1]
    e26 = s.ewm(span=26, adjust=False).mean().iloc[-1]
    dif = float(e12 - e26)

    # 简化DEA
    if len(closes) >= 35:
        # 逐期计算 DIF 序列再平滑
        e12_series = s.ewm(span=12, adjust=False).mean()
        e26_series = s.ewm(span=26, adjust=False).mean()
        dif_series = e12_series - e26_series
        # 取最后有意义的DIF值（从第26个开始）
        dif_valid = dif_series.iloc[26:]
        if len(dif_valid) > 0:
            dea = float(pd.Series(dif_valid.values).ewm(span=9, adjust=False).mean().iloc[-1])
        else:
            dea = dif
    else:
        dea = dif

    hist = float((dif - dea) * 2)
    return dif, dea, hist


def max_drawdown(closes):
    """最大回撤 — numpy 向量化，返回标量"""
    arr = _to_np(closes)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / peak
    return float(np.max(dd))


def volatility(closes, n=20):
    """年化波动率 — numpy 向量化，返回标量"""
    if len(closes) < n + 1:
        return 0.0
    arr = _to_np(closes[-n-1:])
    rets = np.diff(arr) / arr[:-1]
    return float(np.std(rets, ddof=0) * np.sqrt(252))


def sharpe(closes, rf=0.025):
    """夏普比率 — numpy 向量化，返回标量"""
    arr = _to_np(closes)
    rets = np.diff(arr) / arr[:-1]
    avg_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 0.01
    if std_ret == 0:
        return 0.0
    return float((avg_ret - rf / 252) / std_ret * np.sqrt(252))


def sortino(closes, rf=0.025):
    """Sortino比率 — numpy 向量化，返回标量"""
    arr = _to_np(closes)
    rets = np.diff(arr) / arr[:-1]
    avg_ret = np.mean(rets)
    down_rets = rets[rets < 0]
    if len(down_rets) < 2:
        return 3.0 if avg_ret > 0 else -3.0
    down_std = np.std(down_rets, ddof=1)
    if down_std == 0:
        return 0.0
    return float((avg_ret - rf / 252) / down_std * np.sqrt(252))


def consecutive_up(closes):
    """连续上涨天数 — 返回整数"""
    cnt = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            cnt += 1
        else:
            break
    return cnt


def consecutive_down(closes):
    """连续下跌天数 — 返回整数"""
    cnt = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            cnt += 1
        else:
            break
    return cnt


def ret_n(closes, n):
    """N日收益率(%) — 返回标量"""
    if len(closes) < n + 1:
        return 0.0
    return float((closes[-1] / closes[-n - 1] - 1) * 100)


def vol_ratio(volumes, n=5):
    """成交量比率 — 返回标量"""
    if len(volumes) < n * 2:
        return 1.0
    recent = np.mean(volumes[-n:])
    prior = np.mean(volumes[-n * 2:-n])
    return float(recent / prior) if prior > 0 else 1.0


def pma(closes, n):
    """价格相对MA偏离(%) — 返回标量"""
    ma = sma(closes, n)
    if ma == 0:
        return 0.0
    return float((closes[-1] / ma - 1) * 100)


def ma_alignment(closes):
    """MA多头排列程度：MA5>MA10>MA20>MA60 — 返回整数0-3"""
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60) if len(closes) >= 60 else sma(closes, len(closes))
    score = 0
    if ma5 > ma10:
        score += 1
    if ma10 > ma20:
        score += 1
    if ma20 > ma60:
        score += 1
    return score


def bollinger_position(closes, n=20):
    """布林带位置 0-100 — numpy 向量化，返回标量"""
    if len(closes) < n:
        return 50.0
    window = _to_np(closes[-n:])
    avg = np.mean(window)
    std = np.std(window, ddof=0)
    if std == 0:
        return 50.0
    upper = avg + 2 * std
    lower = avg - 2 * std
    if upper == lower:
        return 50.0
    return float((closes[-1] - lower) / (upper - lower) * 100)


def _wilder_seq(series, n):
    """Wilder平滑器辅助函数 — numpy 向量化"""
    if len(series) < n:
        return [sum(series)] if series else [0]
    result = [sum(series[:n])]
    for i in range(n, len(series)):
        result.append(result[-1] - result[-1] / n + series[i])
    return result


def adx(closes, highs, lows, n=14):
    """
    平均趋向指数 (Average Directional Index) — Wilder算法完整实现+numpy辅助
    返回: adx值 (0-100, >25趋势市, <20震荡市)
    """
    if len(closes) < n + 1:
        return SYSTEM_CONFIG['adx_default']

    c_arr = _to_np(closes)
    h_arr = _to_np(highs)
    l_arr = _to_np(lows)

    # Step 1: True Range / +DM / -DM 原始序列 — numpy向量化
    hl = h_arr[1:] - l_arr[1:]
    hc = np.abs(h_arr[1:] - c_arr[:-1])
    lc = np.abs(l_arr[1:] - c_arr[:-1])
    tr_list = np.maximum(np.maximum(hl, hc), lc).tolist()

    up = h_arr[1:] - h_arr[:-1]
    down = l_arr[:-1] - l_arr[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0).tolist()
    minus_dm = np.where((down > up) & (down > 0), down, 0).tolist()

    if len(tr_list) < n:
        return 20.0

    # Step 2: Wilder平滑
    atr_seq = _wilder_seq(tr_list, n)
    pdi_seq = _wilder_seq(plus_dm, n)
    mdi_seq = _wilder_seq(minus_dm, n)

    # Step 3: 逐期计算 DX — numpy向量化
    atr_arr = np.array(atr_seq, dtype=np.float64)
    pdi_arr = np.array(pdi_seq, dtype=np.float64)
    mdi_arr = np.array(mdi_seq, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_arr > 0, pdi_arr / atr_arr * 100, 0)
        mdi = np.where(atr_arr > 0, mdi_arr / atr_arr * 100, 0)
        denom = pdi + mdi
        dx_values = np.where(denom > 0, np.abs(pdi - mdi) / denom * 100, 0)

    dx_list = dx_values.tolist()

    # Step 4: 平滑DX序列 → ADX
    if len(dx_list) <= n:
        return float(sum(dx_list) / len(dx_list)) if dx_list else 20.0

    adx_seq = _wilder_seq(dx_list, n)
    return float(adx_seq[-1] / n)


def bollinger_bandwidth(closes, n=20):
    """布林带宽度 — numpy 向量化，返回标量"""
    if len(closes) < n:
        return 0.05
    window = _to_np(closes[-n:])
    avg = np.mean(window)
    std = np.std(window, ddof=0)
    if avg == 0:
        return 0.0
    return float((4 * std) / avg)


def calc_choppiness(closes, highs, lows, n=14):
    """
    震荡指数 (Choppiness Index) — numpy 向量化
    值越高越震荡 (>61.8 = 震荡), 越低越趋势 (<38.2 = 趋势)
    """
    if len(closes) < n:
        return 50.0

    h_arr = _to_np(highs[-n:])
    l_arr = _to_np(lows[-n:])
    c_arr = _to_np(closes[-(n+1):])

    window_high = float(np.max(h_arr))
    window_low = float(np.min(l_arr))

    # True Range for last n periods
    hl = h_arr[1:] - l_arr[1:] if len(h_arr) > 1 else h_arr - l_arr
    hc_val = np.abs(h_arr[1:] - c_arr[:-n]) if len(c_arr) >= n + 1 and len(h_arr) > 1 else np.abs(h_arr - c_arr[-len(h_arr):])
    lc_val = np.abs(l_arr[1:] - c_arr[:-n]) if len(c_arr) >= n + 1 and len(l_arr) > 1 else np.abs(l_arr - c_arr[-len(l_arr):])

    # Pad or trim to same length
    min_len = min(len(hl), len(hc_val), len(lc_val))
    hl = hl[:min_len]
    hc_val = hc_val[:min_len]
    lc_val = lc_val[:min_len]

    tr_sum = float(np.sum(np.maximum(np.maximum(hl, hc_val), lc_val)))
    if tr_sum == 0 or window_high == window_low:
        return 50.0

    chop = 100 * math.log10(tr_sum / (window_high - window_low)) / math.log10(n)
    return float(max(0, min(100, chop)))


# ============================================================
# 量价背离检测 (v3.0 新增) — 保持原有逻辑（非向量化瓶颈）
# ============================================================
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
        strength = min(10, round((1 - vol_change) * 10 + (max_close_recent / max_close_prior - 1) * 50))
        return {
            "divergence_type": "bearish",
            "strength": strength,
            "description": f"看跌背离: 价格创新高(+{(max_close_recent/max_close_prior-1)*100:.1f}%)但量缩{(1-vol_change)*100:.0f}%",
            "vol_change": round(vol_change, 2),
            "price_action": "新高量缩"
        }

    # 2. 看涨背离: 价格创新低 + 量增 (>10%)
    if min_close_recent < min_close_prior and vol_change > 1.10:
        strength = min(10, round((vol_change - 1) * 8 + (1 - min_close_recent / min_close_prior) * 50))
        return {
            "divergence_type": "bullish",
            "strength": strength,
            "description": f"看涨背离: 价格创新低(-{(1-min_close_recent/min_close_prior)*100:.1f}%)但量增{(vol_change-1)*100:.0f}%",
            "vol_change": round(vol_change, 2),
            "price_action": "新低量增"
        }

    # 3. 缩量反弹: 连涨3天+ + 近3日量<前10日均量80%
    cons_up = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            cons_up += 1
        else:
            break
    recent_3vol = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else avg_vol_recent
    if cons_up >= 3 and recent_3vol < avg_vol_prior * 0.80:
        strength = min(10, cons_up * 2 + round((1 - recent_3vol / avg_vol_prior) * 10))
        return {
            "divergence_type": "weak_rally",
            "strength": strength,
            "description": f"缩量反弹: 连涨{cons_up}天但近3日量仅为前均{(recent_3vol/avg_vol_prior)*100:.0f}%",
            "vol_change": round(recent_3vol / avg_vol_prior, 2),
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
        return {
            "regime": "CHOPPY", "confidence": 0.3,
            "signals": {}, "description": "数据不足，默认震荡"
        }

    # 用收盘价的复制简化highs/lows
    if index_highs is None:
        index_highs = [c * 1.005 for c in index_closes]
    if index_lows is None:
        index_lows = [c * 0.995 for c in index_closes]

    signals = {}

    # Signal 1: ADX趋势强度（v4.0: 使用可配置阈值）
    adx_threshold = OPTIMIZED_PARAMS.get("adx_trend_threshold", 22)
    adx_val = adx(index_closes, index_highs, index_lows, 14)
    signals["adx"] = round(adx_val, 1)
    is_trending = adx_val > adx_threshold
    is_strong_trend = adx_val > (adx_threshold + 6)

    # Signal 2: 均线排列
    ma_align = ma_alignment(index_closes)
    signals["ma_alignment"] = ma_align
    ma_bullish = ma_align >= 2
    ma_bearish = ma_align == 0

    # Signal 3: 布林带宽度（波动率状态）
    bbw = bollinger_bandwidth(index_closes, 20)
    signals["bb_width"] = round(bbw, 4)
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
    below_ma60 = index_closes[-1] < ma60 * 0.95
    signals["below_ma60_pct"] = round((index_closes[-1] / ma60 - 1) * 100, 1) if ma60 > 0 else 0

    # === 状态判定 ===
    regime_signals = []
    confidence = 0.5

    # 危机判定
    if sharp_decline and (below_ma60 or r20 < -15):
        regime = "CRISIS"
        confidence = 0.85
        regime_signals.append(f"急速下跌(5日{r5:.1f}%, 20日{r20:.1f}%)")
        if below_ma60:
            regime_signals.append("跌破MA60超过5%")
    elif ma_bearish and is_trending:
        regime = "TREND_DOWN"
        confidence = 0.80
        regime_signals.append("均线空头排列")
        regime_signals.append(f"ADX={adx_val:.1f}(趋势明确)")
    elif is_choppy or not is_trending:
        regime = "CHOPPY"
        if is_choppy and not is_trending:
            confidence = 0.75
            regime_signals.append(f"震荡指数={chop:.1f}(>55震荡)")
            regime_signals.append(f"ADX={adx_val:.1f}(<22无趋势)")
        else:
            confidence = 0.55
            regime_signals.append("方向不明确")
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


# ============================================================
# 模块3b: 因子计算（导出供 optimizer 使用）
# ============================================================
def compute_indicators(closes, highs, lows, volumes):
    """
    计算所有原始指标。
    输入: ETF的价格序列（按时间升序）
    输出: 指标字典，可直接传给 score_factors()
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
    atr_val = sum(abs(closes[i] - closes[i - 1]) for i in range(-14, 0)) / 14 if len(closes) >= 15 else 0
    atr_pct = atr_val / cur * 100 if cur > 0 else 0

    r5 = ret_n(closes, 5)
    r10 = ret_n(closes, 10)
    r20 = ret_n(closes, 20)
    r60 = ret_n(closes, 60)
    r120 = ret_n(closes, 120) if len(closes) >= 121 else 0

    v_ratio = vol_ratio(volumes, 5)
    v_ratio_10 = vol_ratio(volumes, 10)

    pm5 = pma(closes, 5)
    pm10 = pma(closes, 10)
    pm20 = pma(closes, 20)
    pm60 = pma(closes, 60) if len(closes) >= 60 else pm20

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
        "volume_divergence": detect_volume_price_divergence(closes, volumes)
    }


def score_factors(indicators):
    """
    将原始指标转换为16个因子得分（0-10/0-8/0-6/0-4）。
    输入: compute_indicators() 的输出
    输出: {F1_趋势强度: 5, F2_动量: 8, ...}
    """
    factors = {}
    ind = indicators

    # F1: 趋势强度(0-10)
    f1 = 5
    if ind["dif"] > 0 and ind["hist"] > 0:
        f1 = 10
    elif ind["dif"] > 0:
        f1 = 7
    elif ind["dif"] > -0.01 and ind["hist"] > 0:
        f1 = 6
    elif ind["dif"] > -0.01:
        f1 = 4
    else:
        f1 = 2
    if ind["ma_alignment"] >= 2:
        f1 = min(10, f1 + 1)
    factors['F1_趋势强度'] = f1

    # F2: 动量(0-8)
    r20 = ind["r20d"]
    if 0 <= r20 <= 10:
        f2 = 8
    elif -3 <= r20 < 0:
        f2 = 6
    elif 10 < r20 <= 20:
        f2 = 5
    elif r20 < -10:
        f2 = 3
    else:
        f2 = 2
    factors['F2_动量'] = f2

    # F3: 反转信号(0-8)
    r5 = ind["r5d"]
    if r5 < -5:
        f3 = 8
    elif -5 <= r5 < -2:
        f3 = 6
    elif r5 < 0:
        f3 = 5
    elif 0 <= r5 <= 3:
        f3 = 4
    elif 3 < r5 <= 8:
        f3 = 3
    else:
        f3 = 1
    factors['F3_反转'] = f3

    # F4: RSI位置(0-8)
    rsi_now = ind["rsi"]
    if 40 <= rsi_now <= 50:
        f4 = 8
    elif 35 <= rsi_now < 40:
        f4 = 7
    elif 50 < rsi_now <= 58:
        f4 = 6
    elif 30 <= rsi_now < 35:
        f4 = 5
    elif 58 < rsi_now <= 65:
        f4 = 4
    elif 65 < rsi_now <= 72:
        f4 = 2
    else:
        f4 = 1
    factors['F4_RSI'] = f4

    # F5: 均线偏离(0-6)
    pm5 = ind["pma5"]
    if -3 <= pm5 <= 3:
        f5 = 6
    elif -5 <= pm5 < -3:
        f5 = 5
    elif 3 < pm5 <= 8:
        f5 = 4
    elif -8 <= pm5 < -5:
        f5 = 3
    else:
        f5 = 2
    factors['F5_均线偏离'] = f5

    # F6: 低波动(0-6)
    vol_pct = ind["volatility"] * 100
    if 15 <= vol_pct <= 30:
        f6 = 6
    elif 10 <= vol_pct < 15:
        f6 = 5
    elif 30 < vol_pct <= 40:
        f6 = 4
    elif vol_pct < 10:
        f6 = 3
    else:
        f6 = 2
    factors['F6_低波动'] = f6

    # F7: 成交量健康(0-6)
    v_ratio = ind["vol_ratio_5d"]
    if 0.85 <= v_ratio <= 1.20:
        f7 = 6
    elif 0.70 <= v_ratio <= 1.40:
        f7 = 4
    else:
        f7 = 2
    factors['F7_成交量'] = f7

    # F8: 回调质量(0-10)
    cons_down = ind["consecutive_down"]
    cons_up = ind["consecutive_up"]
    if cons_down >= 3:
        f8 = 10
    elif cons_down >= 2:
        f8 = 8
    elif cons_down == 1:
        f8 = 7
    elif cons_up == 0:
        f8 = 7
    elif cons_up == 1:
        f8 = 6
    elif cons_up == 2:
        f8 = 4
    else:
        f8 = 2
    factors['F8_回调'] = f8

    # F9: Sortino(0-6)
    srt = ind["sortino"]
    if srt > 1.5:
        f9 = 6
    elif srt > 0.8:
        f9 = 5
    elif srt > 0.3:
        f9 = 4
    elif srt > -0.3:
        f9 = 3
    else:
        f9 = 2
    factors['F9_Sortino'] = f9

    # F10: 最大回撤(0-6)
    maxdd = ind["max_drawdown"]
    if maxdd < 0.10:
        f10 = 6
    elif maxdd < 0.18:
        f10 = 5
    elif maxdd < 0.25:
        f10 = 4
    elif maxdd < 0.35:
        f10 = 3
    else:
        f10 = 2
    factors['F10_MaxDD'] = f10

    # F11: 布林带位置(0-6)
    bb_pos = ind["bb_position"]
    if 15 <= bb_pos <= 55:
        f11 = 6
    elif 5 <= bb_pos < 15:
        f11 = 5
    elif 55 < bb_pos <= 75:
        f11 = 4
    elif 75 < bb_pos <= 90:
        f11 = 2
    else:
        f11 = 1
    factors['F11_布林带'] = f11

    # F12: 多周期收益(0-6)
    r60 = ind["r60d"]
    if r5 > 0 and r20 > 0 and r60 > 0:
        f12 = 6
    elif r20 > 0 and r60 > 0:
        f12 = 5
    elif r60 > 0:
        f12 = 4
    elif r20 > 0:
        f12 = 3
    else:
        f12 = 2
    factors['F12_多周期'] = f12

    # F13: 均线排列(0-4)
    factors['F13_均线排列'] = ind["ma_alignment"]

    # F14: 长期收益(0-4)
    r120 = ind["r120d"]
    if r120 > 10:
        f14 = 4
    elif r120 > 0:
        f14 = 3
    elif r120 > -10:
        f14 = 2
    else:
        f14 = 1
    factors['F14_长期'] = f14

    # F15: 夏普比率(0-6)
    shp = ind["sharpe"]
    if shp > 1.5:
        f15 = 6
    elif shp > 0.8:
        f15 = 5
    elif shp > 0.3:
        f15 = 4
    elif shp > -0.3:
        f15 = 3
    else:
        f15 = 2
    factors['F15_夏普'] = f15

    # F16: 量价关系 (0-8) — v3.0 新增
    vol_div = ind.get("volume_divergence", {})
    div_type = vol_div.get("divergence_type")
    if div_type == "bullish":
        f16 = 8
    elif div_type == "bearish":
        f16 = 2
    elif div_type == "weak_rally":
        f16 = 3
    elif div_type == "panic_sell":
        f16 = 4
    else:
        f16 = 5
    factors['F16_量价关系'] = f16

    return factors


# ============================================================
# 综合因子评分（ETF级别）— v2.0 支持自适应权重
# ============================================================
def _apply_premium_penalty(technical_score, premium_pct):
    """
    溢价惩罚函数 — 修正量化模型对QDII ETF溢价的盲区
    """
    if premium_pct is None:
        return technical_score, 1.0, "QDII溢价数据缺失，无法评估溢价风险"

    premium_threshold = OPTIMIZED_PARAMS.get("premium_threshold", 2.0)
    steepness = OPTIMIZED_PARAMS.get("premium_steepness", 0.07)

    if premium_pct < premium_threshold:
        return technical_score, 1.0, None

    excess = premium_pct - premium_threshold
    if premium_pct <= 5.0:
        multiplier = 1.0 - excess * steepness
    elif premium_pct <= 8.0:
        base_loss = (5.0 - premium_threshold) * steepness
        multiplier = max(0.50, 1.0 - base_loss - (premium_pct - 5.0) * steepness * 1.3)
    else:
        base_loss = (5.0 - premium_threshold) * steepness + 3.0 * steepness * 1.3
        multiplier = max(0.45, 1.0 - base_loss - (premium_pct - 8.0) * steepness * 1.6)

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
    16因子综合评分系统 + 溢价惩罚
    """
    if weights is None:
        weights = CURRENT_WEIGHTS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    indicators = compute_indicators(closes, highs, lows, volumes)
    factors = score_factors(indicators)

    weighted_sum = sum(factors[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
    max_weighted = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
    if max_weighted > 0:
        technical_score = round(weighted_sum / max_weighted * 100)
    else:
        technical_score = sum(factors.values())

    adjusted_score, premium_multiplier, premium_warning = _apply_premium_penalty(
        technical_score, premium_pct
    )
    final_score = adjusted_score

    grade_thresholds = OPTIMIZED_PARAMS.get("grade_thresholds", {
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
        "score": final_score,
        "technical_score": technical_score,
        "max_score": 100,
        "grade": grade,
        "price": round(indicators["price"], 4),
        "premium_info": {
            "premium_pct": premium_pct,
            "penalty_multiplier": premium_multiplier,
            "warning": premium_warning,
            "score_penalty": technical_score - final_score
        },
        "indicators": {
            "rsi": round(indicators["rsi"], 1),
            "volatility_pct": round(indicators["volatility"] * 100, 1),
            "sharpe": round(indicators["sharpe"], 2),
            "sortino": round(indicators["sortino"], 2),
            "max_dd_pct": round(indicators["max_drawdown"] * 100, 1),
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
        "volume_divergence": indicators.get("volume_divergence", {})
    }


# ============================================================
# 模块3c: 横截面比较 (v5.0 新增) — numpy 向量化
# ============================================================
CROSS_SECTION_METRICS = [
    ("rsi",              1.5, -1),
    ("r20d",             1.2, -1),
    ("r60d",             1.0, +1),
    ("volatility_pct",   1.3, -1),
    ("sortino",          1.0, +1),
    ("sharpe",           0.8, +1),
    ("max_dd_pct",       1.2, -1),
    ("ma_alignment",     0.6, +1),
    ("bb_position",      0.7, -1),
    ("consecutive_up",   0.8, -1),
]


def compute_cross_sectional_adjustment(raw_metrics_list):
    """
    横截面比较: 对每个指标在所有ETF间做截面z-score标准化 — numpy 向量化
    """
    if len(raw_metrics_list) < 3:
        return {m["code"]: 0.0 for m in raw_metrics_list}

    codes = [m["code"] for m in raw_metrics_list]
    adjustments = {c: 0.0 for c in codes}

    for metric_name, weight, direction in CROSS_SECTION_METRICS:
        values = []
        valid_codes = []
        for m in raw_metrics_list:
            val = m.get(metric_name)
            if val is not None:
                values.append(val)
                valid_codes.append(m["code"])

        if len(values) < 3:
            continue

        arr = np.array(values, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr, ddof=0)
        if std < 1e-8:
            continue

        z = (arr - mean) / std
        z = np.clip(z, -3.0, 3.0)
        raw_adj = np.tanh(z) * weight * direction

        for i, code in enumerate(valid_codes):
            adjustments[code] += float(raw_adj[i])

    # 全局缩放到合理范围 [-8, +8]
    max_abs = max(abs(v) for v in adjustments.values()) if adjustments else 1.0
    if max_abs > 8.0:
        scale = 8.0 / max_abs
        for code in adjustments:
            adjustments[code] = round(adjustments[code] * scale, 1)
    else:
        for code in adjustments:
            adjustments[code] = round(adjustments[code], 1)

    return adjustments


def score_all_etfs_cross_sectional(etf_data_dict, weights=None):
    """
    v5.0: 批量评分 + 横截面比较融合 — 保持与旧版完全兼容
    """
    if weights is None:
        weights = CURRENT_WEIGHTS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Phase 1: 逐只计算绝对指标和因子得分
    raw_metrics = []
    for code, edata in etf_data_dict.items():
        kline = edata.get("kline", [])
        if len(kline) < 30:
            continue
        closes = [k["close"] for k in kline]
        highs = [k["high"] for k in kline]
        lows = [k["low"] for k in kline]
        volumes = [k["volume"] for k in kline]

        indicators = compute_indicators(closes, highs, lows, volumes)
        factors = score_factors(indicators)

        weighted_sum = sum(factors[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
        max_weighted = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
        technical_score = round(weighted_sum / max_weighted * 100) if max_weighted > 0 else sum(factors.values())

        raw_metrics.append({
            "code": code, "name": edata["name"],
            "rsi": indicators["rsi"],
            "r20d": indicators["r20d"],
            "r60d": indicators["r60d"],
            "volatility_pct": indicators["volatility"] * 100,
            "sortino": indicators["sortino"],
            "sharpe": indicators["sharpe"],
            "max_dd_pct": indicators["max_drawdown"] * 100,
            "ma_alignment": indicators["ma_alignment"],
            "bb_position": indicators["bb_position"],
            "consecutive_up": indicators["consecutive_up"],
            "indicators": indicators,
            "factors": factors,
            "technical_score": technical_score,
        })

    # Phase 2: 横截面比较
    cross_adjustments = compute_cross_sectional_adjustment(raw_metrics)

    # Phase 3: 融合评分
    results = []
    grade_thresholds = OPTIMIZED_PARAMS.get("grade_thresholds", {
        "A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42
    })

    for rm in raw_metrics:
        code = rm["code"]
        abs_score = rm["technical_score"]
        cross_adj = cross_adjustments.get(code, 0.0)

        blended_technical = round(abs_score + cross_adj)
        blended_technical = max(0, min(100, blended_technical))

        if blended_technical >= grade_thresholds.get("A_强烈买入", 78):
            grade = "A_强烈买入"
        elif blended_technical >= grade_thresholds.get("B_买入", 65):
            grade = "B_买入"
        elif blended_technical >= grade_thresholds.get("C_观察", 55):
            grade = "C_观察"
        elif blended_technical >= grade_thresholds.get("D_谨慎", 42):
            grade = "D_谨慎"
        else:
            grade = "E_回避"

        ind = rm["indicators"]
        results.append({
            "code": code, "name": rm["name"],
            "score": blended_technical,
            "technical_score": abs_score,
            "cross_sectional_adj": cross_adj,
            "blended_technical": blended_technical,
            "max_score": 100,
            "grade": grade,
            "price": round(ind["price"], 4),
            "indicators": {
                "rsi": round(ind["rsi"], 1),
                "volatility_pct": round(ind["volatility"] * 100, 1),
                "sharpe": round(ind["sharpe"], 2),
                "sortino": round(ind["sortino"], 2),
                "max_dd_pct": round(ind["max_drawdown"] * 100, 1),
                "consecutive_up": ind["consecutive_up"],
                "consecutive_down": ind["consecutive_down"],
                "ma_alignment": ind["ma_alignment"],
                "bb_position": round(ind["bb_position"], 1),
                "atr_pct": round(ind["atr_pct"], 2),
                "vol_ratio_5d": round(ind["vol_ratio_5d"], 2),
                "macd_dif": round(ind["dif"], 4),
                "macd_hist": round(ind["hist"], 4),
            },
            "returns": {
                "r5d": round(ind["r5d"], 1), "r10d": round(ind["r10d"], 1),
                "r20d": round(ind["r20d"], 1), "r60d": round(ind["r60d"], 1),
                "r120d": round(ind["r120d"], 1),
            },
            "vs_ma": {
                "pct_ma5": round(ind["pma5"], 1), "pct_ma10": round(ind["pma10"], 1),
                "pct_ma20": round(ind["pma20"], 1), "pct_ma60": round(ind["pma60"], 1),
            },
            "factors": rm["factors"],
            "volume_divergence": ind.get("volume_divergence", {}),
        })

    return results


# ============================================================
# 模块3d: 行业轮动因子 (v5.0 新增)
# ============================================================
ETF_INDUSTRY_MAP = {
    "562500": ["机械设备"],
    "512760": ["电子", "计算机"],
    "515070": ["计算机", "电子"],
    "159819": ["计算机", "电子"],
    "159995": ["电子"],
    "588200": ["电子"],
    "159516": ["电子"],
    "512880": ["非银金融"],
    "512670": ["国防军工"],
    "512810": ["国防军工"],
    "512400": ["有色金属"],
    "512170": ["医药生物"],
    "159992": ["医药生物"],
    "512890": ["银行", "公用事业"],
    "159928": ["食品饮料"],
    "159869": ["传媒"],
    "159870": ["化工"],
    "516020": ["化工"],
    "159611": ["公用事业"],
    "512200": ["房地产"],
    "159865": ["农林牧渔"],
    "159227": ["国防军工"],
    "159326": ["电气设备"],
    "159183": ["电气设备", "汽车"],
    "159320": ["电气设备"],
    "560390": ["电气设备"],
    "510300": ["综合"],
    "510500": ["综合"],
    "159915": ["综合"],
    "588000": ["电子", "计算机"],
    "512100": ["综合"],
    "518850": ["有色金属"],
    "513100": ["综合"],
    "513500": ["综合"],
    "159659": ["综合"],
}


def compute_industry_rotation_score(industry_returns, n_days=20):
    """行业轮动: 计算每个申万一级行业近N日收益率排名 — numpy向量化"""
    if not industry_returns or len(industry_returns) < 3:
        return {}

    industry_perf = {}
    for code, data in industry_returns.items():
        closes = data.get("closes", [])
        if len(closes) < n_days + 1:
            continue
        ret = (closes[-1] / closes[-n_days - 1] - 1) * 100
        industry_perf[data["name"]] = round(ret, 2)

    if not industry_perf:
        return {}

    names = list(industry_perf.keys())
    rets = np.array([industry_perf[n] for n in names])
    # argsort 获取排名
    order = np.argsort(rets)
    n = len(names)
    result = {}
    for rank_idx, idx in enumerate(order):
        name = names[idx]
        pct = round(rank_idx / (n - 1) * 100, 1) if n > 1 else 50
        result[name] = {"ret": industry_perf[name], "rank_pct": pct}

    return result


def get_etf_industry_momentum(code, industry_rotation):
    """
    获取某只ETF的行业轮动加成。
    如果ETF映射到多个行业，取平均排位。
    返回: bonus (范围 -3 ~ +3)
    """
    industries = ETF_INDUSTRY_MAP.get(code, [])
    if not industries or not industry_rotation:
        return 0.0

    rank_pcts = []
    for ind in industries:
        if ind in industry_rotation:
            rank_pcts.append(industry_rotation[ind]["rank_pct"])

    if not rank_pcts:
        return 0.0

    avg_pct = sum(rank_pcts) / len(rank_pcts)
    bonus = round((avg_pct - 50) / 50 * 3.0, 1)
    return bonus


# ============================================================
# 模块4: 大盘择时引擎
# ============================================================
class MarketTiming:
    """市场择时信号"""

    def __init__(self, index_data, north_flow, total_vol, breadth, margin):
        self.hs300 = None
        self.hs300_highs = None
        self.hs300_lows = None
        if "000300" in index_data:
            data = index_data["000300"]["data"]
            self.hs300 = [d["close"] for d in data]
            self.hs300_highs = [d.get("high", d["close"] * 1.005) for d in data]
            self.hs300_lows = [d.get("low", d["close"] * 0.995) for d in data]

        self.north_flow = north_flow or []
        self.total_vol = total_vol or 0
        self.breadth = breadth or {}
        self.margin = margin or {}

    def calc_signals(self):
        """计算6个择时信号，返回 {signal_name: bool}"""
        signals = {}

        if self.hs300 and len(self.hs300) >= 20:
            ma20 = sma(self.hs300, 20)
            signals['S1_HS300_above_MA20'] = self.hs300[-1] > ma20
        else:
            signals['S1_HS300_above_MA20'] = False

        if self.hs300 and len(self.hs300) >= 61:
            ma60_now = sma(self.hs300, 60)
            ma60_prev = sma(self.hs300[:-1], 60)
            signals['S2_HS300_MA60_up'] = ma60_now > ma60_prev
        else:
            signals['S2_HS300_MA60_up'] = False

        nf_5d = sum(f.get("net_flow", 0) for f in self.north_flow[-5:]) if self.north_flow else 0
        signals['S3_NorthFlow_5d_positive'] = nf_5d > 0

        signals['S4_Volume_active'] = self.total_vol > SYSTEM_CONFIG['volume_active_threshold']

        ld = self.breadth.get("limit_down")
        signals['S5_LimitDown_low'] = ld < SYSTEM_CONFIG['limit_down_danger'] if ld is not None else False

        margin_change = self.margin.get("change", 0)
        signals['S6_Margin_increasing'] = margin_change > 0

        return signals, nf_5d

    def position_advice(self):
        """根据择时信号+市场状态计算建议仓位"""
        signals, nf_5d = self.calc_signals()
        bull_count = sum(1 for v in signals.values() if v)

        if self.hs300 and len(self.hs300) >= 60:
            regime_result = classify_market_regime(
                self.hs300, self.hs300_highs, self.hs300_lows
            )
        else:
            regime_result = {
                "regime": "CHOPPY", "confidence": 0.3,
                "signals": {}, "regime_signals": ["数据不足"],
                "strategy": REGIME_STRATEGY["CHOPPY"],
                "description": "数据不足，默认震荡"
            }

        if bull_count >= 5:
            base_position = 1.0
        elif bull_count >= 4:
            base_position = 0.85
        elif bull_count >= 3:
            base_position = 0.65
        elif bull_count >= 2:
            base_position = 0.40
        elif bull_count >= 1:
            base_position = 0.20
        else:
            base_position = 0.05

        regime_strategy = regime_result.get("strategy", REGIME_STRATEGY["CHOPPY"])
        pos_range = regime_strategy.get("base_position", (0.20, 0.40))
        regime_stop_pct = regime_strategy.get("stop_loss", -0.08)
        buy_grade_min = regime_strategy.get("buy_grade_min", "B_买入")

        base_position = min(base_position, pos_range[1])
        base_position = max(base_position, pos_range[0])

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
            "regime": regime_result["regime"],
            "regime_name": MARKET_REGIME.get(regime_result["regime"], regime_result["regime"]),
            "regime_confidence": regime_result.get("confidence", 0),
            "regime_signals": regime_result.get("regime_signals", []),
            "regime_stop_loss": regime_stop_pct,
            "regime_buy_grade_min": buy_grade_min,
            "regime_description": regime_result.get("description", "")
        }

    def _position_text(self, pos, bulls, capped):
        if pos >= 0.85:
            return f"进攻(仓位{pos*100:.0f}%) — {bulls}个看多信号，市场强势"
        elif pos >= 0.60:
            return f"偏多(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.35:
            return f"中性(仓位{pos*100:.0f}%) — {bulls}个看多信号"
        elif pos >= 0.15:
            return f"偏防御(仓位{pos*100:.0f}%) — 仅{bulls}个看多信号"
        else:
            return f"防御(仓位{pos*100:.0f}%) — 几乎无看多信号" + ("，强制限制生效" if capped else "")


# ============================================================
# 模块5: 交易决策生成器
# ============================================================

def compute_atr_stop_loss(closes, highs, lows, cost_price,
                          atr_period=14, atr_mult=2.5,
                          min_stop_pct=-5.0, max_stop_pct=-12.0):
    """ATR自适应动态止损 — numpy向量化"""
    if len(closes) < atr_period + 1:
        stop_pct = -0.08
        return {
            "stop_price": round(cost_price * (1 + stop_pct), 4),
            "stop_pct": round(stop_pct * 100, 1),
            "atr_value": 0,
            "atr_pct": 0,
            "method": "固定止损(数据不足)"
        }

    tr_list = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hc, lc))

    atr_val = np.mean(tr_list[-atr_period:])
    current_price = closes[-1]
    atr_pct = atr_val / current_price * 100 if current_price > 0 else 2.0

    stop_distance_pct = -(atr_pct * atr_mult)

    if stop_distance_pct > min_stop_pct:
        stop_distance_pct = min_stop_pct
    if stop_distance_pct < max_stop_pct:
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
        self.scores = sorted(etf_scores, key=lambda x: x["score"], reverse=True)
        self.timing = timing_result
        self.portfolio = portfolio or {}

    def generate_plan(self, total_capital=None, max_single=0.25, max_industry=0.40):
        """生成明日操作计划
        total_capital: 总资产，默认从portfolio动态计算（现金+持仓市值）"""
        # v7.0: 从portfolio动态计算总资产
        if total_capital is None:
            cash = self.portfolio.get("_available_cash", 0)
            invested = sum(
                p.get("shares", 0) * p.get("current_price", p.get("cost", 0))
                for p in self.portfolio.values() if isinstance(p, dict) and not str(p.get("shares", "")).startswith("_")
            )
            total_capital = cash + invested
            if total_capital <= 0:
                total_capital = 4000  # fallback
        target_pos = self.timing["base_position"]
        target_amount = total_capital * target_pos
        current_invested = sum(
            p.get("shares", 0) * p.get("current_price", p.get("cost", 0))
            for p in self.portfolio.values() if isinstance(p, dict) and not str(p.get("shares", "")).startswith("_")
        )

        top_n = max(5, len(self.scores) // 2)
        top_etfs = self.scores[:top_n]

        buy_list = []
        sell_list = []
        hold_list = []

        # 卖出判断
        for code, pos in self.portfolio.items():
            if code.startswith("_"):
                continue
            score_data = next((s for s in self.scores if s["code"] == code), None)
            if not score_data:
                continue

            cost = pos.get("cost", 0)
            current_price = pos.get("current_price", cost)
            pnl_pct = (current_price / cost - 1) * 100 if cost > 0 else 0

            premium_pct = score_data.get("premium_info", {}).get("premium_pct") or 0

            # v7.0: 溢价独立卖出逻辑（溢价回归风险）
            premium_sell = False
            if premium_pct > 10:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"溢价{premium_pct:.1f}%极度危险"
                })
                premium_sell = True
            elif premium_pct > 7 and pnl_pct >= 0:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"溢价{premium_pct:.1f}%高位+盈利锁定"
                })
                premium_sell = True
            elif premium_pct > 5 and score_data["score"] < 55:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"溢价{premium_pct:.1f}%+评分偏低"
                })
                premium_sell = True

            if premium_sell:
                pass  # 已加入卖出列表，跳过后续判断
            # v7.0: 追踪止盈（从高点回撤>8%）
            elif pnl_pct > 0:
                r20d = score_data.get("returns", {}).get("r20d", 0)
                r5d = score_data.get("returns", {}).get("r5d", 0)
                # 若20日曾大涨但5日快速回撤，触发追踪止盈
                if r20d > 12 and r5d < -5:
                    sell_list.append({
                        "code": code, "name": pos.get("name", code), "action": "SELL",
                        "shares": pos["shares"], "price": current_price,
                        "pnl_pct": round(pnl_pct, 1),
                        "reason": f"追踪止盈(20日+{r20d:.1f}%→5日{r5d:.1f}%)"
                    })
            # v7.0: 接近止损+评分恶化 组合卖出
            elif -8 < pnl_pct <= -5 and score_data["score"] < 55:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"接近止损({pnl_pct:.1f}%)+评分偏低({score_data['score']})"
                })
            # 止损触发
            elif pnl_pct <= -8:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1), "reason": "止损触发"
                })
            # 评分下滑（D或以下）
            elif score_data["score"] < 42:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1), "reason": "评分下滑"
                })
            else:
                hold_list.append({
                    "code": code, "name": pos.get("name", code),
                    "score": score_data["score"], "grade": score_data.get("grade", ""),
                    "pnl_pct": round(pnl_pct, 1),
                    "premium_pct": round(premium_pct, 2) if premium_pct > 3 else None
                })

        # 买入建议
        available = total_capital - current_invested
        for s in self.scores:
            if len(buy_list) >= 3:
                break
            code = s["code"]
            if code in self.portfolio:
                continue
            if s["score"] < 65:
                continue

            # 按评分线性分配买入资金
            weight = s["score"] / 100 * max_single
            budget = min(available * weight, total_capital * max_single)
            price = s.get("price", 0)
            if price <= 0:
                continue
            shares = int(budget / price / 100) * 100
            if shares < 100:
                continue

            buy_list.append({
                "code": code, "name": s["name"],
                "action": "BUY", "shares": shares,
                "price": round(price, 4),
                "amount": round(shares * price, 2),
                "score": s["score"],
                "reasons": [s.get("grade", ""),
                            f"RSI={s.get('indicators', {}).get('rsi', 50):.0f}"]
            })

        return {
            "buy_list": buy_list,
            "sell_list": sell_list,
            "hold_list": hold_list,
            "target_position": round(target_pos, 2),
            "available": round(available, 2),
            "timing_advice": self.timing.get("advice", "")
        }
