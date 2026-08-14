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

# ============================================================
# 因子权重系统（v2.0 新增）
# ============================================================
FACTOR_MAX = {
    "F1_趋势强度": 10, "F2_动量": 8, "F3_反转": 8,
    "F4_RSI": 8, "F5_均线偏离": 6, "F6_低波动": 6,
    "F7_成交量": 6, "F8_回调": 10, "F9_Sortino": 6,
    "F10_MaxDD": 6, "F11_布林带": 6, "F12_多周期": 6,
    "F13_均线排列": 4, "F14_长期": 4, "F15_夏普": 6,
    "F16_量价关系": 8,  # v3.0 新增：量价背离检测
    "F17_换手率分位": 6,  # v7.6 新增：换手率60日分位
    "F18_份额申赎": 8   # v7.6 新增：机构份额净流入/净流出TOP
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
# v7.2: 市场状态自适应权重调节器
# ============================================================
# 不同市场状态下，因子的预测能力不同：
#   趋势市 → 动量/趋势因子更有效
#   震荡市 → 反转/均值回归因子更有效
#   下跌市 → 低波动/防御因子更有效
#   危机市 → 极度防御，低波动/MaxDD优先

REGIME_WEIGHT_MODIFIERS = {
    "TREND_UP": {
        # 加强趋势+动量，弱化反转
        "F1_趋势强度": 1.35, "F2_动量": 1.25, "F8_回调": 1.20,
        "F12_多周期": 1.15, "F13_均线排列": 1.20, "F14_长期": 1.15,
        "F3_反转": 0.70, "F6_低波动": 0.80, "F11_布林带": 0.85,
    },
    "CHOPPY": {
        # 加强反转+均值回归，弱化趋势跟随
        "F3_反转": 1.50, "F4_RSI": 1.25, "F5_均线偏离": 1.20,
        "F11_布林带": 1.30, "F7_成交量": 1.15, "F6_低波动": 1.10,
        "F1_趋势强度": 0.65, "F2_动量": 0.70, "F14_长期": 0.80,
    },
    "TREND_DOWN": {
        # 防御为主：低波动+低回撤+量价背离
        "F6_低波动": 1.60, "F10_MaxDD": 1.40, "F9_Sortino": 1.25,
        "F16_量价关系": 1.30, "F4_RSI": 1.15, "F7_成交量": 1.10,
        "F1_趋势强度": 0.50, "F2_动量": 0.45, "F8_回调": 0.60,
        "F12_多周期": 0.60, "F13_均线排列": 0.50, "F14_长期": 0.55,
    },
    "CRISIS": {
        # 现金为王：极度防御
        "F6_低波动": 2.00, "F10_MaxDD": 1.80, "F9_Sortino": 1.50,
        "F16_量价关系": 1.50, "F11_布林带": 1.30, "F15_夏普": 1.25,
        "F1_趋势强度": 0.25, "F2_动量": 0.20, "F3_反转": 0.40,
        "F8_回调": 0.30, "F12_多周期": 0.30, "F13_均线排列": 0.25,
        "F14_长期": 0.30,
    },
}

def reload_weights():
    """
    v7.3: 公开权重重载函数 — 避免模块导入时的文件I/O副作用。

    因子权重文件被 optimizer.py 更新后，调用此函数重新加载。
    返回更新后的权重字典。
    """
    global CURRENT_WEIGHTS, WEIGHT_CONFIG
    CURRENT_WEIGHTS, WEIGHT_CONFIG = _load_weights()
    logger.info(f"[WEIGHTS] 重新加载: version={WEIGHT_CONFIG.get('meta', {}).get('version', 0)}")
    return CURRENT_WEIGHTS


def get_regime_weights(base_weights=None, regime=None):
    """
    v7.2: 根据市场状态调整因子权重。

    Args:
        base_weights: 基础权重字典，默认使用CURRENT_WEIGHTS
        regime: 市场状态字符串 (TREND_UP/CHOPPY/TREND_DOWN/CRISIS)

    Returns:
        调整后的权重字典
    """
    if base_weights is None:
        base_weights = CURRENT_WEIGHTS if CURRENT_WEIGHTS else DEFAULT_WEIGHTS

    if regime is None or regime not in REGIME_WEIGHT_MODIFIERS:
        return dict(base_weights)  # 无regime信息时保持不变

    modifiers = REGIME_WEIGHT_MODIFIERS[regime]
    adjusted = {}
    for k in FACTOR_NAMES:
        base = base_weights.get(k, 1.0)
        mod = modifiers.get(k, 1.0)
        adjusted[k] = round(base * mod, 2)
    return adjusted


# ============================================================
# v5.0: 系统运行参数（集中管理，避免魔法数字散落各处）
# ============================================================
SYSTEM_CONFIG = {
    # --- 大盘择时 ---
    "volume_active_threshold": 20000,    # 成交额阈值(亿) — v7.6: 仅作无历史数据时的降级线
    "volume_active_percentile": 70,      # v7.6: 成交额活跃分位数(当日成交额>近60日70%分位=活跃)
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
        # 完全平坦（avg_gain==0）时为中性50，纯上涨时为100
        return 50.0 if avg_gain == 0 else 100.0
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
    """Wilder平滑器 — 递归公式 r[k]=r[k-1]*(n-1)/n + x[k]（EMA变体）

    Wilder平滑本质是指数衰减(α=(n-1)/n)的递推，无法完全并行化。
    使用numpy预分配数组 + 标量递推，比纯Python列表快 ~3x。
    """
    if len(series) < n:
        return [float(sum(series))] if series else [0.0]

    s = np.asarray(series, dtype=np.float64)
    alpha = (n - 1.0) / n
    K = len(s) - n + 1

    result = np.empty(K, dtype=np.float64)
    result[0] = float(np.sum(s[:n]))
    for k in range(1, K):
        result[k] = result[k-1] * alpha + s[n + k - 1]
    return result.tolist()


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
    """布林带宽度 — numpy 向量化，返回标量。

    正常范围 0.02-0.15；数据不足或异常时返回 None 表示不可用。
    """
    if len(closes) < n:
        return None
    window = _to_np(closes[-n:])
    avg = np.mean(window)
    std = np.std(window, ddof=0)
    if avg == 0 or std == 0:
        return None
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
    # 窗口内每一天的前一收盘价（含窗口外前一天的收盘，作为第0天的前一收盘）
    prev_c = _to_np(closes[-(n+1):-1])

    window_high = float(np.max(h_arr))
    window_low = float(np.min(l_arr))

    # True Range for last n periods: TR_i = max(H-L, |H-C_prev|, |L-C_prev|)
    hl = h_arr - l_arr
    hc_val = np.abs(h_arr - prev_c)
    lc_val = np.abs(l_arr - prev_c)

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
        "buy_grade_min": "B_买入",     # 8/14决策: 原A_强烈买入导致震荡市长期空仓，放宽到B级让模拟盘能建仓(仓位仍受40%上限约束)
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
    if bbw is None:
        bbw = 0.05  # 数据不足时使用默认值
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
def compute_indicators(closes, highs, lows, volumes, turnover=None):
    """
    计算所有原始指标。
    输入: ETF的价格序列（按时间升序）
    输出: 指标字典，可直接传给 score_factors()
    turnover: 换手率序列(%, v7.6新增, F17因子用), None=数据缺失
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

    # v7.6: 换手率60日分位 (F17因子) — 数据不足或缺失时None, F17给中性分
    turnover_rank = None
    if turnover:
        tv = [float(t) for t in turnover if t is not None and float(t) > 0]
        if len(tv) >= 20:
            cur_t = tv[-1]
            turnover_rank = float(sum(1 for t in tv if t < cur_t) / len(tv) * 100)

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
        "turnover_pct_rank": turnover_rank,  # v7.6: F17因子
        "volume_divergence": detect_volume_price_divergence(closes, volumes)
    }


# ============================================================
# v7.2: 因子评分曲线定义 — 分段线性插值（替代离散桶）
# 每个因子定义 (breakpoints, scores) 对，通过 np.interp 连续映射
# 优势: 无边界跳变，评分平滑稳定
# ============================================================
_FACTOR_CURVES = {
    # (指标断点, 对应分数) — 所有因子满分归一化到其FACTOR_MAX后再截断
    "F2_动量": ([-20, -10, -3, 0, 10, 20, 30], [3, 3.5, 6.5, 8, 8, 5, 2]),
    "F3_反转": ([-15, -8, -5, -2, 0, 3, 8, 15], [8, 8, 6, 5, 4, 3, 1, 1]),
    "F4_RSI":   ([10, 20, 30, 40, 50, 58, 65, 72, 85], [0.5, 2, 5, 8, 8, 6, 4, 2, 0.5]),
    "F5_均线偏离": ([-15, -8, -5, -3, 0, 3, 8, 15], [2, 3.5, 5, 6, 6, 4, 2.5, 1]),
    "F6_低波动": ([3, 8, 15, 30, 45, 65, 85], [3, 5, 6, 6, 4, 2, 1]),
    "F7_成交量": ([0.3, 0.5, 0.70, 0.85, 1.20, 1.40, 1.80, 2.50], [2, 3, 4, 6, 6, 4, 2.5, 1]),
    "F9_Sortino": ([-2.5, -0.5, 0.3, 0.8, 1.5, 2.5, 4.0], [2, 3, 4, 5, 6, 6, 6]),
    "F10_MaxDD": ([0.0, 0.05, 0.10, 0.18, 0.25, 0.35, 0.55], [6, 6, 5, 4, 3, 2, 1]),
    "F11_布林带": ([0, 5, 15, 55, 75, 90, 100], [1, 5, 6, 4, 2, 1, 0.5]),
    "F14_长期":  ([-40, -10, 0, 10, 25, 50], [1, 2, 3, 4, 4, 4]),
    "F15_夏普":  ([-2.5, -0.5, 0.3, 0.8, 1.5, 2.5, 4.0], [2, 3, 4, 5, 6, 6, 6]),
}

def _continuous_score(value, factor_name):
    """通用连续评分：根据因子名称查找曲线，np.interp 映射"""
    if factor_name in _FACTOR_CURVES:
        bp, sc = _FACTOR_CURVES[factor_name]
        return float(np.clip(np.interp(value, bp, sc), min(sc), max(sc)))
    return None  # 此因子不使用连续映射


def score_factors(indicators, regime=None, flow_signal=0):
    """
    v7.2: 将原始指标转换为18个因子得分（连续映射 + 少数离散因子）。
    输入: compute_indicators() 的输出
    输出: {F1_趋势强度: 7.3, F2_动量: 6.8, ...}  # 现在返回浮点数
    regime: 市场状态(TREND_UP/CHOPPY/TREND_DOWN/CRISIS)，None=未知(默认震荡口径)
    flow_signal: v7.6 ETF份额申赎信号 +1=机构净流入TOP/-1=净流出TOP/0=无(F18因子)
    """
    factors = {}
    ind = indicators
    r5 = ind["r5d"]
    r20 = ind["r20d"]
    r60 = ind["r60d"]
    r120 = ind["r120d"]

    # F1: 趋势强度(0-10) — 基于MACD DIF/Hist幅度 + MA排列
    # v7.6: DIF/Hist按价格归一化(原tanh(dif*80)对1-3元低价ETF立即饱和,所有ETF≈9分无区分度)
    # 标定: DIF≈价格0.5% → tanh(1.5)=0.91(强趋势); ≈0.1% → tanh(0.3)=0.29(弱)
    price_n = max(float(ind.get("price", 1.0)), 0.01)
    dif_mag = np.tanh(ind["dif"] / price_n * 100 * 3)    # 归一化DIF% → [-1, 1]
    hist_mag = np.tanh(ind["hist"] / price_n * 100 * 6)  # 归一化Hist% → [-1, 1]
    f1_base = 5.0 + dif_mag * 3.5 + hist_mag * 1.5
    if ind["ma_alignment"] >= 2:
        f1_base += 0.8
    factors['F1_趋势强度'] = round(float(np.clip(f1_base, 0.5, 10.0)), 1)

    # F2-F15: 连续映射因子
    factors['F2_动量']     = round(_continuous_score(r20, "F2_动量"), 1)
    factors['F3_反转']     = round(_continuous_score(r5, "F3_反转"), 1)
    factors['F4_RSI']      = round(_continuous_score(ind["rsi"], "F4_RSI"), 1)
    factors['F5_均线偏离'] = round(_continuous_score(ind["pma5"], "F5_均线偏离"), 1)
    factors['F6_低波动']   = round(_continuous_score(ind["volatility"] * 100, "F6_低波动"), 1)
    factors['F7_成交量']   = round(_continuous_score(ind["vol_ratio_5d"], "F7_成交量"), 1)

    # F8: 回调质量(0-10) — v7.6: 按市场状态反转方向
    # TREND_UP(趋势市): 连涨=强势应加分, 连跌=动能衰减预警减分
    # 其余/未知(震荡/下跌/危机): 连跌=超跌回调加分(与F3反转互补), 连涨=追高风险减分
    if regime == "TREND_UP":
        f8 = 5.0 + ind["consecutive_up"] * 1.2 - ind["consecutive_down"] * 0.8
    else:
        f8 = 5.0 + ind["consecutive_down"] * 1.5 - ind["consecutive_up"] * 1.0
    factors['F8_回调'] = round(float(np.clip(f8, 1.0, 10.0)), 1)

    factors['F9_Sortino']  = round(_continuous_score(ind["sortino"], "F9_Sortino"), 1)
    factors['F10_MaxDD']   = round(_continuous_score(ind["max_drawdown"], "F10_MaxDD"), 1)
    factors['F11_布林带']  = round(_continuous_score(ind["bb_position"], "F11_布林带"), 1)

    # F12: 多周期收益(0-6) — 半连续：多周期方向 + 强度
    f12 = 2.0
    f12 += np.tanh(r5 + 0.3) * 1.5   # 短期
    f12 += np.tanh(r20 + 0.5) * 1.5  # 中期
    f12 += np.tanh(r60 + 0.5) * 1.0  # 长期
    factors['F12_多周期'] = round(float(np.clip(f12, 1.0, 6.0)), 1)

    # F13: 均线排列(0-4) — 本质离散，保持原样
    factors['F13_均线排列'] = ind["ma_alignment"]

    factors['F14_长期'] = round(_continuous_score(r120, "F14_长期"), 1)
    factors['F15_夏普'] = round(_continuous_score(ind["sharpe"], "F15_夏普"), 1)

    # F16: 量价关系(0-8) — 类型离散但有强度调制(v7.2改进)
    vol_div = ind.get("volume_divergence", {})
    div_type = vol_div.get("divergence_type")
    strength = vol_div.get("strength", 5)
    if div_type == "bullish":
        f16 = 5.0 + strength * 0.35
    elif div_type == "bearish":
        f16 = 5.0 - strength * 0.35
    elif div_type == "weak_rally":
        f16 = 5.0 - strength * 0.25
    elif div_type == "panic_sell":
        # 恐慌抛售是负面信号：强度越大分越低（与其他负面信号方向一致）
        f16 = 5.0 - strength * 0.2
    else:
        f16 = 5.0
    factors['F16_量价关系'] = round(float(np.clip(f16, 0.5, 8.0)), 1)

    # F17: 换手率分位(0-6) — v7.6新增: 当日换手率处于近60日分位
    # 高分位=放量资金关注(与F7短期量比互补, 中长期视角); 低分位=冷清
    tr = ind.get("turnover_pct_rank")
    if tr is None:
        f17 = 3.0  # 数据缺失, 中性
    else:
        f17 = 3.0 + (tr - 50) / 50 * 2.5  # 分位50→3.0, 100→5.5, 0→0.5
    factors['F17_换手率分位'] = round(float(np.clip(f17, 0.5, 6.0)), 1)

    # F18: 份额申赎(0-8) — v7.6新增: 机构净申购TOP加分/净赎回TOP减分
    # 份额增加=聪明钱申购, 是最可靠的底部/加速信号之一
    if flow_signal > 0:
        f18 = 6.8
    elif flow_signal < 0:
        f18 = 2.8
    else:
        f18 = 5.0
    factors['F18_份额申赎'] = round(float(f18), 1)

    return factors


# ============================================================
# 综合因子评分（ETF级别）— v2.0 支持自适应权重
# ============================================================
def _apply_premium_penalty(technical_score, premium_pct, history=None):
    """
    溢价惩罚函数 — 修正量化模型对QDII ETF溢价的盲区
    history: compute_etf_premium_history 返回值（可选）。
             有历史数据时，在绝对水平惩罚基础上叠加历史维度：
             1) 历史高分位（当前溢价处自身历史高位）→ 加罚
             2) 折价且历史低位 → 安全垫微加分
             3) 近10日均溢价明显高于全窗口（快速扩大）→ 加罚
    """
    if premium_pct is None:
        return technical_score, 1.0, "QDII溢价数据缺失，无法评估溢价风险"

    premium_threshold = OPTIMIZED_PARAMS.get("premium_threshold", 2.0)
    steepness = OPTIMIZED_PARAMS.get("premium_steepness", 0.07)

    if premium_pct < premium_threshold:
        multiplier = 1.0
    else:
        excess = premium_pct - premium_threshold
        if premium_pct <= 5.0:
            multiplier = 1.0 - excess * steepness
        elif premium_pct <= 8.0:
            base_loss = (5.0 - premium_threshold) * steepness
            multiplier = max(0.50, 1.0 - base_loss - (premium_pct - 5.0) * steepness * 1.3)
        else:
            base_loss = (5.0 - premium_threshold) * steepness + 3.0 * steepness * 1.3
            multiplier = max(0.45, 1.0 - base_loss - (premium_pct - 8.0) * steepness * 1.6)

    # === 历史溢价维度（v7.5，8/5新增）：分位数 + 趋势 ===
    hist_note = None
    if history and history.get("has_history"):
        percentile = history.get("percentile")
        trend_gap = history.get("trend_gap")
        median = history.get("median")
        trend_10d = history.get("trend_10d")

        # 1) 历史高分位：当前溢价处自身历史高位 → 回归风险大，加罚
        if percentile is not None and percentile >= 90:
            multiplier *= 0.94
            hist_note = f"🚨 历史分位{percentile:.0f}%（中枢{median}%），溢价处自身极高位"
        elif percentile is not None and percentile >= 75:
            multiplier *= 0.98
            hist_note = f"⚠️ 历史分位{percentile:.0f}%（中枢{median}%），偏高"
        # 2) 折价且历史低位：安全垫机会，微加分
        elif percentile is not None and percentile < 25 and premium_pct < 0:
            multiplier = min(1.02, multiplier * 1.02)
            hist_note = f"💚 历史分位{percentile:.0f}%（中枢{median}%），折价低位"

        # 3) 溢价快速扩大（近10日均溢价 - 全窗口 > 1.5pp）→ 加罚
        if trend_gap is not None and trend_gap > 1.5:
            multiplier *= 0.97
            gap_note = f"⚠️ 近10日溢价{trend_10d}%快速扩大（{trend_gap:+.1f}pp）"
            hist_note = f"{hist_note}；{gap_note}" if hist_note else gap_note

        multiplier = max(0.40, round(multiplier, 4))

    if premium_pct > 8:
        warning = f"🚨 溢价{premium_pct:.1f}%极度危险！市价远超净值，面临停牌+溢价回归双重风险"
    elif premium_pct > 5:
        warning = f"⚠️ 溢价{premium_pct:.1f}%偏高，买入即多付{premium_pct:.1f}%成本，需等溢价回落"
    elif premium_pct > 3:
        warning = f"⚡ 溢价{premium_pct:.1f}%，略高于安全线，关注溢价收敛趋势"
    else:
        warning = None

    if hist_note:
        warning = f"{warning}；{hist_note}" if warning else hist_note

    adjusted = round(technical_score * multiplier)
    return adjusted, multiplier, warning


def score_etf_comprehensive(code, name, closes, highs, lows, volumes,
                             north_flow_5d=None, industry_return=None,
                             weights=None, premium_pct=None, regime=None,
                             turnover=None):
    """
    18因子综合评分系统 + 溢价惩罚 + v7.2市场状态自适应权重

    Args:
        code, name: ETF/个股标识
        closes, highs, lows, volumes: OHLCV序列
        weights: 手动指定权重（优先级最高）
        premium_pct: QDII ETF溢价率
        regime: 市场状态 (TREND_UP/CHOPPY/TREND_DOWN/CRISIS) 用于自适应权重
        turnover: v7.6 换手率序列(F17因子), 个股/ETF均可传入
    """
    # v7.2: 权重优先级: 手动指定 > 状态自适应 > CURRENT_WEIGHTS > DEFAULT_WEIGHTS
    if weights is None:
        weights = get_regime_weights(CURRENT_WEIGHTS, regime)  # 从不返回None

    # 数据不足(<30根K线)时返回兜底评分（E_回避），避免 IndexError/NaN 传播击穿主流程
    if not closes or len(closes) < 30:
        return {
            "code": code, "name": name,
            "score": 0, "technical_score": 0, "max_score": 100,
            "grade": "E_回避", "price": 0.0,
            "premium_info": {
                "premium_pct": premium_pct, "penalty_multiplier": 1.0,
                "warning": "K线数据不足，无法评分", "score_penalty": 0
            },
            "indicators": {}, "returns": {}, "vs_ma": {},
            "factors": {}, "volume_divergence": {}
        }

    indicators = compute_indicators(closes, highs, lows, volumes, turnover)  # v7.6: 个股换手率F17
    factors = score_factors(indicators, regime)  # v7.6: F8按市场状态反转方向

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
# 横截面比较指标 (指标名, 权重, 方向)
# 方向: -1=低值更优(排名反转), +1=高值更优(排名同向)
#   RSI:   -1  高位过热→扣分，中低位(30-50)→加分  [有争议：强趋势中RSI 50-65可持续，但A股均值回归强]
#   r20d:  -1  短期涨幅过大→追高风险
#   r60d:  +1  中期趋势向上→看涨
#   vol:   -1  低波动ETF更安全
#   sortino:+1 风险调整收益越高越好
#   sharpe: +1 同上
#   maxdd: -1  回撤越小越好
#   ma_align:+1 多头排列越整齐越强
#   bb_pos: -1  布林带低位(接近下轨)→反弹概率大
#   cons_up:-1 连涨太久→获利盘压力
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


# ============================================================
# v7.3: 因子行业中性化引擎
# ============================================================

def _get_primary_industry(code):
    """获取ETF的主行业（用于分组中性化）"""
    industries = ETF_INDUSTRY_MAP.get(code, ["综合"])
    return industries[0] if industries else "综合"


def _industry_preserve_score(code):
    """
    不应被行业中性化的ETF: QDII(跟踪海外) + 宽基(综合行业)
    """
    if code.startswith(("513", "159659", "15966")):
        return True
    return _get_primary_industry(code) == "综合"


def compute_cross_sectional_adjustment_neutralized(raw_metrics_list):
    """
    v7.3: 行业内中性化横截面比较

    与旧版 compute_cross_sectional_adjustment 的区别:
    - 旧版: 全局z-score → 热门行业所有ETF都得高分
    - 新版: 行业内z-score → ETF只和同行比较，真正alpha才加分

    分组逻辑:
    - 行业ETF: 按主行业分组(电子/医药/银行...)，组内z-score
    - 宽基ETF(QDII/沪深300等): 保留全局z-score
    - 小行业(组内<3只): 合并到全局组
    """
    if len(raw_metrics_list) < 3:
        return {m["code"]: 0.0 for m in raw_metrics_list}

    codes = [m["code"] for m in raw_metrics_list]
    adjustments = {c: 0.0 for c in codes}

    # 分组: {行业: [metric_dicts]}
    industry_groups = {}
    global_etfs = []
    for m in raw_metrics_list:
        code = m["code"]
        if _industry_preserve_score(code):
            global_etfs.append(m)
        else:
            ind = _get_primary_industry(code)
            if ind not in industry_groups:
                industry_groups[ind] = []
            industry_groups[ind].append(m)

    for metric_name, weight, direction in CROSS_SECTION_METRICS:
        # === 行业内z-score ===
        for ind_name, group in industry_groups.items():
            if len(group) < 3:
                continue
            values, valid_codes = [], []
            for m in group:
                val = m.get(metric_name)
                if val is not None:
                    values.append(val)
                    valid_codes.append(m["code"])
            if len(values) < 3:
                continue

            arr = np.array(values, dtype=np.float64)
            mean, std = np.mean(arr), np.std(arr, ddof=0)
            if std < 1e-8:
                continue
            z = np.clip((arr - mean) / std, -2.5, 2.5)
            raw_adj = np.tanh(z) * weight * direction
            for i, code in enumerate(valid_codes):
                adjustments[code] += float(raw_adj[i])

        # === 全局z-score: 宽基 + QDII + 小组ETF ===
        global_vals, global_valid = [], []
        for m in global_etfs:
            val = m.get(metric_name)
            if val is not None:
                global_vals.append(val)
                global_valid.append(m["code"])
        for ind_name, group in industry_groups.items():
            if len(group) < 3:
                for m in group:
                    if m["code"] not in global_valid:
                        val = m.get(metric_name)
                        if val is not None:
                            global_vals.append(val)
                            global_valid.append(m["code"])

        if len(global_vals) >= 3:
            arr = np.array(global_vals, dtype=np.float64)
            mean, std = np.mean(arr), np.std(arr, ddof=0)
            if std > 1e-8:
                z = np.clip((arr - mean) / std, -2.5, 2.5)
                raw_adj = np.tanh(z) * weight * direction
                for i, code in enumerate(global_valid):
                    adjustments[code] += float(raw_adj[i])

    # 全局缩放到 [-8, +8]
    max_abs = max((abs(v) for v in adjustments.values()), default=1.0)
    if max_abs > 8.0:
        scale = 8.0 / max_abs
        for code in adjustments:
            adjustments[code] = round(adjustments[code] * scale, 1)
    else:
        for code in adjustments:
            adjustments[code] = round(adjustments[code], 1)

    return adjustments


def score_all_etfs_cross_sectional(etf_data_dict, weights=None, regime=None, use_neutralized=True,
                                    etf_flow_map=None):
    """
    v5.0: 批量评分 + 横截面比较融合
    v7.2: 支持regime自适应权重
    etf_flow_map: v7.6 {code: +1/-1/0} 机构份额申赎信号(F18因子), 无则None
    """
    if weights is None:
        weights = get_regime_weights(CURRENT_WEIGHTS, regime)  # 从不返回None

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
        turnovers = [k.get("turnover", 0) for k in kline]  # v7.6: F17因子

        indicators = compute_indicators(closes, highs, lows, volumes, turnovers)
        factors = score_factors(indicators, regime, (etf_flow_map or {}).get(code, 0))  # v7.6: F8/F18

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

    # Phase 2: 横截面比较 (v7.3: 行业中性化)
    if use_neutralized:
        cross_adjustments = compute_cross_sectional_adjustment_neutralized(raw_metrics)
    else:
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
    # 科技
    "562500": ["机械设备"],
    "512760": ["电子", "计算机"],
    "515070": ["计算机", "电子"],
    "159819": ["计算机", "电子"],
    "159995": ["电子"],
    "588200": ["电子"],
    "159516": ["电子"],
    "159732": ["电子"],
    "515880": ["通信"],
    "560800": ["计算机", "电子", "传媒"],
    # 金融周期
    "512880": ["非银金融"],
    "512800": ["银行"],
    "512400": ["有色金属"],
    "516780": ["有色金属"],
    "515220": ["采掘"],
    "516950": ["建筑装饰", "建筑材料"],
    "512200": ["房地产"],
    # 消费医药
    "512170": ["医药生物"],
    "159992": ["医药生物"],
    "159647": ["医药生物"],
    "512690": ["食品饮料"],
    "159928": ["食品饮料"],
    "159996": ["家用电器"],
    "159865": ["农林牧渔"],
    "515650": ["食品饮料"],           # 消费50ETF富国
    # 制造能源
    "512670": ["国防军工"],
    "512810": ["国防军工"],
    "159227": ["国防军工"],
    "515790": ["电气设备"],
    "159857": ["电气设备"],
    "516160": ["电气设备", "汽车"],
    "159183": ["电气设备", "汽车"],
    "159611": ["公用事业"],
    "159790": ["电气设备", "公用事业"],
    "159870": ["化工"],
    "516020": ["化工"],
    "159869": ["传媒"],
    # 电网设备
    "159326": ["电气设备"],
    "159320": ["电气设备"],
    "560390": ["电气设备"],
    # 策略/风格
    "512890": ["银行", "公用事业"],
    "510880": ["钢铁", "银行", "交通运输"],
    # 宽基
    "510300": ["综合"],
    "510500": ["综合"],
    "159915": ["综合"],
    "588000": ["电子", "计算机"],
    "588050": ["电子", "计算机"],
    "512100": ["综合"],
    "159845": ["综合"],
    # 黄金
    "518850": ["有色金属"],
    # QDII/跨境/全球
    "513100": ["综合"],
    "513500": ["综合"],
    "159659": ["综合"],
    "513180": ["综合"],
    "513130": ["综合"],
    "513050": ["综合"],
    "513520": ["综合"],
    "513800": ["综合"],               # 日本东证指数ETF南方
    "159529": ["综合"],               # 标普消费ETF景顺
    "159750": ["综合"],               # 港股科技50ETF招商
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

    def __init__(self, index_data, north_flow, total_vol, breadth, margin, volume_history=None,
                 bond_yield=None):
        self.hs300 = None
        self.hs300_highs = None
        self.hs300_lows = None
        if "000300" in index_data:
            data = index_data["000300"]["data"]
            self.hs300 = [d["close"] for d in data]
            self.hs300_highs = [d.get("high", d["close"] * 1.005) for d in data]
            self.hs300_lows = [d.get("low", d["close"] * 0.995) for d in data]

        self.north_flow = north_flow or []
        self.volume_history = volume_history or []  # v7.6: 全市场历史成交额序列
        self.bond_yield = bond_yield  # v7.6: 10Y国债收益率(流动性环境, S7信号)
        self.total_vol = total_vol or 0
        self.breadth = breadth or {}
        self.margin = margin or {}

    def calc_signals(self):
        """计算6个择时信号，返回 {signal_name: bool}"""
        signals = {}

        if self.hs300 and len(self.hs300) >= 20:
            ma20 = sma(self.hs300, 20)
            signals['S1_HS300_above_MA20'] = self.hs300[-1] > ma20
        # 数据不足时 S1 保持缺失（None），避免被当作"看跌"误触发强制压仓

        if self.hs300 and len(self.hs300) >= 61:
            ma60_now = sma(self.hs300, 60)
            ma60_prev = sma(self.hs300[:-1], 60)
            signals['S2_HS300_MA60_up'] = ma60_now > ma60_prev
        # 数据不足时 S2 保持缺失（None），避免被当作"看跌"误触发强制压仓

        # S3 (v7.6): 北向净买额2024-08-19起停披露 → 改用沪深股通成交活跃度(5日均 vs 前5日均)
        # 数据缺失时置None(中性不计)，不再像旧版判False压制仓位
        nf_5d = None
        flows = self.north_flow[-5:] if self.north_flow else []
        if flows and flows[0].get("deal_amt") is not None:
            amts = [f.get("deal_amt", 0) for f in flows]
            nf_5d = round(sum(amts) / len(amts), 1)  # 5日均成交额(亿)
            prev_flows = self.north_flow[-10:-5]
            if len(prev_flows) >= 3:
                prev_avg = sum(f.get("deal_amt", 0) for f in prev_flows) / len(prev_flows)
                signals['S3_NorthFlow_5d_positive'] = nf_5d > prev_avg
            else:
                signals['S3_NorthFlow_5d_positive'] = None  # 历史不足，中性
        else:
            signals['S3_NorthFlow_5d_positive'] = None  # 数据缺失，中性

        # S4 (v7.6): 成交额活跃度改动态分位 — 当日成交额 vs 近60日分布(70分位)
        # 旧固定2万亿死线在2026年A股常态1.2-1.6万亿下恒False，长期压制仓位
        vol_hist = getattr(self, "volume_history", None) or []
        if self.total_vol and len(vol_hist) >= 20:
            amts = [v.get("amount", 0) for v in vol_hist]
            percentile = float(sum(1 for a in amts if a < self.total_vol)) / len(amts) * 100
            signals['S4_Volume_active'] = percentile >= SYSTEM_CONFIG['volume_active_percentile']
        elif self.total_vol:
            signals['S4_Volume_active'] = self.total_vol > SYSTEM_CONFIG['volume_active_threshold']  # 降级
        else:
            signals['S4_Volume_active'] = None  # 数据缺失，中性

        ld = self.breadth.get("limit_down")
        # S5: 跌停家数缺失时置None(中性)，不误判为"跌停多"压仓
        signals['S5_LimitDown_low'] = (ld < SYSTEM_CONFIG['limit_down_danger']) if ld is not None else None

        # S6: 融资余额缺失时置None(中性)
        if self.margin:
            signals['S6_Margin_increasing'] = self.margin.get("change", 0) > 0
        else:
            signals['S6_Margin_increasing'] = None

        # S7 (v7.6): 10Y国债收益率 — 流动性环境信号(股债性价比降级版)
        # 收益率<1.5%=流动性宽松利好权益(+1); 1.5~2.2%中性; 缺失→None
        by = self.bond_yield
        by_val = by.get("yield") if isinstance(by, dict) else None
        if by_val is not None:
            signals['S7_Bond_low'] = by_val < 1.5
        else:
            signals['S7_Bond_low'] = None  # 数据缺失, 中性

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

        # 仅当沪深300数据完整（信号真实存在）时才应用强制压仓，避免数据缺失误伤
        hs300_below_ma20 = signals.get('S1_HS300_above_MA20') is False
        ma60_down = signals.get('S2_HS300_MA60_up') is False
        force_cap = hs300_below_ma20 and ma60_down

        if force_cap:
            base_position = min(base_position, 0.30)

        return {
            "bull_signals": bull_count,
            "total_signals": len(signals),
            "signal_detail": signals,
            "base_position": round(base_position, 2),
            "force_capped": force_cap,
            # v7.6: north_flow_5d 语义=沪深股通5日均成交额(亿)，缺失时为None
            "north_flow_5d": round(nf_5d, 1) if nf_5d is not None else None,
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

    # v7.3: numpy向量化 True Range 计算
    c = np.asarray(closes, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    l_arr = np.asarray(lows, dtype=np.float64)

    hl = h[1:] - l_arr[1:]
    hc = np.abs(h[1:] - c[:-1])
    lc = np.abs(l_arr[1:] - c[:-1])
    tr = np.maximum(np.maximum(hl, hc), lc)

    atr_val = float(np.mean(tr[-atr_period:]))
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

    # ============================================================
    # v7.3: 凯利公式仓位管理
    # ============================================================
    @staticmethod
    def _kelly_fraction(score, sortino, max_single=0.25):
        """
        凯利公式计算最优下注比例。

        f* = (p*b - (1-p)) / b

        参数估计:
        - p (胜率): 从评分映射 (78分≈55%, 90分≈65%)
        - b (盈亏比): 从Sortino映射 (>0.8≈1.5, >1.5≈2.0)

        使用半凯利(f*/2)以降低波动，上限max_single(25%)。
        """
        # 映射评分→胜率
        if score >= 90:
            p = 0.65
        elif score >= 82:
            p = 0.60
        elif score >= 74:
            p = 0.57
        elif score >= 68:
            p = 0.54
        elif score >= 65:
            p = 0.52
        else:
            return 0.0  # 无统计学优势，不下注

        # 映射Sortino→盈亏比
        if sortino > 2.0:
            b = 2.2
        elif sortino > 1.2:
            b = 1.8
        elif sortino > 0.6:
            b = 1.4
        elif sortino > 0.2:
            b = 1.15
        else:
            b = 1.05

        # 凯利公式
        edge = p * b - (1.0 - p)
        if edge <= 0:
            return 0.0
        f_star = edge / b

        # 半凯利（降低波动）+ 上限
        f_half = f_star / 2.0
        return round(min(f_half, max_single), 4)

    # ============================================================
    # v8.0 P2-8: 波动率加权仓位 (风险平价)
    # ============================================================
    @staticmethod
    def _volatility_adjustments(candidates):
        """
        计算反波动率仓位权重 (Inverse Volatility / Risk Parity)。

        逻辑: 低波动ETF应配更多仓位，高波动ETF应配更少。
        w_i = (1/vol_i) / sum(1/vol_j)

        Args:
            candidates: [{code, indicators: {volatility: ...}}, ...]

        Returns:
            {code: vol_adjustment}, 其中 adjustment > 1 表示加仓, < 1 表示减仓
        """
        if len(candidates) <= 1:
            return {c["code"]: 1.0 for c in candidates}

        # 提取年化波动率（小数形式，如0.25=25%）
        # 生产数据流中 indicators 只有 volatility_pct（百分数），需兼容两种键
        vols = {}
        for c in candidates:
            vol_raw = c.get("indicators", {}).get("volatility")
            if vol_raw is None:
                vol_raw = c.get("indicators", {}).get("volatility_pct", 25.0) / 100.0
            # 约束在合理范围
            vols[c["code"]] = max(0.08, min(0.60, float(vol_raw)))

        # 反波动率权重
        inv_vols = {code: 1.0 / v for code, v in vols.items()}
        total_inv = sum(inv_vols.values())
        raw_weights = {code: iv / total_inv for code, iv in inv_vols.items()}

        # 标准化为调整因子: 1.0 = 平均，>1 = 低波动加仓，<1 = 高波动减仓
        n = len(candidates)
        adjustments = {}
        for code in vols:
            raw = raw_weights[code]
            # 调整因子 = raw_weight / (1/n) = raw_weight * n
            adj = raw * n
            # 约束在0.5~1.8之间（不极端）
            adjustments[code] = round(max(0.5, min(1.8, adj)), 3)

        return adjustments


    def generate_plan(self, total_capital=None, max_single=0.25, max_industry=0.40):
        """生成明日操作计划
        total_capital: 总资产，默认从portfolio动态计算（现金+持仓市值）"""
        def _holding_value(p, k):
            """持仓市值；_开头的元数据键（_available_cash等）跳过"""
            if k.startswith("_"):
                return 0
            return p.get("shares", 0) * p.get("current_price", p.get("cost", 0))

        # v7.0: 从portfolio动态计算总资产
        if total_capital is None:
            cash = self.portfolio.get("_available_cash", 0)
            invested = sum(
                _holding_value(p, k)
                for k, p in self.portfolio.items() if isinstance(p, dict)
            )
            total_capital = cash + invested
            if total_capital <= 0:
                total_capital = 4000  # fallback
        target_pos = self.timing["base_position"]
        target_amount = total_capital * target_pos
        current_invested = sum(
            _holding_value(p, k)
            for k, p in self.portfolio.items() if isinstance(p, dict)
        )

        top_n = max(5, len(self.scores) // 2)
        top_etfs = self.scores[:top_n]

        buy_list = []
        sell_list = []
        hold_list = []

        # 卖出判断 (v7.6: 止损线跟随市场状态regime_stop_loss，CHOPPY=-5%/TREND_UP=-8%)
        stop_pct = self.timing.get("regime_stop_loss", -0.08) * 100  # 转百分数
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
            elif stop_pct < pnl_pct <= stop_pct + 3 and score_data["score"] < 55:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"接近止损({pnl_pct:.1f}%>{stop_pct:.0f}%)+评分偏低({score_data['score']})"
                })
            # 止损触发 (v7.6: 用regime止损，CHOPPY=-5%/TREND_UP=-8%)
            elif pnl_pct <= stop_pct:
                sell_list.append({
                    "code": code, "name": pos.get("name", code), "action": "SELL",
                    "shares": pos["shares"], "price": current_price,
                    "pnl_pct": round(pnl_pct, 1),
                    "reason": f"止损触发({stop_pct:.0f}%)"
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

        # 买入建议 (v8.0: 凯利公式 × 波动率加权)
        available = total_capital - current_invested

        # v3.1: 市场状态约束买入 — TREND_DOWN/CRISIS 禁止买入，CHOPPY 仅 A_强烈买入
        regime_grade_min = self.timing.get("regime_buy_grade_min", "B_买入")
        regime_blocks_buy = regime_grade_min is None
        grade_thresholds = OPTIMIZED_PARAMS.get("grade_thresholds", {})
        min_buy_score = grade_thresholds.get(regime_grade_min, 65) if regime_grade_min else float("inf")

        # v7.6: 行业集中度检查 — 单行业持仓市值不得超过 max_industry 比例
        def _industry_of(code):
            return (ETF_INDUSTRY_MAP.get(code) or [None])[0]

        industry_value = {}
        for k, v in self.portfolio.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            ind = _industry_of(k)
            val = v.get("shares", 0) * v.get("current_price", v.get("cost", 0))
            industry_value[ind] = industry_value.get(ind, 0) + val
        max_industry_value = max_industry * total_capital

        # v8.0: 先收集所有候选，计算波动率调整因子
        buy_candidates = []
        if not regime_blocks_buy:
            for s in self.scores:
                if len(buy_candidates) >= 6:  # 收集更多候选用于波动率比较
                    break
                if s["code"] in self.portfolio:
                    continue
                # v7.6: 行业集中度上限（max_industry参数此前从未生效）
                ind = _industry_of(s["code"])
                if ind and industry_value.get(ind, 0) >= max_industry_value:
                    continue
                if s["score"] < min_buy_score:
                    continue
                sortino_val = s.get("indicators", {}).get("sortino", 0.5)
                kelly_pct = self._kelly_fraction(s["score"], sortino_val, max_single)
                if kelly_pct <= 0:
                    continue
                buy_candidates.append({**s, "_kelly_pct": kelly_pct, "_sortino": sortino_val})

        # v8.0: 波动率调整因子
        vol_adjs = self._volatility_adjustments(buy_candidates)

        # P0修复: 跟踪本次计划已占用的资金，防止多只买入超支（原bug: available永不更新，3只各买50%可超支150%）
        spent = 0.0
        # v7.6: 启用 max_total_holdings（此前参数定义了从未执行）
        holdings_count = sum(1 for k, v in self.portfolio.items() if not k.startswith("_") and isinstance(v, dict))
        max_holdings = SYSTEM_CONFIG.get("max_total_holdings", 5)
        for s in buy_candidates:
            if len(buy_list) + holdings_count >= max_holdings:
                break
            code = s["code"]
            kelly_pct = s["_kelly_pct"]
            sortino_val = s["_sortino"]

            # v8.0: 凯利 × 波动率调整 = 最终仓位（cap在单只上限，防止vol_adj突破max_single）
            vol_adj = vol_adjs.get(code, 1.0)
            final_pct = min(kelly_pct * vol_adj, max_single)

            budget = total_capital * final_pct
            budget = min(budget, (available - spent) * 0.5)  # 单次不超过剩余可用资金50%
            # P0修复(8/14): target_amount此前只计算未使用，v8.0凯利重写丢了目标仓位上限，
            # 导致计划买入可超择时目标仓位(如震荡市40%目标却买68%)。买入上限=目标仓位-已持仓-本次已计划
            budget = min(budget, max(0.0, target_amount - current_invested - spent))
            price = s.get("price", 0)
            if price <= 0:
                continue
            shares = int(budget / price / 100) * 100
            if shares < 100:
                continue
            # 若剩余可用资金已不足1手，停止买入
            if (available - spent) < price * 100:
                break
            spent += shares * price

            # 构建理由
            reasons = [s.get("grade", "")]
            vol_label = "低波动+" if vol_adj > 1.15 else ("高波动-" if vol_adj < 0.85 else "")
            reasons.append(f"凯利{kelly_pct*100:.0f}%×{vol_label}波动率调整={final_pct*100:.1f}%仓位(Sortino={sortino_val:.2f})")

            buy_list.append({
                "code": code, "name": s["name"],
                "action": "BUY", "shares": shares,
                "price": round(price, 4),
                "amount": round(shares * price, 2),
                "score": s["score"],
                "kelly_pct": round(kelly_pct * 100, 1),
                "vol_adjustment": round(vol_adj, 2),  # v8.0
                "final_pct": round(final_pct * 100, 1),  # v8.0
                "reasons": reasons
            })

        return {
            "buy_list": buy_list,
            "sell_list": sell_list,
            "hold_list": hold_list,
            "target_position": round(target_pos, 2),
            "available": round(available, 2),
            "timing_advice": self.timing.get("advice", "")
        }


# ============================================================
# v7.3: 因子IC跟踪面板
# ============================================================

def compute_factor_ic_ranking(etf_data_dict, lookback=40, min_etfs=5):
    """
    计算16个因子的近期Rank IC（信息系数），用于日报展示。

    方法:
    - 对每只ETF，使用最近lookback天数据，计算16个因子的历史暴露序列
    - 每时间点计算因子暴露与未来15日收益的横截面Spearman秩相关
    - 取时间序列均值作为该因子的当前IC

    Args:
        etf_data_dict: {code: {name, kline: [...]}}
        lookback: 回看天数
        min_etfs: 最少ETF数量要求

    Returns:
        [{name, ic, status, bar}]  按IC从高到低排序
    """
    # 收集所有ETF的K线数据
    etf_list = []
    for code, edata in etf_data_dict.items():
        kline = edata.get("kline", [])
        if len(kline) < lookback + 25:
            continue
        closes = np.array([k["close"] for k in kline[-lookback-25:]], dtype=np.float64)
        highs = np.array([k["high"] for k in kline[-lookback-25:]], dtype=np.float64)
        lows = np.array([k["low"] for k in kline[-lookback-25:]], dtype=np.float64)
        volumes = np.array([k["volume"] for k in kline[-lookback-25:]], dtype=np.float64)
        turnovers = [k.get("turnover", 0) for k in kline[-lookback-25:]]  # v7.6: F17因子
        etf_list.append({"code": code, "name": edata.get("name", code),
                         "closes": closes, "highs": highs, "lows": lows, "volumes": volumes,
                         "turnovers": turnovers})

    if len(etf_list) < min_etfs:
        return []

    # 每个因子在每个时间点的暴露值
    factor_timeseries = {k: [] for k in FACTOR_NAMES}
    forward_returns = []

    n_etfs = len(etf_list)
    # 取每只ETF的最短长度
    min_len = min(len(e["closes"]) for e in etf_list)

    for t in range(30, min_len - 20):  # 从第30天开始，确保指标有意义
        frame_scores = []
        frame_rets = []
        for e in etf_list:
            c = e["closes"][:t+1].tolist()
            h = e["highs"][:t+1].tolist()
            l = e["lows"][:t+1].tolist()
            v = e["volumes"][:t+1].tolist()

            try:
                indicators = compute_indicators(c, h, l, v, e.get("turnovers", [])[:t+1])
                factors = score_factors(indicators)
                # 未来15日收益率（标准化后0-1范围，便于计算IC）
                fwd_ret = (e["closes"][t+15] / e["closes"][t] - 1.0) if t+15 < len(e["closes"]) else 0.0
                if abs(fwd_ret) < 0.50:  # 过滤异常值
                    frame_scores.append(factors)
                    frame_rets.append(fwd_ret)
            except Exception:
                continue

        if len(frame_scores) < min_etfs:
            continue

        # 对每个因子计算横截面Rank IC
        for fname in FACTOR_NAMES:
            f_vals = [fs[fname] for fs in frame_scores]
            if len(set(f_vals)) < 3:  # 因子值无变化
                continue
            # 计算Spearman秩相关
            ic = _spearman_rank_ic(f_vals, frame_rets)
            factor_timeseries[fname].append(ic)

        forward_returns.extend(frame_rets)

    # 取均值
    result = []
    for fname in FACTOR_NAMES:
        ic_list = factor_timeseries.get(fname, [])
        if len(ic_list) < 5:
            result.append({"name": fname, "ic": 0.0, "status": "数据不足", "bar": "", "direction": ""})
            continue

        mean_ic = round(float(np.mean(ic_list)), 4)

        # 判定状态
        if mean_ic > 0.04:
            status = "🟢 强预测"
        elif mean_ic > 0.015:
            status = "🟡 有效"
        elif mean_ic > -0.01:
            status = "⚪ 弱效"
        elif mean_ic > -0.03:
            status = "🟠 可能失效"
        else:
            status = "🔴 反向信号"

        bar_len = max(1, int(abs(mean_ic) * 100))
        bar = "█" * bar_len
        direction = "→" if mean_ic > 0 else "←"

        result.append({"name": fname, "ic": mean_ic, "status": status,
                       "bar": bar, "direction": direction})

    result.sort(key=lambda x: x["ic"], reverse=True)
    return result


def spearman_ic(x, y, min_samples=5):
    """
    Spearman秩相关系数（统一实现，供 optimizer / IC面板 共用）

    自动处理平局（average rank），返回 [-1, 1] 之间的值。
    """
    n = len(x)
    if n < min_samples:
        return 0.0

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    try:
        x_ranks = _rank(x)
        y_ranks = _rank(y)
        mx = sum(x_ranks) / n
        my = sum(y_ranks) / n
        num = sum((x_ranks[i] - mx) * (y_ranks[i] - my) for i in range(n))
        den_x = math.sqrt(sum((r - mx) ** 2 for r in x_ranks))
        den_y = math.sqrt(sum((r - my) ** 2 for r in y_ranks))
        return float(num / (den_x * den_y)) if den_x > 1e-10 and den_y > 1e-10 else 0.0
    except Exception:
        return 0.0


# 向后兼容别名
_spearman_rank_ic = spearman_ic


def format_factor_ic_section(ic_results):
    """格式化因子IC排名为文本板块"""
    if not ic_results:
        return "\n  [因子IC] 数据不足，无法计算"

    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [因子IC跟踪] v7.3 — 近40日Rank IC排名")
    lines.append(f"  {'─'*60}")
    lines.append(f"  {'因子':<16s} {'IC':>7s}  {'状态':<10s}  {'强度'}")
    lines.append(f"  {'─'*16} {'─'*7}  {'─'*10}  {'─'*30}")

    for f in ic_results:
        lines.append(f"  {f['name']:<16s} {f['direction']}{abs(f['ic']):.4f}  {f['status']:<10s}  {f['bar']}")

    # 汇总
    active = [f for f in ic_results if f["ic"] > 0.015]
    weak = [f for f in ic_results if -0.01 < f["ic"] <= 0.015]
    failing = [f for f in ic_results if f["ic"] <= -0.01]
    lines.append(f"\n  📊 汇总: {len(active)}个有效因子 | {len(weak)}个弱效 | {len(failing)}个可能失效")
    if failing:
        lines.append(f"  ⚠️ 失效因子: {', '.join(f['name'] for f in failing)} → 权重已自动衰减")

    return "\n".join(lines)
