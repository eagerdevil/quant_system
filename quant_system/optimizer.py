#!/usr/bin/env python
"""
量化系统 v7.1 — 自进化参数优化引擎
===================================
每月自动运行（或手动执行）
- 回测历史数据，评估因子+阈值+溢价惩罚的预测能力
- 时间加权IC（近期数据权重更高 → 自适应市场变化）
- 随机搜索 + 局部精化 → 最优参数
- 自动保存 + 进化日志 → 可追溯每次变化

用法:
  python optimizer.py              # 正常优化
  python optimizer.py --dry-run    # 预览不保存
  python optimizer.py --fast       # 快速模式（80次搜索）
"""
import json, math, sys, os, time, random, logging, traceback, io, re as _re
from datetime import datetime, timedelta
from collections import defaultdict

# 编码修复 (仅直接运行时)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger("optimizer")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_etf_kline, KEY_ETFS, USER_WATCHLIST
from quant_engine import (
    compute_indicators, score_factors,
    FACTOR_NAMES, FACTOR_MAX, DEFAULT_WEIGHTS,
    spearman_ic  # v7.3: 统一实现，替代本文件重复代码
)

TODAY = datetime.now().strftime("%Y%m%d")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, "factor_weights.json")
EVOLUTION_LOG = os.path.join(SCRIPT_DIR, "evolution_log.json")

# QDII ETF列表（这些ETF有跨境溢价，优化器模拟溢价惩罚时需要识别）
QDII_ETFS = {"513100", "159659", "513500", "159941", "513050", "159866", "513300"}

# ============================================================
# ============================================================
# 统计工具
# ============================================================
# spearman_ic 统一实现在 quant_engine.py，此处通过 import 使用


# time_weighted_ic 已删除 (v7.1起由 evaluate_params() 替代，v7.3清理)


# ============================================================
# v4.0: 扩展搜索空间
# ============================================================
# grade_thresholds / adx_trend_threshold 不参与 IC 目标函数
# （evaluate_params 只评估 factor_weights + premium_* 对评分的影响）。
# 若将它们随机搜索，赢家参数里这两个值只是"随机附带值"，会被当作优化
# 结果写进生产（quant_engine 的 OPTIMIZED_PARAMS）——即"假优化"。
# 因此固定为默认值，仅保留人工配置能力（load_current_config 读文件值）。
DEFAULT_GRADE_THRESHOLDS = {"A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42}
DEFAULT_ADX_THRESHOLD = 22.0


def random_params():
    """
    生成随机参数向量（v4.0: 权重+惩罚曲线）
    返回: {
        "factor_weights": {F1: 1.2, ...},
        "grade_thresholds": {A:78, B:65, C:55, D:42},  # 固定，不参与搜索
        "premium_steepness": 0.06,
        "premium_threshold": 2.0,
        "adx_trend_threshold": 22  # 固定，不参与搜索
    }
    """
    return {
        "factor_weights": {
            k: round(random.uniform(0.5, 2.0), 2) for k in FACTOR_NAMES
        },
        "grade_thresholds": dict(DEFAULT_GRADE_THRESHOLDS),
        "premium_steepness": round(random.uniform(0.04, 0.10), 3),
        "premium_threshold": round(random.uniform(1.0, 3.5), 1),
        "adx_trend_threshold": DEFAULT_ADX_THRESHOLD
    }


def perturb_params(base, scale=0.10):
    """在基准参数附近随机扰动（grade_thresholds/adx_trend_threshold 不扰动）"""
    new_weights = {}
    for k in FACTOR_NAMES:
        w = base["factor_weights"][k] * (1.0 + random.uniform(-scale, scale))
        new_weights[k] = round(max(0.3, min(2.5, w)), 2)

    return {
        "factor_weights": new_weights,
        "grade_thresholds": dict(base["grade_thresholds"]),  # 无IC目标，不扰动
        "premium_steepness": round(
            max(0.03, min(0.12,
                base["premium_steepness"] * (1.0 + random.uniform(-scale, scale))
            )), 3),
        "premium_threshold": round(
            max(0.5, min(4.0,
                base["premium_threshold"] + random.uniform(-0.5, 0.5)
            )), 1),
        "adx_trend_threshold": float(base["adx_trend_threshold"])  # 无IC目标，不扰动
    }


def compute_scores_with_params(params, precomputed):
    """
    v7.1: 使用全部参数计算最终得分（含溢价惩罚）
    与quant_engine的get_etf_score()对齐
    - factor_weights: 16因子权重 → 计算加权technical_score
    - premium_steepness + premium_threshold: 对QDII ETF施加溢价惩罚
    """
    weights = params["factor_weights"]
    steepness = params.get("premium_steepness", 0.07)
    premium_threshold_val = params.get("premium_threshold", 2.5)

    final_scores = []
    for pt in precomputed:
        # Step 1: 16因子加权得分（与quant_engine score_factors + get_etf_score一致）
        ws = sum(pt['factors'][k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
        max_w = sum(FACTOR_MAX[k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
        score = ws / max_w * 100 if max_w > 0 else sum(pt['factors'].values())

        # Step 2: 溢价惩罚 — 仅对QDII ETF，模拟quant_engine._apply_premium_penalty()
        code = pt.get('code', '')
        if code in QDII_ETFS:
            # 使用3%作为QDII ETF的默认溢价（A股QDII ETF历史中位水平）
            # 在3%溢价下，penalty取决于premium_threshold和steepness两个参数
            premium_pct = 3.0
            if premium_pct >= premium_threshold_val:
                excess = premium_pct - premium_threshold_val
                if premium_pct <= 5.0:
                    multiplier = 1.0 - excess * steepness
                elif premium_pct <= 8.0:
                    base_loss = (5.0 - premium_threshold_val) * steepness
                    multiplier = max(0.50, 1.0 - base_loss - (premium_pct - 5.0) * steepness * 1.3)
                else:
                    base_loss = (5.0 - premium_threshold_val) * steepness + 3.0 * steepness * 1.3
                    multiplier = max(0.45, 1.0 - base_loss - (premium_pct - 8.0) * steepness * 1.6)
                score = round(score * multiplier)

        final_scores.append(score)
    return final_scores


def evaluate_params(params, precomputed):
    """
    v7.1: 完整参数评估函数
    使用全部可IC评估的参数（因子权重+溢价参数）计算IC
    """
    scores = compute_scores_with_params(params, precomputed)
    rets = [pt['forward_return'] for pt in precomputed]

    # 时间加权：近期数据权重更高（半衰期=60交易日）
    n = len(precomputed)
    half_life = 60
    decay = math.log(2) / half_life
    time_weights = [math.exp(-decay * (n - 1 - i)) for i in range(n)]

    total_w = sum(time_weights)
    if total_w < 1e-10 or n < 10:
        return spearman_ic(scores, rets)

    # 加权重采样
    n_eff = min(n, 500)
    step = total_w / n_eff
    sampled_scores = []
    sampled_rets = []
    cumsum = 0.0
    j = 0
    for i in range(n_eff):
        target = (i + 0.5) * step
        while j < n - 1 and cumsum + time_weights[j] < target:
            cumsum += time_weights[j]
            j += 1
        idx = min(j, n - 1)
        sampled_scores.append(scores[idx])
        sampled_rets.append(rets[idx])

    return spearman_ic(sampled_scores, sampled_rets)


# ============================================================
# 数据准备（与v1.0相同但新增时间标记）
# ============================================================
def collect_backtest_data(etf_list, max_days=250, min_days=65):
    """采集回测所需历史数据并预计算因子"""
    logger.info(f"  采集 {len(etf_list)} 只ETF历史K线...")

    all_points = []
    etf_with_data = 0

    for i, code in enumerate(etf_list):
        name = KEY_ETFS.get(code, code)
        try:
            kline = fetch_etf_kline(code, days=max_days)
        except Exception as e:
            logger.info(f"    [{i+1}/{len(etf_list)}] {name}({code}) API错误: {e}")
            continue

        if not kline or len(kline) < min_days:
            continue

        closes = [k['close'] for k in kline]
        highs = [k['high'] for k in kline]
        lows = [k['low'] for k in kline]
        volumes = [k['volume'] for k in kline]
        n = len(kline)

        etf_points = 0
        for t in range(60, n - 16):
            try:
                ind = compute_indicators(
                    closes[:t + 1], highs[:t + 1],
                    lows[:t + 1], volumes[:t + 1]
                )
                factors = score_factors(ind)
                forward_ret = closes[t + 15] / closes[t] - 1.0  # v7.1: 15日forward

                if abs(forward_ret) > 0.50:
                    continue

                all_points.append({
                    'factors': factors,
                    'forward_return': forward_ret,
                    'date': kline[t].get('date', ''),  # 用于时间加权
                    'code': code
                })
                etf_points += 1
            except Exception:
                continue

        etf_with_data += 1
        logger.info(f"    [{i+1}/{len(etf_list)}] {name}({code}) ✓ {len(kline)}天 → {etf_points}点")

        if i < len(etf_list) - 1:
            time.sleep(0.4)

    # 按日期排序（时间加权需要）
    all_points.sort(key=lambda p: p.get('date', ''))

    logger.info(f"  合计: {etf_with_data}只ETF, {len(all_points)}个数据点")
    return all_points, {'etf_count': etf_with_data, 'point_count': len(all_points)}


# ============================================================
# 配置管理
# ============================================================
def load_current_config():
    """加载当前完整配置"""
    try:
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 确保所有字段存在（兼容旧版本）
        if "factor_weights" not in config:
            config["factor_weights"] = dict(DEFAULT_WEIGHTS)
        if "grade_thresholds" not in config:
            config["grade_thresholds"] = dict(DEFAULT_GRADE_THRESHOLDS)
        if "premium_steepness" not in config:
            config["premium_steepness"] = 0.07
        if "premium_threshold" not in config:
            config["premium_threshold"] = 2.5
        if "adx_trend_threshold" not in config:
            config["adx_trend_threshold"] = DEFAULT_ADX_THRESHOLD
        if "meta" not in config:
            config["meta"] = {"version": 0, "last_optimized": "", "ic_score": 0.0}
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "meta": {"version": 0, "last_optimized": "", "ic_score": 0.0},
            "factor_weights": dict(DEFAULT_WEIGHTS),
            "grade_thresholds": dict(DEFAULT_GRADE_THRESHOLDS),
            "premium_steepness": 0.07,
            "premium_threshold": 2.5,
            "adx_trend_threshold": DEFAULT_ADX_THRESHOLD
        }


def current_params_from_config(config):
    """从配置文件中提取当前参数"""
    return {
        "factor_weights": config.get("factor_weights", dict(DEFAULT_WEIGHTS)),
        "grade_thresholds": config.get("grade_thresholds", {}),
        "premium_steepness": config.get("premium_steepness", 0.07),
        "premium_threshold": config.get("premium_threshold", 2.5),
        "adx_trend_threshold": config.get("adx_trend_threshold", DEFAULT_ADX_THRESHOLD)
    }


def save_config(params, ic_score, old_config=None, reason=""):
    """保存最优参数到 factor_weights.json"""
    if old_config is None:
        old_config = load_current_config()

    old_version = old_config.get("meta", {}).get("version", 0)

    config = {
        "meta": {
            "version": old_version + 1,
            "last_optimized": TODAY,
            "ic_score": round(ic_score, 5),
            "optimization_reason": reason,
            "description": "自进化参数配置 — optimizer.py每月自动更新"
        },
        "factor_weights": params["factor_weights"],
        "grade_thresholds": params["grade_thresholds"],
        "premium_steepness": params["premium_steepness"],
        "premium_threshold": params["premium_threshold"],
        "adx_trend_threshold": params["adx_trend_threshold"]
    }

    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config


def load_evolution_log():
    """加载进化日志"""
    try:
        with open(EVOLUTION_LOG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": [], "summary": {"total_optimizations": 0, "total_ic_improvement": 0.0}}


def append_evolution_log(run_record):
    """追加一条进化记录"""
    log = load_evolution_log()
    log["runs"].append(run_record)
    # 只保留最近50次
    if len(log["runs"]) > 50:
        log["runs"] = log["runs"][-50:]
    log["summary"]["total_optimizations"] += 1
    log["summary"]["total_ic_improvement"] += run_record.get("ic_change", 0)
    with open(EVOLUTION_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ============================================================
# v8.0: 因子相关性分析
# ============================================================
def compute_factor_correlation(precomputed):
    """
    P0-3: 计算16因子两两相关性矩阵，检测多重共线性。

    对每对因子，跨所有数据点计算皮尔逊相关系数。
    标记 |r| > 0.7 的因子对为"高度相关"，建议合并或降权。

    返回: {
        "matrix": {F1: {F2: 0.85, ...}, ...},  # 16×16 下三角矩阵
        "high_corr_pairs": [(F1, F2, 0.85), ...],  # 按|r|降序排列
        "avg_abs_corr": 0.35,  # 平均绝对相关系数
        "n_high_pairs": 5,     # 高度相关因子对数量
        "warnings": ["⚠️ F1↔F2 r=0.85 高度正相关，建议合并或降低其中之一的权重"]
    }
    """
    logger.info(f"\n  [因子相关性分析] 计算16因子两两相关系数...")

    # 从预计算数据中提取因子矩阵 (n_points × 16)
    factor_matrix = {k: [] for k in FACTOR_NAMES}
    for pt in precomputed:
        factors = pt.get('factors', {})
        for k in FACTOR_NAMES:
            factor_matrix[k].append(factors.get(k, 5.0))

    n = len(precomputed)
    if n < 30:
        logger.info(f"  [WARNING] 数据点不足({n})，跳过低相关性分析")
        return {"matrix": {}, "high_corr_pairs": [], "avg_abs_corr": 0, "n_high_pairs": 0, "warnings": ["数据不足"]}

    # 计算两两皮尔逊相关系数
    import numpy as np
    high_pairs = []
    all_abs_corrs = []
    matrix = {}

    for i, f1 in enumerate(FACTOR_NAMES):
        matrix[f1] = {}
        arr1 = np.array(factor_matrix[f1], dtype=np.float64)
        for j, f2 in enumerate(FACTOR_NAMES):
            if i <= j:
                arr2 = np.array(factor_matrix[f2], dtype=np.float64)
                # 去除NaN和Inf
                mask = np.isfinite(arr1) & np.isfinite(arr2)
                if mask.sum() < 10:
                    r = 0.0
                else:
                    corr = np.corrcoef(arr1[mask], arr2[mask])
                    r = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0
                matrix[f1][f2] = round(r, 4)
                if i != j:
                    all_abs_corrs.append(abs(r))
                    if abs(r) > 0.70:
                        direction = "正" if r > 0 else "负"
                        high_pairs.append((f1, f2, round(r, 3), direction))

    # 按|r|降序排列
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # 生成警告
    warnings = []
    for f1, f2, r, direction in high_pairs:
        if abs(r) > 0.85:
            warnings.append(f"🚨 {f1}↔{f2} r={r:+.3f} 极度{direction}相关 → 强烈建议合并或大幅降权其一")
        elif abs(r) > 0.75:
            warnings.append(f"⚠️ {f1}↔{f2} r={r:+.3f} 高度{direction}相关 → 建议降低其中之一的权重")
        else:
            warnings.append(f"⚡ {f1}↔{f2} r={r:+.3f} {direction}相关 → 关注，可能重复计票")

    avg_abs_corr = float(np.mean(all_abs_corrs)) if all_abs_corrs else 0.0

    # 输出报告
    logger.info(f"  平均|相关系数|: {avg_abs_corr:.4f}  (0=完全独立, 1=完全冗余)")
    logger.info(f"  高度相关因子对 (|r|>0.7): {len(high_pairs)}对")
    for w in warnings:
        logger.info(f"  {w}")

    if len(high_pairs) == 0:
        logger.info(f"  ✓ 因子间独立性良好，无严重多重共线性")

    return {
        "matrix": matrix,
        "high_corr_pairs": [(f1, f2, r, d) for f1, f2, r, d in high_pairs],
        "avg_abs_corr": round(avg_abs_corr, 4),
        "n_high_pairs": len(high_pairs),
        "warnings": warnings,
        "n_points": n
    }


# ============================================================
# v8.0: 滚动窗口样本外验证 (Walk-Forward Cross Validation)
# ============================================================
def walk_forward_cv(precomputed, best_params, current_params, baseline_params,
                     n_windows=5, train_days=120, test_days=20, min_start=0):
    """
    P0-1: 滚动窗口 Walk-Forward 交叉验证。

    与单次70/30切分不同，WFCV在多个滚动窗口上评估OOS表现，
    更真实地反映策略在不同市场环境下的稳定性。

    方法:
    1. 将数据按时间排序，划分为 n_windows 个滚动窗口
    2. 每个窗口: train [t_i, t_i+train_days], test [t_i+train_days, t_i+train_days+test_days]
    3. 在每个窗口的测试集上计算IC
    4. 报告: 平均OOS IC ± std, 正IC窗口比例, 因子级稳定性

    v7.6 泄漏修复: 新增 min_start 参数 — 窗口测试段起点不得早于 min_start。
    此前WFCV窗口均匀分布在全程, 早窗口测试段落在主切分前70%训练集内,
    而优化参数正是在前70%上拟合的 → OOS分数虚高。现在测试段全部落在
    主切分测试集(参数未见过)之后。

    返回: {
        "windows": [{train_range, test_range, n_train, n_test, ic, ...}, ...],
        "mean_oos_ic": 0.035,
        "std_oos_ic": 0.018,
        "positive_rate": 0.80,       # 正IC窗口占比
        "significant_rate": 0.60,    # IC>0.01窗口占比
        "stability": "稳定" | "一般" | "不稳定",
        "factor_stability": {F1: {mean_ic, std_ic, positive_rate}, ...}
    }
    """
    logger.info(f"\n  [Walk-Forward CV] 滚动窗口样本外验证 (v8.0)")
    logger.info(f"  窗口数={n_windows}, 训练={train_days}天, 测试={test_days}天")

    n_total = len(precomputed)
    window_size = train_days + test_days

    if n_total < window_size * 2:
        logger.info(f"  [WARNING] 数据点不足({n_total} < {window_size*2})，回退到单窗口验证")
        n_windows = 1

    # v7.6 泄漏修复: 窗口测试段起点不得早于min_start(主切分测试集起点=参数未见过数据)
    min_start = max(int(min_start), 0)
    max_test_start = n_total - test_days
    if min_start >= max_test_start:
        logger.warning(f"  [WARNING] min_start={min_start} ≥ 可用测试空间{max_test_start}，WFCV跳过(数据不足)")
        return {
            "windows": [], "mean_oos_ic": None, "std_oos_ic": None,
            "positive_rate": None, "note": "WFCV跳过: 主测试集起点之后空间不足"
        }

    # 测试段起点均匀分布且全部 ≥ min_start（窗口间测试段不重叠，步进=test_days）
    if n_windows > 1:
        step = max(test_days, (max_test_start - min_start) // (n_windows - 1))
    else:
        step = 0

    windows = []
    oos_ics_best = []
    oos_ics_current = []
    oos_ics_baseline = []

    # v8.0: 因子级OOS追踪 — 每个窗口的因子IC
    factor_oos_ics = {k: [] for k in FACTOR_NAMES}

    for w in range(n_windows):
        start = min(min_start + w * step, max_test_start)
        test_end = min(start + test_days, n_total)
        # 窗口训练段用于过拟合差距参照(ic_best_train)，可落在拟合数据内；
        # 窗口测试段必须完全在参数拟合数据之外（start >= min_start 已保证）
        train_start = max(0, start - train_days)
        if train_start >= start:
            continue  # 训练段为空(窗口起点为0), 无法计算过拟合差距, 跳过
        window_train = precomputed[train_start:start]
        window_test = precomputed[start:test_end]

        if len(window_test) < 10:
            continue  # 测试集太小

        train_date_range = (window_train[0].get('date','?'), window_train[-1].get('date','?'))
        test_date_range = (window_test[0].get('date','?'), window_test[-1].get('date','?'))

        ic_best = evaluate_params(best_params, window_test)
        ic_current = evaluate_params(current_params, window_test)
        ic_baseline = evaluate_params(baseline_params, window_test)

        oos_ics_best.append(ic_best)
        oos_ics_current.append(ic_current)
        oos_ics_baseline.append(ic_baseline)

        # v8.0: 因子级IC — 对16个因子独立计算Spearman IC
        for f_name in FACTOR_NAMES:
            f_scores = [pt['factors'].get(f_name, 5.0) for pt in window_test]
            f_rets = [pt['forward_return'] for pt in window_test]
            if len(f_scores) >= 10:
                f_ic = spearman_ic(f_scores, f_rets)
                factor_oos_ics[f_name].append(f_ic)

        # v8.0: 窗口过拟合检测 (训练vs测试IC差距)
        ic_best_train = evaluate_params(best_params, window_train)
        overfit_gap = ic_best_train - ic_best

        windows.append({
            "window": w + 1,
            "train_range": f"{train_date_range[0]}~{train_date_range[1]}",
            "test_range": f"{test_date_range[0]}~{test_date_range[1]}",
            "n_train": len(window_train),
            "n_test": len(window_test),
            "ic_best_oos": round(ic_best, 5),
            "ic_current_oos": round(ic_current, 5),
            "ic_baseline_oos": round(ic_baseline, 5),
            "ic_best_train": round(ic_best_train, 5),
            "overfit_gap": round(overfit_gap, 5),
            "overfit_warning": overfit_gap > 0.01 and ic_best < 0
        })

    if not windows:
        logger.info(f"  [ERROR] 无法生成有效窗口")
        return {"error": "无法生成有效窗口", "windows": []}

    # 汇总统计
    import numpy as np
    mean_oos = float(np.mean(oos_ics_best))
    std_oos = float(np.std(oos_ics_best, ddof=1)) if len(oos_ics_best) > 1 else 0.0
    positive_rate = sum(1 for ic in oos_ics_best if ic > 0) / len(oos_ics_best)
    significant_rate = sum(1 for ic in oos_ics_best if ic > 0.01) / len(oos_ics_best)
    overfit_count = sum(1 for w in windows if w["overfit_warning"])

    # 稳定性评级
    if std_oos < 0.02 and positive_rate >= 0.8:
        stability = "✅ 稳定"
    elif std_oos < 0.04 and positive_rate >= 0.5:
        stability = "⚡ 一般"
    else:
        stability = "⚠️ 不稳定"

    t_stat = mean_oos / (std_oos / np.sqrt(len(oos_ics_best))) if std_oos > 0 else 999
    mean_current = float(np.mean(oos_ics_current))
    mean_baseline = float(np.mean(oos_ics_baseline))

    # v8.0: 因子稳定性分析
    factor_stability = {}
    for f_name in FACTOR_NAMES:
        f_ics = factor_oos_ics[f_name]
        if len(f_ics) >= 2:
            f_mean = float(np.mean(f_ics))
            f_std = float(np.std(f_ics, ddof=1))
            f_pos = sum(1 for ic in f_ics if ic > 0) / len(f_ics)
            factor_stability[f_name] = {
                "mean_ic": round(f_mean, 4),
                "std_ic": round(f_std, 4),
                "positive_rate": round(f_pos, 2),
                "reliable": f_pos >= 0.6 and abs(f_mean) > 0.005
            }

    # 输出报告
    logger.info(f"\n  {'─'*60}")
    logger.info(f"  [WFCV 结果] 滚动窗口样本外验证")
    logger.info(f"  有效窗口: {len(windows)}/{n_windows}")
    logger.info(f"  {'指标':<20s} {'最优参数':>10s} {'当前参数':>10s} {'等权基线':>10s}")
    logger.info(f"  {'─'*52}")
    logger.info(f"  {'平均OOS IC':<20s} {mean_oos:>10.5f} {mean_current:>10.5f} {mean_baseline:>10.5f}")
    logger.info(f"  {'OOS IC标准差':<20s} {std_oos:>10.5f}")
    logger.info(f"  {'正IC窗口占比':<20s} {positive_rate:>10.0%}")
    logger.info(f"  {'显著正IC占比':<20s} {significant_rate:>10.0%}")
    logger.info(f"  {'t统计量':<20s} {t_stat:>10.2f}")
    logger.info(f"  {'过拟合窗口数':<20s} {overfit_count:>10d}")
    logger.info(f"  {'稳定性评级':<20s} {stability:>10s}")

    # 详细的窗口IC
    logger.info(f"\n  [各窗口OOS IC]")
    for w in windows:
        flag = " ⚠️过拟合" if w["overfit_warning"] else ""
        logger.info(f"  W{w['window']}: 训练{w['n_train']}点 → 测试{w['n_test']}点 | "
                    f"IC最优={w['ic_best_oos']:+.5f} IC当前={w['ic_current_oos']:+.5f} "
                    f"训练IC={w['ic_best_train']:+.5f} 过拟合gap={w['overfit_gap']:+.5f}{flag}")

    # v8.0: 不稳定的因子报告
    unreliable_factors = [k for k, v in factor_stability.items() if not v.get("reliable")]
    if unreliable_factors:
        logger.info(f"\n  [⚠️ 不稳定因子] (正IC率<60% 或 IC≈0)")
        for f_name in unreliable_factors[:5]:
            fs = factor_stability.get(f_name, {})
            logger.info(f"  {f_name}: mean_IC={fs.get('mean_ic',0):+.4f} "
                        f"std={fs.get('std_ic',0):.4f} pos_rate={fs.get('positive_rate',0):.0%}")

    return {
        "windows": windows,
        "n_windows": len(windows),
        "mean_oos_ic": round(mean_oos, 5),
        "std_oos_ic": round(std_oos, 5),
        "mean_oos_ic_current": round(mean_current, 5),
        "mean_oos_ic_baseline": round(mean_baseline, 5),
        "positive_rate": round(positive_rate, 3),
        "significant_rate": round(significant_rate, 3),
        "t_statistic": round(t_stat, 2),
        "overfit_windows": overfit_count,
        "stability": stability,
        "factor_stability": factor_stability,
        "unreliable_factors": unreliable_factors
    }


# ============================================================
# 主流程
# ============================================================
def main():
    dry_run = "--dry-run" in sys.argv
    fast_mode = "--fast" in sys.argv

    n_random = 80 if fast_mode else 200
    n_local = 15 if fast_mode else 35

    logger.info("=" * 60)
    logger.info(f"  [量化系统 v7.1] 自进化参数优化引擎")
    logger.info(f"  [时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  [模式] {'预览(不保存)' if dry_run else ('快速' if fast_mode else '标准')}")
    logger.info(f"  [搜索范围] 16因子权重 + 溢价惩罚曲线(IC评估) + 等级阈值 + ADX(随机搜索)")
    logger.info("=" * 60)

    # Load current config
    old_config = load_current_config()
    current_params = current_params_from_config(old_config)
    old_ic = old_config.get("meta", {}).get("ic_score", 0.0)
    logger.info(f"\n  当前配置版本: v{old_config['meta'].get('version', 0)}")
    logger.info(f"  上次优化IC: {old_ic}")

    # ================================================================
    # Phase 1: 采集数据 + 训练/测试集切分（v7.2: 样本外验证）
    # ================================================================
    logger.info(f"\n  [Phase 1/4] 采集历史数据...")
    all_etfs = list(dict.fromkeys(list(KEY_ETFS.keys()) + USER_WATCHLIST))
    precomputed, stats = collect_backtest_data(all_etfs)

    if stats['etf_count'] < 5:
        logger.info(f"\n  [ERROR] 有效ETF不足({stats['etf_count']}只)。退出。")
        sys.exit(1)
    if len(precomputed) < 100:
        logger.info(f"\n  [ERROR] 回测数据点不足({len(precomputed)}个)。退出。")
        sys.exit(1)

    # ================================================================
    # v8.0: 因子相关性分析（P0-3）
    # ================================================================
    factor_corr_result = compute_factor_correlation(precomputed)

    # v7.2: 按时间切分训练/测试集（时序数据不可随机打乱）
    # 训练集: 前70%时间 → 优化参数
    # 测试集: 后30%时间 → 验证过拟合
    split_idx = int(len(precomputed) * 0.70)
    train_data = precomputed[:split_idx]
    test_data = precomputed[split_idx:]
    train_dates = (train_data[0].get('date', '?'), train_data[-1].get('date', '?'))
    test_dates = (test_data[0].get('date', '?'), test_data[-1].get('date', '?'))
    logger.info(f"  [样本外验证] 训练集: {len(train_data)}点 ({train_dates[0]}~{train_dates[1]})")
    logger.info(f"  [样本外验证] 测试集: {len(test_data)}点 ({test_dates[0]}~{test_dates[1]})")
    if len(test_data) < 50:
        logger.info(f"\n  [WARNING] 测试集数据点不足({len(test_data)}个)，回退全量优化")
        train_data = precomputed
        test_data = []

    # ================================================================
    # Phase 2: 基线测试（在训练集上计算IC）
    # ================================================================
    logger.info(f"\n  [Phase 2/4] 基线测试 (训练集)...")

    baseline_params = {
        "factor_weights": dict(DEFAULT_WEIGHTS),
        "grade_thresholds": dict(DEFAULT_GRADE_THRESHOLDS),
        "premium_steepness": 0.07,
        "premium_threshold": 2.5,
        "adx_trend_threshold": DEFAULT_ADX_THRESHOLD
    }
    baseline_ic_train = evaluate_params(baseline_params, train_data)
    current_ic_train = evaluate_params(current_params, train_data)
    logger.info(f"  等权基线 IC (训练): {baseline_ic_train:.5f}")
    logger.info(f"  当前权重 IC (训练): {current_ic_train:.5f}")
    logger.info(f"  (使用时间加权: 近期数据权重更高，半衰期=60交易日)")

    # ================================================================
    # Phase 3: 随机搜索（仅在训练集上优化）
    # ================================================================
    logger.info(f"\n  [Phase 3/4] 随机搜索 ({n_random}次迭代, 仅在训练集)...")

    best_params = dict(current_params)
    best_ic_train = current_ic_train

    random.seed(int(TODAY) % 10000)

    milestone = max(10, n_random // 5)
    for i in range(n_random):
        params = random_params()
        ic = evaluate_params(params, train_data)
        if ic > best_ic_train:
            best_ic_train = ic
            best_params = params
        if (i + 1) % milestone == 0:
            logger.info(f"    [{i+1}/{n_random}] 最佳IC(训练): {best_ic_train:.5f}")

    logger.info(f"  随机搜索完成，最佳IC(训练): {best_ic_train:.5f}")

    # ================================================================
    # Phase 4: 局部精化（仅在训练集上）
    # ================================================================
    logger.info(f"\n  [Phase 4/4] 局部精化 ({n_local}次扰动, 仅在训练集)...")

    refine_best_ic_train = best_ic_train
    refine_best_params = best_params

    for i in range(n_local):
        for scale in [0.08, 0.20]:
            p = perturb_params(best_params, scale=scale)
            ic = evaluate_params(p, train_data)
            if ic > refine_best_ic_train:
                refine_best_ic_train = ic
                refine_best_params = p

    logger.info(f"  精化完成，最佳IC(训练): {refine_best_ic_train:.5f}")

    # ================================================================
    # v7.2: 样本外验证 — 在测试集上评估最优参数
    # ================================================================
    if test_data:
        refine_best_ic_test = evaluate_params(refine_best_params, test_data)
        current_ic_test = evaluate_params(current_params, test_data)
        baseline_ic_test = evaluate_params(baseline_params, test_data)
        logger.info(f"\n  [样本外验证] 测试集IC:")
        logger.info(f"  等权基线 IC (测试): {baseline_ic_test:.5f}")
        logger.info(f"  当前参数 IC (测试): {current_ic_test:.5f}")
        logger.info(f"  最优参数 IC (测试): {refine_best_ic_test:.5f}")
    else:
        refine_best_ic_test = refine_best_ic_train
        current_ic_test = current_ic_train
        baseline_ic_test = baseline_ic_train
        logger.info(f"\n  [样本外验证] 跳过（测试集不足50点）")

    # ================================================================
    # v8.0: 滚动窗口样本外验证 (P0-1)
    # ================================================================
    wfcv_result = walk_forward_cv(
        precomputed, refine_best_params, current_params, baseline_params,
        n_windows=5, train_days=120, test_days=20,
        min_start=split_idx  # v7.6: 泄漏修复 — 测试段不早于主切分测试集起点
    )

    # ================================================================
    # 决策：是否更新（v7.2: 综合训练+测试IC判断）
    # ================================================================
    # 使用训练集IC作为主决策指标，测试集IC作为过拟合检测
    ic_improvement_train = refine_best_ic_train - current_ic_train
    ic_improvement_test = refine_best_ic_test - current_ic_test

    logger.info(f"\n  {'─' * 50}")
    logger.info(f"  [优化结果] (v7.2 样本外验证)")
    logger.info(f"  基线IC:    训练={baseline_ic_train:.5f}  测试={baseline_ic_test:.5f}")
    logger.info(f"  当前参数IC: 训练={current_ic_train:.5f}  测试={current_ic_test:.5f}")
    logger.info(f"  最优参数IC: 训练={refine_best_ic_train:.5f}  测试={refine_best_ic_test:.5f}")
    logger.info(f"  IC提升:    训练={ic_improvement_train:+.5f}  测试={ic_improvement_test:+.5f}")

    should_update = False
    final_params = dict(current_params)
    final_ic = current_ic_train
    reason = ""

    # v7.2: 过拟合检测
    overfitting = (refine_best_ic_train > current_ic_train + 0.003
                   and refine_best_ic_test < current_ic_test - 0.002)

    if overfitting:
        reason = (f"⚠️ 过拟合警告! 训练IC大幅提升({ic_improvement_train:+.5f})"
                  f"但测试IC下降({ic_improvement_test:+.5f})，拒绝更新")
    elif refine_best_ic_train > current_ic_train + 0.003:
        should_update = True
        final_params = refine_best_params
        final_ic = refine_best_ic_train
        reason = (f"IC显著提升 训练{ic_improvement_train:+.5f} 测试{ic_improvement_test:+.5f} ✓")
    elif refine_best_ic_train < -0.01 and current_ic_train < -0.01:
        should_update = True
        final_params = baseline_params
        final_ic = baseline_ic_train
        reason = "IC持续为负，回退等权基线"
    elif abs(ic_improvement_train) <= 0.003:
        reason = f"IC变化不显著(训练{ic_improvement_train:+.5f})，保持现有参数"
    else:
        reason = "现有参数已是最优"

    logger.info(f"  决策: {'✓ 更新' if should_update else '✗ 保持'} — {reason}")

    # ================================================================
    # 输出参数变化
    # ================================================================
    logger.info(f"\n  [因子权重变化]")
    logger.info(f"  {'因子':<16s} {'旧':>6s} {'新':>6s}  {'变化':>6s}")
    logger.info(f"  {'─' * 42}")
    for k in FACTOR_NAMES:
        old_w = current_params["factor_weights"].get(k, 1.0)
        new_w = final_params["factor_weights"].get(k, 1.0)
        diff = new_w - old_w
        arrow = "↑↑" if diff > 0.3 else ("↑" if diff > 0.05 else ("↓↓" if diff < -0.3 else ("↓" if diff < -0.05 else " ·")))
        logger.info(f"  {k:<16s} {old_w:6.2f} {new_w:6.2f}  {diff:+5.2f} {arrow}")

    # 阈值变化（固定参数：不参与IC优化，仅展示当前值，避免被误认为优化结果）
    logger.info(f"\n  [等级阈值] (固定参数，不参与IC优化)")
    for k in ["A_强烈买入", "B_买入", "C_观察", "D_谨慎"]:
        t = final_params["grade_thresholds"].get(k, 0)
        logger.info(f"  {k:<12s}: {t:3d}")

    # 溢价惩罚参数
    logger.info(f"\n  [溢价惩罚曲线]")
    logger.info(f"  陡峭度: {current_params['premium_steepness']:.3f} → {final_params['premium_steepness']:.3f}")
    logger.info(f"  阈值:   {current_params['premium_threshold']:.1f}% → {final_params['premium_threshold']:.1f}%")

    # ADX（固定参数：不参与IC优化，仅展示当前值）
    logger.info(f"\n  [ADX趋势阈值] (固定参数，不参与IC优化)")
    logger.info(f"  {final_params['adx_trend_threshold']}")

    # ================================================================
    # 保存 + 进化日志
    # ================================================================
    # v7.2: OOS (out-of-sample) IC fields added
    oos_available = len(test_data) >= 50
    evolution_record = {
        "date": TODAY,
        "timestamp": datetime.now().isoformat(),
        "version": old_config.get("meta", {}).get("version", 0) + 1,
        "baseline_ic": round(baseline_ic_train, 5),
        "current_ic": round(current_ic_train, 5),
        "best_ic": round(refine_best_ic_train, 5),
        "baseline_ic_oos": round(baseline_ic_test, 5) if oos_available else None,
        "current_ic_oos": round(current_ic_test, 5) if oos_available else None,
        "best_ic_oos": round(refine_best_ic_test, 5) if oos_available else None,
        "ic_change": round(ic_improvement_train, 5),
        "ic_change_oos": round(ic_improvement_test, 5) if oos_available else None,
        "overfitting_detected": overfitting,
        "updated": should_update,
        "reason": reason,
        "train_points": len(train_data),
        "test_points": len(test_data),
        # v8.0: WFCV 指标
        "wfcv": {
            "n_windows": wfcv_result.get("n_windows", 0),
            "mean_oos_ic": wfcv_result.get("mean_oos_ic"),
            "std_oos_ic": wfcv_result.get("std_oos_ic"),
            "positive_rate": wfcv_result.get("positive_rate"),
            "stability": wfcv_result.get("stability"),
            "overfit_windows": wfcv_result.get("overfit_windows", 0),
            "unreliable_factors": wfcv_result.get("unreliable_factors", [])[:5]
        },
        # v8.0: 因子相关性
        "factor_correlation": {
            "n_high_pairs": factor_corr_result.get("n_high_pairs", 0),
            "avg_abs_corr": factor_corr_result.get("avg_abs_corr", 0),
            "warnings": factor_corr_result.get("warnings", [])[:3]
        },
        "param_changes": {
            "weights_changed": sum(
                1 for k in FACTOR_NAMES
                if abs(final_params["factor_weights"].get(k, 1.0) -
                       current_params["factor_weights"].get(k, 1.0)) > 0.03
            ),
            # 等级阈值不再参与优化（固定参数），恒为0
            "thresholds_changed": 0,
            "premium_curve_changed": (
                abs(final_params["premium_steepness"] - current_params["premium_steepness"]) > 0.003
            )
        },
        "etf_count": stats['etf_count'],
        "point_count": len(precomputed)
    }

    append_evolution_log(evolution_record)

    if dry_run:
        logger.info(f"\n  [DRY-RUN] 未保存。要应用: python optimizer.py")
    elif should_update:
        save_config(final_params, final_ic, old_config, reason)
        logger.info(f"\n  ✓ 参数已更新！下次运行将使用新参数。")
        logger.info(f"  配置版本: v{old_config['meta'].get('version', 0) + 1}")
    else:
        logger.info(f"\n  ✓ 无需更新，当前参数保持不变。")

    # ================================================================
    # 进化趋势摘要
    # ================================================================
    log = load_evolution_log()
    runs = log.get("runs", [])
    if len(runs) >= 2:
        recent_ics = [r["best_ic"] for r in runs[-5:]]
        ic_trend = "上升" if len(recent_ics) >= 2 and recent_ics[-1] > recent_ics[0] else \
                   "下降" if len(recent_ics) >= 2 and recent_ics[-1] < recent_ics[0] else "平稳"
        logger.info(f"\n  [进化趋势] 最近{min(5, len(runs))}次优化IC趋势: {ic_trend}")
        logger.info(f"  累计优化次数: {log['summary']['total_optimizations']}")

    # v8.0: 因子相关性摘要
    if factor_corr_result.get("n_high_pairs", 0) > 0:
        logger.info(f"\n  [因子相关性] 发现{factor_corr_result['n_high_pairs']}对高度相关因子 "
                    f"(平均|r|={factor_corr_result['avg_abs_corr']:.3f})")
        for w in factor_corr_result.get("warnings", [])[:3]:
            logger.info(f"  {w}")
    else:
        logger.info(f"\n  [因子相关性] ✓ 独立性良好")

    # v8.0: WFCV 综合判定
    if not wfcv_result.get("error"):
        logger.info(f"\n  [WFCV 综合判定] {wfcv_result.get('stability', '?')} | "
                    f"平均OOS IC={wfcv_result.get('mean_oos_ic', 0):+.5f} | "
                    f"正IC率={wfcv_result.get('positive_rate', 0):.0%}")
        overfit_count = wfcv_result.get("overfit_windows", 0)
        if should_update and overfit_count >= 2:
            logger.info(f"  🚨 WFCV检测到{overfit_count}个窗口过拟合，但单次OOS验证通过，更新继续")
        elif should_update and "⚠️" in str(wfcv_result.get("stability", "")):
            logger.info(f"  ⚠️ WFCV稳定性较差({wfcv_result.get('stability')})，建议观察而非立即更新")

    # 输出JSON结果 (v8.0: 含WFCV + 因子相关性)
    result = {
        "date": TODAY,
        "baseline_ic_train": round(baseline_ic_train, 5),
        "current_ic_train": round(current_ic_train, 5),
        "best_ic_train": round(refine_best_ic_train, 5),
        "baseline_ic_test": round(baseline_ic_test, 5) if oos_available else None,
        "current_ic_test": round(current_ic_test, 5) if oos_available else None,
        "best_ic_test": round(refine_best_ic_test, 5) if oos_available else None,
        "overfitting": overfitting,
        "updated": should_update,
        "reason": reason,
        "train_points": len(train_data),
        "test_points": len(test_data),
        "etf_count": stats['etf_count'],
        "point_count": len(precomputed),
        "evolution_log_entries": len(runs),
        # v8.0
        "wfcv": {
            "n_windows": wfcv_result.get("n_windows", 0),
            "mean_oos_ic": wfcv_result.get("mean_oos_ic"),
            "std_oos_ic": wfcv_result.get("std_oos_ic"),
            "positive_rate": wfcv_result.get("positive_rate"),
            "stability": wfcv_result.get("stability"),
            "windows": wfcv_result.get("windows", [])
        },
        "factor_correlation": {
            "n_high_pairs": factor_corr_result.get("n_high_pairs", 0),
            "avg_abs_corr": factor_corr_result.get("avg_abs_corr", 0),
            "high_corr_pairs": factor_corr_result.get("high_corr_pairs", [])[:5],
            "warnings": factor_corr_result.get("warnings", [])
        }
    }
    result_path = os.path.join(SCRIPT_DIR, f"optimize_result_{TODAY}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Optimizer failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)
