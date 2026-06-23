#!/usr/bin/env python
"""
量化系统 自适应参数优化引擎 v1.0
================================
每周日自动运行（也可手动执行）
- 回测历史数据，评估因子预测能力
- 随机搜索 + 局部精化 最优因子权重
- 自动保存到 factor_weights.json
- 下次 daily_runner.py 运行时自动生效

用法:
  python optimizer.py              # 正常优化
  python optimizer.py --dry-run    # 预览不保存
  python optimizer.py --fast       # 快速模式（50次搜索）
"""
import json, sys, os, math, random, io, time
from datetime import datetime
from collections import defaultdict

# 编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_etf_kline, KEY_ETFS, USER_WATCHLIST
from quant_engine import (
    compute_indicators, score_factors,
    FACTOR_NAMES, FACTOR_MAX, DEFAULT_WEIGHTS
)

TODAY = datetime.now().strftime("%Y%m%d")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, "factor_weights.json")

# ============================================================
# 统计工具
# ============================================================
def spearman_ic(x, y):
    """
    Spearman 秩相关系数 (Information Coefficient)
    衡量预测排名与实际收益排名的相关性
    正值 = 预测有效，负值 = 反向预测
    """
    n = len(x)
    if n < 10:
        return 0.0

    # 排名（等值取平均秩）
    def rank_values(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-indexed average
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    x_ranks = rank_values(x)
    y_ranks = rank_values(y)

    mean_xr = sum(x_ranks) / n
    mean_yr = sum(y_ranks) / n

    cov = sum((x_ranks[i] - mean_xr) * (y_ranks[i] - mean_yr) for i in range(n))
    std_x = math.sqrt(sum((r - mean_xr) ** 2 for r in x_ranks))
    std_y = math.sqrt(sum((r - mean_yr) ** 2 for r in y_ranks))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    return cov / (std_x * std_y)


# ============================================================
# 权重搜索
# ============================================================
def random_weights():
    """生成随机权重向量 [0.5, 2.0]"""
    return {k: round(random.uniform(0.5, 2.0), 2) for k in FACTOR_NAMES}


def perturb_weights(base, scale=0.12):
    """在基准权重附近随机扰动"""
    new = {}
    for k in FACTOR_NAMES:
        w = base[k] * (1.0 + random.uniform(-scale, scale))
        new[k] = round(max(0.3, min(2.5, w)), 2)
    return new


def compute_ic(weights, precomputed):
    """
    计算给定权重的 Information Coefficient
    precomputed: [{factors: {F1:5,...}, forward_return: 0.023}, ...]
    """
    scores = []
    rets = []
    for pt in precomputed:
        ws = sum(pt['factors'][k] * weights.get(k, 1.0) for k in FACTOR_NAMES)
        scores.append(ws)
        rets.append(pt['forward_return'])
    return spearman_ic(scores, rets)


# ============================================================
# 数据准备
# ============================================================
def collect_backtest_data(etf_list, max_days=250, min_days=65):
    """
    采集回测所需历史数据并预计算因子。

    返回:
        precomputed: [{factors, forward_return}, ...]
        stats: {etf_count, point_count, date_range}
    """
    print(f"  采集 {len(etf_list)} 只ETF历史K线...", file=sys.stderr)

    all_points = []
    etf_with_data = 0
    total_fetched = 0

    for i, code in enumerate(etf_list):
        name = KEY_ETFS.get(code, USER_WATCHLIST[USER_WATCHLIST.index(code)] if code in USER_WATCHLIST else code)

        try:
            kline = fetch_etf_kline(code, days=max_days)
        except Exception as e:
            print(f"    [{i+1}/{len(etf_list)}] {name}({code}) API错误: {e}", file=sys.stderr)
            continue

        if not kline or len(kline) < min_days:
            if kline:
                print(f"    [{i+1}/{len(etf_list)}] {name}({code}) 数据不足({len(kline)}天)，跳过", file=sys.stderr)
            else:
                print(f"    [{i+1}/{len(etf_list)}] {name}({code}) 无数据，跳过", file=sys.stderr)
            continue

        total_fetched += 1

        closes = [k['close'] for k in kline]
        highs = [k['high'] for k in kline]
        lows = [k['low'] for k in kline]
        volumes = [k['volume'] for k in kline]
        n = len(kline)

        etf_points = 0
        for t in range(60, n - 6):
            try:
                ind = compute_indicators(
                    closes[:t + 1], highs[:t + 1],
                    lows[:t + 1], volumes[:t + 1]
                )
                factors = score_factors(ind)
                forward_ret = closes[t + 5] / closes[t] - 1.0

                # 过滤极端值（可能是数据错误）
                if abs(forward_ret) > 0.50:
                    continue

                all_points.append({
                    'factors': factors,
                    'forward_return': forward_ret
                })
                etf_points += 1
            except Exception:
                continue

        etf_with_data += 1
        print(f"    [{i+1}/{len(etf_list)}] {name}({code}) ✓ {len(kline)}天 → {etf_points}个回测点", file=sys.stderr)

        # API限速
        if i < len(etf_list) - 1:
            time.sleep(0.4)

    print(f"  合计: {etf_with_data}只ETF, {len(all_points)}个数据点", file=sys.stderr)

    return all_points, {
        'etf_count': etf_with_data,
        'point_count': len(all_points),
        'fetched': total_fetched
    }


# ============================================================
# 权重保存
# ============================================================
def load_current_config():
    """加载当前配置（含meta信息）"""
    try:
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "meta": {"version": 0, "last_optimized": "", "ic_score": 0.0},
            "factor_weights": dict(DEFAULT_WEIGHTS),
            "grade_thresholds": {
                "A_强烈买入": 78, "B_买入": 65,
                "C_观察": 55, "D_谨慎": 42
            }
        }


def save_weights(weights, ic_score, old_config=None):
    """保存最优权重到 factor_weights.json"""
    if old_config is None:
        old_config = load_current_config()

    old_version = old_config.get("meta", {}).get("version", 0)

    config = {
        "meta": {
            "version": old_version + 1,
            "last_optimized": TODAY,
            "ic_score": round(ic_score, 5),
            "description": "因子权重配置文件 — 由optimizer.py每周日自动更新"
        },
        "factor_weights": weights,
        "grade_thresholds": old_config.get("grade_thresholds", {
            "A_强烈买入": 78, "B_买入": 65,
            "C_观察": 55, "D_谨慎": 42
        })
    }

    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n  权重已保存: {WEIGHTS_FILE}", file=sys.stderr)
    print(f"  版本: {config['meta']['version']}", file=sys.stderr)
    print(f"  IC: {config['meta']['ic_score']}", file=sys.stderr)


# ============================================================
# 主流程
# ============================================================
def main():
    dry_run = "--dry-run" in sys.argv
    fast_mode = "--fast" in sys.argv

    n_random = 50 if fast_mode else 150
    n_local = 15 if fast_mode else 30

    print("=" * 60, file=sys.stderr)
    print(f"  [量化系统] 自适应参数优化引擎", file=sys.stderr)
    print(f"  [时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"  [模式] {'预览(不保存)' if dry_run else ('快速' if fast_mode else '标准')}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load current config
    old_config = load_current_config()
    old_weights = old_config.get("factor_weights", dict(DEFAULT_WEIGHTS))
    old_ic = old_config.get("meta", {}).get("ic_score", 0.0)
    print(f"\n  当前配置版本: {old_config['meta'].get('version', 0)}", file=sys.stderr)
    print(f"  当前IC: {old_ic}", file=sys.stderr)

    # ================================================================
    # Phase 1: 采集数据 + 预计算因子
    # ================================================================
    print(f"\n  [Phase 1/4] 采集历史数据...", file=sys.stderr)
    all_etfs = list(dict.fromkeys(list(KEY_ETFS.keys()) + USER_WATCHLIST))
    precomputed, stats = collect_backtest_data(all_etfs)

    if stats['etf_count'] < 5:
        print(f"\n  [ERROR] 有效ETF不足({stats['etf_count']}只)，至少需要5只。退出。", file=sys.stderr)
        sys.exit(1)

    if len(precomputed) < 100:
        print(f"\n  [ERROR] 回测数据点不足({len(precomputed)}个)，至少需要100个。退出。", file=sys.stderr)
        sys.exit(1)

    # ================================================================
    # Phase 2: 基线测试
    # ================================================================
    print(f"\n  [Phase 2/4] 基线测试...", file=sys.stderr)

    baseline_ic = compute_ic(DEFAULT_WEIGHTS, precomputed)
    current_ic = compute_ic(old_weights, precomputed)
    print(f"  等权基线 IC: {baseline_ic:.5f}", file=sys.stderr)
    print(f"  当前权重 IC: {current_ic:.5f}", file=sys.stderr)

    # ================================================================
    # Phase 3: 随机搜索
    # ================================================================
    print(f"\n  [Phase 3/4] 随机搜索 ({n_random}次迭代)...", file=sys.stderr)

    best_weights = dict(old_weights)
    best_ic = current_ic

    random.seed(int(TODAY) % 10000)  # 每天不同种子，但同一天可复现

    milestone = max(10, n_random // 5)
    for i in range(n_random):
        w = random_weights()
        ic = compute_ic(w, precomputed)
        if ic > best_ic:
            best_ic = ic
            best_weights = w
        if (i + 1) % milestone == 0:
            print(f"    [{i+1}/{n_random}] 最佳IC: {best_ic:.5f}", file=sys.stderr)

    print(f"  随机搜索完成，最佳IC: {best_ic:.5f}", file=sys.stderr)

    # ================================================================
    # Phase 4: 局部精化
    # ================================================================
    print(f"\n  [Phase 4/4] 局部精化 ({n_local}次扰动)...", file=sys.stderr)

    refine_best_ic = best_ic
    refine_best_weights = dict(best_weights)

    for i in range(n_local):
        w = perturb_weights(best_weights, scale=0.10)
        ic = compute_ic(w, precomputed)
        if ic > refine_best_ic:
            refine_best_ic = ic
            refine_best_weights = w
        # 尝试更大扰动
        w2 = perturb_weights(best_weights, scale=0.25)
        ic2 = compute_ic(w2, precomputed)
        if ic2 > refine_best_ic:
            refine_best_ic = ic2
            refine_best_weights = w2

    print(f"  精化完成，最终IC: {refine_best_ic:.5f}", file=sys.stderr)

    # ================================================================
    # 决策：是否更新
    # ================================================================
    ic_improvement = refine_best_ic - current_ic

    print(f"\n  {'─' * 50}", file=sys.stderr)
    print(f"  [优化结果]", file=sys.stderr)
    print(f"  基线IC (等权):    {baseline_ic:.5f}", file=sys.stderr)
    print(f"  当前权重IC:       {current_ic:.5f}", file=sys.stderr)
    print(f"  最优权重IC:       {refine_best_ic:.5f}", file=sys.stderr)
    print(f"  IC提升:           {ic_improvement:+.5f}", file=sys.stderr)

    # 更新策略
    should_update = False
    final_weights = dict(old_weights)
    final_ic = current_ic
    reason = ""

    if refine_best_ic > current_ic + 0.003:
        should_update = True
        final_weights = refine_best_weights
        final_ic = refine_best_ic
        reason = f"IC显著提升 {ic_improvement:+.5f}"
    elif refine_best_ic < 0 and current_ic < 0:
        # 全都负相关 → 回退到等权
        should_update = True
        final_weights = dict(DEFAULT_WEIGHTS)
        final_ic = baseline_ic
        reason = "IC持续为负，回退等权"
    elif abs(ic_improvement) <= 0.003:
        reason = f"IC变化不显著({ic_improvement:+.5f})，保持现有权重"
    else:
        reason = "现有权重已是最优"

    print(f"  决策: {'✓ 更新' if should_update else '✗ 保持'} — {reason}", file=sys.stderr)

    # ================================================================
    # 输出因子权重变化
    # ================================================================
    print(f"\n  [因子权重变化]", file=sys.stderr)
    print(f"  {'因子':<16s} {'旧权重':>6s} {'新权重':>6s}  {'变化':>6s}", file=sys.stderr)
    print(f"  {'─' * 42}", file=sys.stderr)

    for k in FACTOR_NAMES:
        old_w = old_weights.get(k, 1.0)
        new_w = final_weights.get(k, 1.0)
        diff = new_w - old_w
        arrow = "↑↑" if diff > 0.3 else ("↑" if diff > 0.05 else ("↓↓" if diff < -0.3 else ("↓" if diff < -0.05 else " ·")))
        print(f"  {k:<16s} {old_w:6.2f} {new_w:6.2f}  {diff:+5.2f} {arrow}", file=sys.stderr)

    # ================================================================
    # 保存
    # ================================================================
    if dry_run:
        print(f"\n  [DRY-RUN] 未保存（预览模式）。要应用，请运行: python optimizer.py", file=sys.stderr)
    elif should_update:
        save_weights(final_weights, final_ic, old_config)
        print(f"\n  ✓ 优化完成！下次 daily_runner.py 将使用新权重。", file=sys.stderr)
    else:
        print(f"\n  ✓ 无需更新，当前权重保持不变。", file=sys.stderr)

    # ================================================================
    # 输出JSON结果（供GitHub Actions捕获）
    # ================================================================
    result = {
        "date": TODAY,
        "baseline_ic": round(baseline_ic, 5),
        "current_ic": round(current_ic, 5),
        "best_ic": round(refine_best_ic, 5),
        "updated": should_update,
        "reason": reason,
        "etf_count": stats['etf_count'],
        "point_count": len(precomputed)
    }
    result_path = os.path.join(SCRIPT_DIR, f"optimize_result_{TODAY}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  优化记录: {result_path}", file=sys.stderr)

    return 0 if not dry_run or should_update else 0


if __name__ == "__main__":
    sys.exit(main())
