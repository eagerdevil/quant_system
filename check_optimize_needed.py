#!/usr/bin/env python
"""
量化系统 v4.0 — 优化触发条件检查
================================
在优化器之前运行，判断是否需要优化。

输出:
  exit 0 = 需要优化 (触发 optimizer.py)
  exit 1 = 错误
  exit 2 = 跳过 (条件不满足，本次不优化)

条件:
  1. 上次优化距今 > 30天 (每月保底)
  2. 当前权重IC < 0.01 (严重退化)
  3. 市场状态切换 (如 TREND_UP→CRISIS)
  4. 新增交易日 > 20天 且 IC下降 > 20%
"""

import json, sys, os, io, logging
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
sys.path.insert(0, SCRIPT_DIR)

TODAY = datetime.now().strftime("%Y%m%d")
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, "factor_weights.json")
EVOLUTION_LOG = os.path.join(SCRIPT_DIR, "evolution_log.json")


def load_weights_config():
    try:
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_evolution_log():
    try:
        with open(EVOLUTION_LOG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": []}


def get_current_regime():
    """快速获取当前市场状态（不拉取完整数据）"""
    try:
        from data_engine import fetch_index_daily
        from quant_engine import classify_market_regime

        # 只拉沪深300最近60天（轻量）
        data = fetch_index_daily("000300", 60)
        if not data:
            return None, "无法获取指数数据"

        closes = [d["close"] for d in data]
        highs = [d.get("high", c * 1.005) for d, c in zip(data, closes)]
        lows = [d.get("low", c * 0.995) for d, c in zip(data, closes)]

        result = classify_market_regime(closes, highs, lows)
        return result["regime"], result.get("regime_signals", [])
    except Exception as e:
        return None, f"状态检测异常: {e}"


def check_conditions():
    """检查四个触发条件，返回 (should_optimize: bool, reasons: [str])"""
    reasons = []
    config = load_weights_config()

    # === 条件0: 首次运行（无历史配置）===
    if config is None:
        return True, ["首次运行，无历史配置 → 必须优化"]

    meta = config.get("meta", {})
    last_opt = meta.get("last_optimized", "")
    current_ic = meta.get("ic_score", 0)
    version = meta.get("version", 0)

    # === 条件1: 30天保底 ===
    try:
        last_date = datetime.strptime(last_opt, "%Y%m%d") if last_opt else None
    except ValueError:
        last_date = None

    if last_date is None:
        reasons.append(f"条件1✅ 无历史优化记录")
    else:
        days_since = (datetime.now() - last_date).days
        if days_since >= 30:
            reasons.append(f"条件1✅ 距上次优化{days_since}天 (>30天，月度保底)")
        else:
            reasons.append(f"条件1❌ 距上次优化仅{days_since}天 (<30天)")

    # === 条件2: IC严重退化 ===
    if current_ic < 0.01:
        reasons.append(f"条件2✅ 当前IC={current_ic:.5f} < 0.01 (严重退化)")
    else:
        reasons.append(f"条件2❌ 当前IC={current_ic:.5f} >= 0.01 (尚可)")

    # === 条件3: 市场状态切换 ===
    try:
        current_regime, regime_signals = get_current_regime()
    except Exception:
        current_regime = None
        regime_signals = []

    if current_regime:
        # 从进化日志中找到上次优化时的市场状态
        log = load_evolution_log()
        runs = log.get("runs", [])
        last_regime = None
        for r in reversed(runs):
            if r.get("updated") and r.get("market_regime"):
                last_regime = r["market_regime"]
                break

        if last_regime and last_regime != current_regime:
            # 关键状态切换：上涨↔下跌，或进入危机
            critical_switches = [
                ("TREND_UP", "TREND_DOWN"),
                ("TREND_UP", "CRISIS"),
                ("TREND_DOWN", "CRISIS"),
                ("CHOPPY", "CRISIS"),
            ]
            is_critical = (last_regime, current_regime) in critical_switches or \
                          (current_regime, last_regime) in critical_switches

            if is_critical:
                reasons.append(f"条件3✅ 市场状态切换: {last_regime}→{current_regime} (关键切换)")
            else:
                reasons.append(f"条件3❌ 市场状态变化({last_regime}→{current_regime})，非关键切换")
        elif last_regime and last_regime == current_regime:
            reasons.append(f"条件3❌ 市场状态未变 (仍为{current_regime})")
        else:
            reasons.append(f"条件3⚠️ 无历史状态记录，当前状态: {current_regime}")
    else:
        reasons.append(f"条件3⚠️ 无法获取当前市场状态")

    # === 条件4: IC显著下降（需20天以上数据）===
    log = load_evolution_log()
    runs = log.get("runs", [])
    if len(runs) >= 2 and last_date:
        # 找上次IC > 当前IC*1.2 的记录
        recent_runs = [r for r in runs
                       if r.get("best_ic", 0) > current_ic * 1.2
                       and r.get("date", "") > last_opt]
        # 从上次优化至今新增的交易日（近似：自然日 * 0.7）
        new_days = int((datetime.now() - last_date).days * 0.7) if last_date else 0

        if new_days >= 20 and recent_runs:
            reasons.append(f"条件4✅ 新增~{new_days}交易日 + IC曾显著优于当前")
        elif new_days >= 20:
            reasons.append(f"条件4❌ 新增~{new_days}交易日但IC无显著下降")
        else:
            reasons.append(f"条件4❌ 新增~{new_days}交易日 (<20)")
    else:
        reasons.append(f"条件4❌ 进化日志不足 (需≥2次历史)")

    # === 综合判定: 任一条件满足即触发 ===
    triggers = [r for r in reasons if "✅" in r]
    should_run = len(triggers) > 0

    # 但条件1是保底条件，确保每月至少一次
    # 如果30天已过但其他条件都不满足，仍然触发
    if last_date and (datetime.now() - last_date).days >= 30:
        should_run = True

    return should_run, reasons


def main():
    print("=" * 60)
    print(f"  [量化系统 v4.0] 优化触发条件检查")
    print(f"  [时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    should_run, reasons = check_conditions()

    print(f"\n  检查结果:")
    for r in reasons:
        print(f"    {r}")

    conf = load_weights_config()
    if conf:
        meta = conf.get("meta", {})
        print(f"\n  当前状态: v{meta.get('version', '?')} | IC={meta.get('ic_score', '?')} | "
              f"上次优化={meta.get('last_optimized', '?')}")

    if should_run:
        print(f"\n  → 触发优化！运行 optimizer.py")
        # 输出标记供 workflow 读取
        logger.info("::set-output name=should_optimize::true")
        sys.exit(0)
    else:
        print(f"\n  → 跳过优化（条件不满足）")
        print(f"  提示: 下次触发检查将在下一个计划日")
        sys.exit(2)


if __name__ == "__main__":
    main()
