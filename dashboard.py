#!/usr/bin/env python
"""
可视化面板生成器 v1.0
====================
将回测结果转换为交互式HTML仪表盘。

logger = logging.getLogger(__name__)
数据源: backtest_engine.py 输出的 JSON
输出: 单文件HTML, 使用 Chart.js CDN (零依赖)

用法:
  python dashboard.py                          # 读取最新回测结果
  python dashboard.py --input backtest_xxx.json
  python dashboard.py --output quant_dashboard.html
"""
import json, sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================
# HTML 模板
# ============================================================
HTML_HEADER = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略仪表盘 | Quant Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4e8;
    --text2: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --gold: #d2991d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
  }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px 20px; }
  h1 {
    font-size: 28px; font-weight: 700; margin-bottom: 4px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { color: var(--text2); font-size: 14px; margin-bottom: 28px; }
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px; margin-bottom: 28px;
  }
  .metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px;
  }
  .metric-card .label { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-card .value.positive { color: var(--green); }
  .metric-card .value.negative { color: var(--red); }
  .metric-card .value.neutral { color: var(--blue); }
  .chart-row {
    display: grid; grid-template-columns: 2fr 1fr;
    gap: 14px; margin-bottom: 14px;
  }
  @media (max-width: 900px) { .chart-row { grid-template-columns: 1fr; } }
  .chart-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px;
  }
  .chart-box.full { grid-column: 1 / -1; }
  .chart-box h3 { font-size: 15px; color: var(--text2); margin-bottom: 16px; }
  .chart-box canvas { width: 100% !important; max-height: 400px; }
  table {
    width: 100%; border-collapse: collapse; font-size: 13px;
  }
  th { text-align: left; color: var(--text2); font-weight: 500; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid #1e2030; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .tag-buy { color: var(--green); font-weight: 600; }
  .tag-sell { color: var(--red); font-weight: 600; }
  .heatmap { display: grid; grid-template-columns: repeat(auto-fill, minmax(70px,1fr)); gap: 4px; }
  .heat-cell { text-align: center; padding: 8px 4px; border-radius: 4px; font-size: 12px; }
  .heat-cell .month { color: var(--text2); font-size: 10px; }
  .heat-cell .ret { font-weight: 600; }
  .footer {
    text-align: center; color: var(--text2); font-size: 12px;
    padding: 20px; margin-top: 20px; border-top: 1px solid var(--border);
  }
</style>
</head>
<body>
<div class="container">
  <h1>📊 量化策略仪表盘</h1>
  <p class="subtitle">15因子多空模型 · 回测绩效报告 · 更新于 <span id="updateTime"></span></p>
  <div class="metrics-grid" id="metricsGrid"></div>
  <div class="chart-row">
    <div class="chart-box"><h3>📈 累计收益曲线 (组合 vs 沪深300)</h3><canvas id="equityChart"></canvas></div>
    <div class="chart-box"><h3>📉 回撤曲线</h3><canvas id="drawdownChart"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><h3>📊 日收益率分布</h3><canvas id="returnsChart"></canvas></div>
    <div class="chart-box"><h3>🗓 月度收益热力图</h3><div class="heatmap" id="heatmap"></div></div>
  </div>
  <div class="chart-row">
    <div class="chart-box full"><h3>📋 交易记录</h3>
      <div style="max-height:400px;overflow-y:auto;"><table id="tradesTable"></table></div>
    </div>
  </div>
  <div class="footer">
    © 2026 Quant System · 数据来源: 东方财富 · 免责声明: 仅供研究参考，不构成投资建议
  </div>
</div>
<script>
// ===== 数据注入 =====
const DATA = __DATA_PLACEHOLDER__;

// ===== 工具函数 =====
function fmtMoney(n) { return '¥' + n.toLocaleString('zh-CN', {maximumFractionDigits:0}); }
function fmtPct(n) { return (n>=0?'+':'') + n.toFixed(2) + '%'; }

// ===== 指标卡片 =====
(function renderMetrics() {
  const m = DATA.metrics;
  const cards = [
    {label:'累计收益', value:fmtPct(m.total_return), cls:m.total_return>=0?'positive':'negative'},
    {label:'年化收益', value:fmtPct(m.annual_return), cls:m.annual_return>=0?'positive':'negative'},
    {label:'夏普比率', value:m.sharpe.toFixed(2), cls:m.sharpe>=1?'positive':m.sharpe>=0?'neutral':'negative'},
    {label:'最大回撤', value:'-'+m.max_drawdown.toFixed(2)+'%', cls:'negative'},
    {label:'卡玛比率', value:m.calmar.toFixed(2), cls:m.calmar>=0.5?'positive':'neutral'},
    {label:'超额收益', value:fmtPct(m.alpha), cls:m.alpha>=0?'positive':'negative'},
    {label:'胜率', value:m.win_rate.toFixed(1)+'%', cls:m.win_rate>=50?'positive':'neutral'},
    {label:'交易次数', value:m.total_trades, cls:'neutral'},
  ];
  document.getElementById('metricsGrid').innerHTML = cards.map(c =>
    `<div class="metric-card"><div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div></div>`
  ).join('');
  document.getElementById('updateTime').textContent = DATA.config.end_date;
})();

// ===== 净值曲线 =====
(function equityCurve() {
  const eq = DATA.equity_curve;
  const labels = eq.map(e => e.date.slice(4));
  const navData = eq.map(e => e.nav);
  const benchData = eq.map(e => e.benchmark_nav);
  const initial = DATA.config.initial_capital;

  new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: '策略组合', data: navData.map(v => ((v/initial-1)*100)),
        borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.05)',
        borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true
      }, {
        label: '沪深300', data: benchData.map(v => ((v/initial-1)*100)),
        borderColor: '#8b949e', borderDash: [5,5],
        borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e', usePointStyle: true } } },
      scales: {
        x: { ticks: { color: '#484f58', maxTicksLimit: 15 }, grid: { color: '#1e2030' } },
        y: { ticks: { color: '#484f58', callback: v => v.toFixed(0)+'%' }, grid: { color: '#1e2030' } }
      },
      interaction: { intersect: false, mode: 'index' }
    }
  });
})();

// ===== 回撤曲线 =====
(function drawdown() {
  const eq = DATA.equity_curve;
  const labels = eq.map(e => e.date.slice(4));
  let peak = eq[0].nav;
  const ddData = eq.map(e => {
    if (e.nav > peak) peak = e.nav;
    return -((peak - e.nav) / peak * 100);
  });

  new Chart(document.getElementById('drawdownChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: '回撤', data: ddData,
        borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.1)',
        borderWidth: 2, pointRadius: 0, fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#484f58', maxTicksLimit: 12 }, grid: { color: '#1e2030' } },
        y: { ticks: { color: '#484f58', callback: v => v.toFixed(1)+'%' }, grid: { color: '#1e2030' }, max: 0 }
      },
      interaction: { intersect: false, mode: 'index' }
    }
  });
})();

// ===== 日收益分布 =====
(function dailyReturns() {
  const eq = DATA.equity_curve;
  const returns = [];
  for (let i=1; i<eq.length; i++) {
    returns.push((eq[i].nav / eq[i-1].nav - 1) * 100);
  }
  // 分桶
  const min = Math.floor(Math.min(...returns) * 10) / 10;
  const max = Math.ceil(Math.max(...returns) * 10) / 10;
  const step = Math.max(0.2, (max - min) / 30);
  const bins = [];
  for (let v=min; v<=max; v+=step) bins.push({x: v, y: 0});

  returns.forEach(r => {
    const idx = Math.floor((r - min) / step);
    if (idx >= 0 && idx < bins.length-1) bins[idx].y++;
  });

  new Chart(document.getElementById('returnsChart'), {
    type: 'bar',
    data: {
      labels: bins.map(b => b.x.toFixed(1)+'%'),
      datasets: [{
        label: '频次', data: bins.map(b => b.y),
        backgroundColor: bins.map(b => b.x >= 0 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)'),
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#484f58', maxTicksLimit: 20 }, grid: { color: '#1e2030' } },
        y: { ticks: { color: '#484f58' }, grid: { color: '#1e2030' } }
      }
    }
  });
})();

// ===== 月度热力图 =====
(function monthlyHeatmap() {
  const eq = DATA.equity_curve;
  const monthly = {};
  eq.forEach((e, i) => {
    const m = e.date.slice(0, 6);
    if (!monthly[m]) monthly[m] = {first: e.nav, last: e.nav};
    monthly[m].last = e.nav;
  });
  const months = Object.keys(monthly).sort();
  const grid = document.getElementById('heatmap');
  grid.innerHTML = months.map(m => {
    const ret = (monthly[m].last / monthly[m].first - 1) * 100;
    const intensity = Math.min(Math.abs(ret) / 10, 1);
    const bg = ret >= 0
      ? `rgba(63,185,80,${0.3 + intensity*0.7})`
      : `rgba(248,81,73,${0.3 + intensity*0.7})`;
    return `<div class="heat-cell" style="background:${bg}">
      <div class="month">${m.slice(4)}月</div>
      <div class="ret" style="color:${ret>=0?'#3fb950':'#f85149'}">${fmtPct(ret)}</div>
    </div>`;
  }).join('');
})();

// ===== 交易记录表 =====
(function tradesTable() {
  const trades = DATA.trades.slice(-80).reverse(); // 最近80条
  document.getElementById('tradesTable').innerHTML = `
    <thead><tr>
      <th>日期</th><th>方向</th><th>代码</th><th>名称</th>
      <th>价格</th><th>数量</th><th>金额</th><th>原因</th>
    </tr></thead>
    <tbody>${trades.map(t => `
      <tr>
        <td>${t.date}</td>
        <td class="${t.action==='BUY'?'tag-buy':'tag-sell'}">${t.action==='BUY'?'买入':'卖出'}</td>
        <td>${t.code}</td><td>${t.name}</td>
        <td>${t.price.toFixed(t.price<1?4:3)}</td>
        <td>${t.shares}</td>
        <td>${fmtMoney(t.amount)}</td>
        <td style="font-size:12px;color:var(--text2)">${t.reason}</td>
      </tr>`).join('')}
    </tbody>`;
})();
</script>
</body>
</html>"""

# ============================================================
# 生成器
# ============================================================
def generate_dashboard(backtest_json_path, output_path=None):
    """
    从回测JSON生成HTML仪表盘。

    参数:
        backtest_json_path: 回测结果JSON文件路径
        output_path: 输出HTML路径 (默认同目录下 quant_dashboard.html)
    """
    # 读取数据
    with open(backtest_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 注入数据到HTML
    html = HTML_HEADER.replace("__DATA_PLACEHOLDER__", json.dumps(data, ensure_ascii=False))

    # 写入
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "quant_dashboard.html"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size = os.path.getsize(output_path) / 1024
    logger.info(f"[Dashboard] 仪表盘已生成: {output_path} ({file_size:.0f} KB)")
    return output_path

# ============================================================
# 从 daily_runner 输出直接生成（无需中间文件）
# ============================================================
def generate_from_backtest_result(result, output_path=None):
    """
    直接从 backtest.run_backtest() 的返回值生成仪表盘。

    参数:
        result: backtest.run_backtest() 返回的字典
        output_path: 输出HTML路径
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "quant_dashboard.html"
        )

    data = {
        "generated": result["config"]["end_date"],
        "config": result["config"],
        "metrics": result["metrics"],
        "equity_curve": result["equity_curve"],
        "trades": result["trades"],
    }

    html = HTML_HEADER.replace("__DATA_PLACEHOLDER__", json.dumps(data, ensure_ascii=False))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"[Dashboard] 仪表盘已生成: {output_path}")
    return output_path

# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="量化仪表盘生成器")
    parser.add_argument("--input", default=None, help="回测JSON文件路径（默认自动查找最新）")
    parser.add_argument("--output", default=None, help="输出HTML路径")
    args = parser.parse_args()

    # 自动查找最新回测结果
    input_path = args.input
    if input_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        for f in os.listdir(script_dir):
            if f.startswith("backtest_") and f.endswith(".json"):
                candidates.append(os.path.join(script_dir, f))
        if not candidates:
            logger.info("[Dashboard] 错误: 未找到回测结果JSON，请先运行 backtest_engine.py")
            sys.exit(1)
        input_path = max(candidates, key=os.path.getmtime)
        logger.info(f"[Dashboard] 自动选择: {os.path.basename(input_path)}")

    output_path = generate_dashboard(input_path, args.output)
    logger.info(f"[Dashboard] ✅ 完成! 用浏览器打开 {output_path}")

if __name__ == "__main__":
    main()
