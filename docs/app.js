/* 量化模拟盘监控 — 数据渲染逻辑
 * 数据源(相对路径, 与GitHub Pages子路径部署兼容):
 *   data/reports.json                     — manifest(报告列表, workflow每次运行自动生成)
 *   ../quant_system/report_paper_*.json   — 每日报告
 *   ../quant_system/portfolio_paper.json  — 账户状态(实时)
 */
const fmt = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? "--" : v.toLocaleString("zh-CN", { minimumFractionDigits: d, maximumFractionDigits: d });
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMoney = v => "¥" + fmt(v);
const pctCls = v => v > 0 ? "up" : (v < 0 ? "down" : "flat");
const pctSign = v => (v > 0 ? "+" : "") + fmt(v) + "%";
const dateFmt = d => d.length === 8 ? `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6)}` : d;

let manifest = null;
let navChart = null;

async function fetchJSON(paths) {
  if (!Array.isArray(paths)) paths = [paths];
  let lastErr;
  for (const p of paths) {
    try {
      const r = await fetch(p, { cache: "no-store" });
      if (r.ok) return r.json();
      lastErr = new Error(`HTTP ${r.status}: ${p}`);
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}
// 数据文件路径: 线上站点根=仓库docs/ (用 quant_system/ 前缀); 本地开发模式=仓库根 (用 ../quant_system/ 前缀)
const DATA_PATH = f => [`quant_system/${f}`, `../quant_system/${f}`];

async function load() {
  const btn = document.getElementById("btn-refresh");
  btn.disabled = true; btn.textContent = "🔄 加载中…";
  try {
    manifest = await fetchJSON("data/reports.json");
    const latest = manifest.reports && manifest.reports.length ? manifest.reports[manifest.reports.length - 1] : null;
    const report = latest ? await fetchJSON(DATA_PATH(latest.file)) : null;
    render(report, latest);
    document.getElementById("status-badge").textContent = "已连接 · 自动更新";
    document.getElementById("status-badge").className = "badge ok";
  } catch (e) {
    console.error(e);
    const badge = document.getElementById("status-badge");
    badge.textContent = "数据加载失败";
    badge.className = "badge warn";
    // 尝试直接读账户状态(首次部署、manifest尚未生成时兜底)
    try {
      const pf = await fetchJSON(DATA_PATH("portfolio_paper.json"));
      renderMinimal(pf);
    } catch (_) { /* 数据还不存在 */ }
  } finally {
    btn.disabled = false; btn.textContent = "🔄 刷新";
  }
}

function renderMinimal(pf) {
  document.getElementById("ov-initial").textContent = fmtMoney(pf._initial_capital);
  document.getElementById("ov-total-assets").textContent = fmtMoney(pf._available_cash);
  document.getElementById("ov-cash").textContent = fmtMoney(pf._available_cash);
  showEmpty("pending-empty", false);
  const tbody = document.querySelector("#tbl-pending tbody");
  (pf._pending || []).forEach(p => tbody.appendChild(pendingRow(p)));
  document.getElementById("pending-count").textContent = (pf._pending || []).length;
}

function showEmpty(id, show) { document.getElementById(id).hidden = !show; }

function render(report, latest) {
  if (!report) return;
  const acct = report.account || {};
  const pf = report.portfolio || {};
  const bench = report.benchmark || {};
  const d = report.date || (latest && latest.date) || "";

  document.getElementById("latest-date").textContent = `数据日期 ${dateFmt(d)}`;
  document.getElementById("ov-initial").textContent = fmtMoney(acct.initial_capital ?? pf.total_assets ?? 0);
  document.getElementById("ov-total-assets").textContent = fmtMoney(acct.total_assets ?? 0);
  const tr = document.getElementById("ov-total-return");
  tr.textContent = pctSign(acct.total_return_pct ?? 0);
  tr.className = "card-value " + pctCls(acct.total_return_pct ?? 0);
  document.getElementById("ov-cash").textContent = fmtMoney(pf.available_cash ?? 0);
  const posPct = pf.cash_ratio !== undefined ? 100 - pf.cash_ratio : 0;
  document.getElementById("ov-position").textContent = fmt(posPct, 1) + "%";
  document.getElementById("ov-daily-pnl").textContent = fmtMoney(pf.total_daily_pnl ?? 0);
  const dp = document.getElementById("ov-daily-pnl");
  dp.className = "card-value " + pctCls(pf.total_daily_pnl ?? 0);
  document.getElementById("ov-daily-pnl-pct").textContent = pctSign(pf.total_pnl_pct ?? 0);

  // 基准对比
  const ex = document.getElementById("ov-excess");
  ex.textContent = (bench.excess_return !== undefined ? pctSign(bench.excess_return * 100) : "--");
  ex.className = "card-value " + (bench.beat_benchmark ? "up" : "down");
  document.getElementById("ov-benchmark").textContent =
    `vs ${esc(bench.benchmark_name) || "基准"} ${pctSign((bench.benchmark_return ?? 0) * 100)}`;

  // 持仓
  const holds = pf.holdings || [];
  const holdBody = document.querySelector("#tbl-holdings tbody");
  holdBody.innerHTML = "";
  holds.forEach(h => {
    const pnl = h.pnl_pct ?? ((h.market_value - h.cost * h.shares) / (h.cost * h.shares) * 100);
    const row = document.createElement("tr");
    row.innerHTML = `<td>${esc(h.code)}</td><td>${esc(h.name)}</td><td>${fmt(h.shares, 0)}</td>` +
      `<td>${fmt(h.cost)}</td><td>${fmt(h.price)}</td><td>${fmtMoney(h.market_value)}</td>` +
      `<td class="${pctCls(pnl)}">${pctSign(pnl)}</td>`;
    holdBody.appendChild(row);
  });
  showEmpty("holdings-empty", holds.length === 0);

  // 待执行订单
  const pending = report.pending || [];
  const pendBody = document.querySelector("#tbl-pending tbody");
  pendBody.innerHTML = "";
  pending.forEach(p => pendBody.appendChild(pendingRow(p)));
  document.getElementById("pending-count").textContent = pending.length;
  showEmpty("pending-empty", pending.length === 0);

  // 成交记录
  const trades = report.executed_trades || [];
  const tradeBody = document.querySelector("#tbl-trades tbody");
  tradeBody.innerHTML = "";
  trades.slice().reverse().forEach(t => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${dateFmt(t.date || d)}</td><td>${esc(t.code)}</td><td>${esc(t.name)}</td>` +
      `<td><span class="tag ${t.action === "BUY" ? "buy" : "sell"}">${t.action === "BUY" ? "买入" : "卖出"}</span></td>` +
      `<td>${fmt(t.price)}</td><td>${fmt(t.shares, 0)}</td><td>${fmtMoney(t.amount)}</td>` +
      `<td class="${pctCls(t.pnl_pct ?? 0)}">${t.pnl_pct !== undefined ? pctSign(t.pnl_pct) : "--"}</td>` +
      `<td class="hint">${esc(t.reason)}</td>`;
    tradeBody.appendChild(row);
  });
  showEmpty("trades-empty", trades.length === 0);

  // 报告列表
  renderReportList();

  // 净值曲线
  renderChart();
}

function pendingRow(p) {
  const row = document.createElement("tr");
  const stale = p.signal_date && p.signal_date >= String(new Date().toISOString().slice(0, 10).replace(/-/g, ""));
  row.innerHTML = `<td>${esc(p.code)}</td><td>${esc(p.name)}</td>` +
    `<td><span class="tag ${p.action === "BUY" ? "buy" : "sell"}">${p.action === "BUY" ? "买入" : "卖出"}</span></td>` +
    `<td>${fmt(p.shares, 0)}</td><td>${dateFmt(p.signal_date || "")}</td>` +
    `<td><span class="tag wait">${p.retries > 0 ? `延期×${p.retries}` : "待执行"}</span></td>`;
  return row;
}

function renderReportList() {
  const list = document.getElementById("report-list");
  list.innerHTML = "";
  if (!manifest || !manifest.reports || !manifest.reports.length) { showEmpty("reports-empty", true); return; }
  showEmpty("reports-empty", false);
  manifest.reports.slice().reverse().forEach(r => {
    const item = document.createElement("div");
    item.className = "report-item";
    const retCls = pctCls(r.total_return_pct ?? 0);
    item.innerHTML = `
      <span class="r-date">${dateFmt(r.date)}</span>
      <span class="r-meta">总资产 ${fmtMoney(r.total_assets)}</span>
      <span class="r-ret ${retCls}">${pctSign(r.total_return_pct ?? 0)}</span>
      <span class="r-meta ${r.beat_benchmark ? "up" : "down"}">${r.beat_benchmark ? "跑赢" : "跑输"}基准 ${pctSign(r.excess_return ?? 0)}</span>
      <span class="hint">${String(r.updated_at || "").slice(0, 10)} 更新</span>`;
    item.onclick = () => toggleDetail(item, r);
    list.appendChild(item);
  });
}

async function toggleDetail(item, r) {
  const existing = item.nextElementSibling;
  if (existing && existing.classList.contains("report-detail")) { existing.remove(); return; }
  const old = document.querySelector(".report-detail");
  if (old) old.remove();
  try {
    const rep = await fetchJSON(DATA_PATH(r.file));
    const detail = document.createElement("div");
    detail.className = "report-detail";
    // 文本版报告 + 关键结构化信息
    let txt = rep.report || "";
    if (rep.scores && rep.scores.length) {
      const top = rep.scores.slice().sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, 10);
      txt += "\n\n【评分 TOP10】\n" + top.map(s => `${s.code} ${s.name} 评分${s.score} ${s.grade || ""}`).join("\n");
    }
    detail.textContent = txt;
    item.after(detail);
    detail.scrollIntoView({ block: "nearest" });
  } catch (e) {
    console.error("加载报告详情失败", e);
  }
}

function renderChart() {
  const canvas = document.getElementById("nav-chart");
  const wrap = canvas.parentElement;
  if (!manifest || !manifest.reports || manifest.reports.length < 1) { showEmpty("chart-empty", true); wrap.style.height = "auto"; return; }
  showEmpty("chart-empty", false);
  const pts = manifest.reports.map(r => ({ x: dateFmt(r.date), y: r.total_assets }));
  if (navChart) { navChart.destroy(); }
  navChart = new Chart(canvas, {
    type: "line",
    data: { labels: pts.map(p => p.x), datasets: [{ label: "总资产", data: pts.map(p => p.y), borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.12)", fill: true, tension: .3, pointRadius: 4, pointBackgroundColor: "#3b82f6" }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => "总资产 " + fmtMoney(c.parsed.y) } } },
      scales: {
        x: { ticks: { color: "#8b95ad" }, grid: { color: "rgba(42,52,80,.4)" } },
        y: { ticks: { color: "#8b95ad", callback: v => fmtMoney(v) }, grid: { color: "rgba(42,52,80,.4)" } }
      }
    }
  });
}

document.getElementById("btn-refresh").addEventListener("click", load);
load();
// 每5分钟自动刷新(页面本身就是打开时拉最新, 此逻辑保证常开页面也能更新)
setInterval(load, 5 * 60 * 1000);
