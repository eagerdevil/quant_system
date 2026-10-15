#!/usr/bin/env python
"""
邮件报告 + 每月复盘 + 自适应参数优化
=====================================
- 每日收盘后生成HTML报告
- 发送到用户QQ邮箱
- 每月复盘并自适应调整因子权重
"""
import json, smtplib, os, math, urllib.request, urllib.parse, logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

def _load_smtp_user():
    """从环境变量→配置文件→默认值 读取QQ邮箱地址"""
    # 优先级1: 环境变量
    user = os.environ.get("QQMAIL_USER")
    if user:
        return user
    # 优先级2: email_config.json
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        user = config.get("QQMAIL_USER")
        if user:
            return user
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    # 优先级3: 默认值（向后兼容）
    logger.warning("QQMAIL_USER 未配置，使用默认邮箱")
    return "1239617073@qq.com"

SMTP_CONFIG = {
    "host": "smtp.qq.com",
    "port": 465,
    "sender_name": "A股量化决策系统"
}
# user 延迟加载（避免模块导入时文件I/O）
SMTP_CONFIG["user"] = _load_smtp_user()

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))

# Signal labels shared across HTML and WeChat report generators
SIGNAL_LABELS = {
    "S1_HS300_above_MA20": "沪深300在20日线上", "S2_HS300_MA60_up": "沪深300的60日线向上",
    "S3_NorthFlow_5d_positive": "北向资金5日净流入", "S4_Volume_active": "成交额>2万亿",
    "S5_LimitDown_low": "跌停<20家", "S6_Margin_increasing": "融资余额增加",
    "S7_Bond_low": "10Y国债收益率<1.5%"  # 9/22: 原先漏配, 邮件里显示原始signal名
}

def load_latest_report():
    """加载最新报告"""
    today = datetime.now().strftime("%Y%m%d")
    path = os.path.join(REPORT_DIR, f"report_{today}.json")
    if not os.path.exists(path):
        # 尝试昨天的
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        path = os.path.join(REPORT_DIR, f"report_{yesterday}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"[MAIL] 报告解析失败 {path}: {e}")
        return None

def _generate_speedview(report_data):
    """v8.0 P1-4: 30秒速览 — 邮件最顶部的精华摘要"""
    timing = report_data.get("timing", {})
    port = report_data.get("portfolio", {})
    benchmark = report_data.get("benchmark", {})
    scores = report_data.get("scores", [])

    # 1. 市场状态
    regime_name = timing.get("regime_name", "?")
    bull_signals = timing.get("bull_signals", 0)
    total_signals = timing.get("total_signals", 6)
    pos_pct = int(timing.get("base_position", 0) * 100)

    regime_colors = {
        "趋势上涨": "#00ff88", "震荡整理": "#ffa502",
        "趋势下跌": "#ff4757", "危机模式": "#ff0000"
    }
    regime_color = regime_colors.get(regime_name, "#ffa502")

    # 2. 持仓快照
    holdings = port.get("holdings", [])
    holding_alerts = []
    for h in holdings:
        pnl = h.get("pnl_pct", 0)
        daily = h.get("daily_pnl_pct", 0)
        alerts = []
        if pnl <= -7:
            alerts.append(f"⚠️接近止损({pnl:.1f}%)")
        elif pnl <= -5:
            alerts.append(f"⚠️亏损{pnl:.1f}%")
        if daily >= 3:
            alerts.append(f"🔥今日大涨{daily:+.1f}%")
        elif daily <= -3:
            alerts.append(f"📉今日大跌{daily:.1f}%")
        if alerts:
            holding_alerts.append({"name": h.get("name", "?"), "code": h.get("code", ""),
                                   "pnl_pct": pnl, "daily_pnl_pct": daily,
                                   "score": h.get("score", 0), "alerts": alerts})

    # 3. 今日关注（买入机会）
    buy_signals = []
    buy_list = report_data.get("plan", {}).get("buy_list", [])
    if buy_list:
        best = buy_list[0]
        buy_signals.append(f"买入机会: {best.get('name','?')} "
                          f"({best.get('code','?')}) {best.get('shares',0)}股")
    sell_list = report_data.get("plan", {}).get("sell_list", [])
    if sell_list:
        urgent = sell_list[0]
        buy_signals.append(f"⚠️ 卖出信号: {urgent.get('name','?')} — {urgent.get('reason','?')}")

    # 4. 构建HTML
    total_assets = port.get("total_assets", 0)
    daily_pnl = port.get("total_daily_pnl", 0)
    daily_color = "#00ff88" if daily_pnl >= 0 else "#ff4757"
    daily_icon = "📈" if daily_pnl >= 0 else "📉"
    # 9/6: 总盈亏(总资产-累计净投入) — 无流水时显示"—"
    total_pnl_all = port.get("total_pnl_all")
    alloc_color = "#00ff88" if (total_pnl_all or 0) >= 0 else "#ff4757"
    alloc_text = f"{total_pnl_all:+.0f}元" if total_pnl_all is not None else "—"

    html = '<div class="card" style="border: 2px solid #e94560;"><h2>⚡ 30秒速览</h2>'

    # 数字概览行
    html += '<div class="summary-grid">'
    html += f'<div class="summary-item"><div class="label">总资产</div><div class="big" style="color:#58a6ff">{total_assets:.0f}元</div></div>'
    # 9/30: 终身口径(2024/9入市至今) 提到速览首位之后, 用户要求看到"从2024年到目前"的总亏损
    lifetime_pnl_top = port.get("lifetime_pnl")
    if lifetime_pnl_top is not None:
        lt_color_top = "#00ff88" if lifetime_pnl_top >= 0 else "#ff4757"
        html += f'<div class="summary-item"><div class="label">累计总盈亏<br><small>2024/9入市至今</small></div><div class="big" style="color:{lt_color_top}">{lifetime_pnl_top:+.0f}元</div></div>'
    pnl_label = "模拟账户累计盈亏" if report_data.get("type") == "paper" else "今年盈亏<br><small>2026年</small>"
    html += f'<div class="summary-item"><div class="label">{pnl_label}</div><div class="big" style="color:{alloc_color}">{alloc_text}</div></div>'
    html += f'<div class="summary-item"><div class="label">{port.get("pnl_period_label", "今日盈亏")}</div><div class="big" style="color:{daily_color}">{daily_icon} {daily_pnl:+.0f}元</div></div>'
    html += f'<div class="summary-item"><div class="label">市场状态</div><div class="big" style="color:{regime_color}">{regime_name}</div></div>'
    html += '</div>'

    # 持仓状态行
    html += '<div style="margin-top:12px">'
    html += f'<b>📊 仓位:</b> 建议{pos_pct}% | 信号{bull_signals}/{total_signals} | '
    if benchmark and benchmark.get("beat_benchmark") is not None:
        beat = benchmark.get("beat_benchmark", False)
        excess = benchmark.get("excess_return", 0)
        html += f'基准: {"✅跑赢" if beat else "❌跑输"}(超额{excess:+.1%})'
    else:
        html += '基准: -'
    html += '</div>'

    # 持仓告警
    if holding_alerts:
        for ha in holding_alerts:
            alert_class = "summary-alert" if ha["pnl_pct"] < -5 else "summary-good"
            html += f'<div class="{alert_class}">'
            html += f'<b>{ha["name"]}</b>({ha["code"]}) | 评分{ha["score"]} | 累计{ha["pnl_pct"]:+.1f}% | 今日{ha["daily_pnl_pct"]:+.1f}%'
            for a in ha["alerts"]:
                html += f' | {a}'
            html += '</div>'

    # 今日关注
    if buy_signals:
        for bs in buy_signals:
            if "卖出" in bs:
                html += f'<div class="summary-alert">{bs}</div>'
            else:
                html += f'<div class="summary-good">{bs}</div>'
    elif not holding_alerts:
        quality_warnings = (report_data.get("research", {}).get("warnings", []) +
                            report_data.get("execution_warnings", []) + report_data.get("stale_codes", []))
        if report_data.get("timing", {}).get("data_insufficient") or quality_warnings:
            html += '<div class="summary-alert">⚠️ 数据或实验口径待核验，详见顶部运行状态；无交易不代表数据完整。</div>'
        else:
            html += '<div class="summary-good">✅ 今日无持仓价格或评分告警</div>'

    html += '</div>'
    return html


def _generate_stale_banner(report_data):
    """8/19加固: 数据滞后时生成醒目红色横幅（实盘/模拟盘报告通用）"""
    stale = report_data.get("stale_codes") or []
    if not stale:
        return ""
    return (
        '<div style="background:#3d0a0a;border:3px solid #ff4757;border-radius:8px;'
        'padding:16px 20px;margin:16px 0;font-size:16px;font-weight:bold;color:#ff6b6b;">'
        f'⚠️⚠️⚠️ 数据滞后告警：{", ".join(stale)} K线未更新到当日<br>'
        '<span style="font-weight:normal;font-size:13px;color:#ffa5a5;">'
        '今日盈亏/总资产为陈旧数据，不反映当日实际行情，请勿据此操作</span></div>'
    )


def generate_html_report(report_data):
    """Render the same compact daily view for both accounts."""
    if not report_data:
        return "<p>报告生成失败</p>"
    from daily_brief import render_daily_html
    return render_daily_html(report_data)


def send_email(report_html, password, report_date=None, subject=None):
    """发送邮件报告
    report_date: 报告日期（YYYYMMDD 或 YYYY-MM-DD），缺省时用今天
    subject: 邮件主题，缺省时用"A股量化日报 - {date}"（模拟盘传入自定义主题）
    """
    if not password:
        logger.warning("[MAIL] 未提供邮箱授权码，跳过发送")
        return False

    try:
        from email.header import Header
        from email.utils import formataddr

        msg = MIMEMultipart('alternative')
        if report_date:
            date_str = str(report_date).replace('-', '.')
        else:
            date_str = datetime.now().strftime('%Y.%m.%d')
        if subject:
            msg['Subject'] = Header(subject, 'utf-8')
        else:
            msg['Subject'] = Header(f"A股量化日报 - {date_str}", 'utf-8')
        msg['From'] = formataddr(('A股量化系统', SMTP_CONFIG['user']))
        msg['To'] = SMTP_CONFIG['user']

        msg.attach(MIMEText(report_html, 'html', 'utf-8'))

        with smtplib.SMTP_SSL(SMTP_CONFIG['host'], SMTP_CONFIG['port']) as server:
            server.login(SMTP_CONFIG['user'], password)
            server.sendmail(SMTP_CONFIG['user'], SMTP_CONFIG['user'], msg.as_string())

        logger.info(f"[MAIL] 报告已发送到 {SMTP_CONFIG['user']}")
        return True
    except Exception as e:
        logger.error(f"[MAIL] 发送失败: {e}")
        return False


def send_crash_report(error_msg, traceback_text, password):
    if not password:
        return False
    try:
        from email.header import Header
        from email.utils import formataddr
        now = datetime.now()
        body = "<html><body><h2>Quant System Error</h2><p>Time: " + now.strftime("%Y-%m-%d %H:%M:%S") + "</p><h3>Error:</h3><pre style='background:#fff3f3;padding:10px;border-left:4px solid #e74c3c;'>" + str(error_msg) + "</pre><h3>Traceback:</h3><pre style='background:#f8f8f8;padding:10px;font-size:11px;max-height:400px;overflow:auto;'>" + str(traceback_text) + "</pre><p style='color:#999;font-size:12px;'>Auto crash report.</p></body></html>"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header("Quant Error - " + now.strftime("%Y.%m.%d %H:%M"), "utf-8")
        msg["From"] = formataddr(("A-Quant", SMTP_CONFIG["user"]))
        msg["To"] = SMTP_CONFIG["user"]
        msg.attach(MIMEText(body, "html", "utf-8"))

        with smtplib.SMTP_SSL(SMTP_CONFIG["host"], SMTP_CONFIG["port"]) as server:
            server.login(SMTP_CONFIG["user"], password)
            server.sendmail(SMTP_CONFIG["user"], SMTP_CONFIG["user"], msg.as_string())

        logger.info("[MAIL] Crash report sent")
        return True
    except Exception as e:
        logger.error(f"[MAIL] Crash report failed: {e}")
        return False


def generate_wechat_markdown(report_data):
    if not report_data:
        return "报告生成失败"
    from daily_brief import render_daily_text
    return render_daily_text(report_data)


def send_wechat(report_data, token):
    """通过PushPlus推送到微信（免费200条/天，支持Markdown）"""
    if not token:
        logger.warning("[WECHAT] 未提供PushPlus Token，跳过微信推送")
        return False

    try:
        title = f"A股量化日报 {datetime.now().strftime('%m.%d')}"
        markdown = generate_wechat_markdown(report_data)

        # PushPlus API
        url = "https://www.pushplus.plus/send"
        data = json.dumps({
            "token": token,
            "title": title,
            "content": markdown,
            "template": "markdown"
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            if result.get("code") == 200:
                logger.info(f"[WECHAT] 微信推送成功 ✓")
                return True
            else:
                logger.error(f"[WECHAT] 推送失败: {result.get('msg', 'unknown')}")
                return False

    except Exception as e:
        logger.error(f"[WECHAT] 推送异常: {e}")
        return False


# ============================================================
# 每月复盘 + 自适应参数优化
# ============================================================
class MonthlyReview:  # 每月第一个周六运行，原命名WeeklyReview有歧义
    """每月复盘引擎 —— 追踪因子表现，自适应调整权重"""

    def __init__(self):
        self.factor_ic = defaultdict(list)       # 每个因子每日IC
        self.factor_ir = defaultdict(float)       # 每个因子IR
        self.default_weights = {                   # 默认权重
            "F1_趋势强度": 1.0, "F2_动量": 1.0, "F3_反转": 1.0,
            "F4_RSI": 1.0, "F5_均线偏离": 1.0, "F6_低波动": 1.0,
            "F7_成交量": 1.0, "F8_回调": 1.0, "F9_Sortino": 1.0,
            "F10_MaxDD": 1.0, "F11_布林带": 1.0, "F12_多周期": 1.0,
            "F13_均线排列": 1.0, "F14_长期": 1.0, "F15_夏普": 1.0
        }
        self.current_weights = dict(self.default_weights)
        self.review_history = []

    def load_history(self):
        """加载历史报告，计算因子IC（最多扫描500个文件防止IO爆炸）"""
        reports = []
        fnames = sorted(os.listdir(REPORT_DIR))
        # 只取最近500个报告文件（~2年数据），避免无限扫描
        for fname in fnames[-500:]:
            # 排除 report_paper_* 模拟盘报告(P0修复: 模拟盘污染实盘月度复盘)
            if fname.startswith("report_") and fname.endswith(".json") and fname[7:15].isdigit():
                try:
                    with open(os.path.join(REPORT_DIR, fname), 'r', encoding='utf-8') as f:
                        reports.append(json.load(f))
                except:
                    pass

        if len(reports) < 5:
            return  # 数据不够

        # 对每个因子，计算 IC (因子得分 vs 次日收益)
        for i in range(len(reports) - 1):
            today_scores = reports[i].get("scores", [])
            next_day = reports[i + 1]
            next_scores = {s["code"]: s for s in next_day.get("scores", [])}

            for s in today_scores:
                code = s["code"]
                factors = s.get("factors", {})
                if code not in next_scores:
                    continue
                next_ret = next_scores[code].get("returns", {}).get("r5d", 0)

                for fname, fval in factors.items():
                    # IC: factor value vs next return correlation (简化版)
                    self.factor_ic[fname].append(fval * next_ret)

    def calculate_ir(self):
        """计算每个因子的IR (IC均值/IC标准差)"""
        for fname, ic_list in self.factor_ic.items():
            if len(ic_list) < 5:
                self.factor_ir[fname] = 0
                continue
            avg = sum(ic_list) / len(ic_list)
            std = math.sqrt(sum((x-avg)**2 for x in ic_list) / (len(ic_list)-1)) if len(ic_list) > 1 else 1
            self.factor_ir[fname] = avg / std if std != 0 else 0

    def adapt_weights(self):
        """自适应调整权重"""
        self.calculate_ir()

        # 规则1: 连续20日IC为负 → 权重降为0.3
        # 规则2: IR > 0.5 → 权重提升到1.5
        # 规则3: IR < -0.3 → 权重降到0.5

        for fname in self.default_weights:
            ir = self.factor_ir.get(fname, 0)
            ic_list = self.factor_ic.get(fname, [])

            if len(ic_list) >= 20:
                recent_ic = ic_list[-20:]
                if sum(1 for x in recent_ic if x < 0) >= 15:
                    self.current_weights[fname] = 0.3  # 连续失效
                    continue

            if ir > 0.5:
                self.current_weights[fname] = 1.5
            elif ir > 0.2:
                self.current_weights[fname] = 1.2
            elif ir < -0.3:
                self.current_weights[fname] = 0.5
            else:
                self.current_weights[fname] = 1.0

    def get_weights(self):
        """获取当前自适应权重"""
        return self.current_weights

    def monthly_summary(self):  # 原函数名weekly_summary，实际每月运行
        """生成月度复盘报告"""
        self.load_history()
        self.adapt_weights()

        lines = []
        lines.append("=" * 60)
        lines.append(f"  月度复盘报告 - {datetime.now().strftime('%Y.%m.%d')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("  因子表现 (IC/IR):")
        lines.append("  " + "-" * 50)

        for fname in sorted(self.default_weights.keys()):
            ir = self.factor_ir.get(fname, 0)
            weight = self.current_weights.get(fname, 1.0)
            ic_list = self.factor_ic.get(fname, [])
            avg_ic = sum(ic_list)/len(ic_list) if ic_list else 0

            status = ""
            if weight > 1.0: status = " [UP]"
            elif weight < 1.0: status = " [DOWN]"
            elif avg_ic < 0: status = " [WATCH]"

            lines.append(f"  {fname:20s} | IR:{ir:+6.3f} | W:{weight:.1f} | IC均值:{avg_ic:+.4f}{status}")

        lines.append("")
        lines.append("  参数调整建议:")
        lines.append("  " + "-" * 50)

        changed = [(f, w) for f, w in self.current_weights.items() if w != 1.0]
        if changed:
            for f, w in changed:
                direction = "增" if w > 1 else "降"
                lines.append(f"  {f}: 权重{direction}为 {w:.1f}")
        else:
            lines.append("  所有因子权重保持默认 (1.0)")

        lines.append("")
        lines.append("  模型健康度检查:")
        lines.append("  " + "-" * 50)

        # 统计有效因子数
        active = sum(1 for w in self.current_weights.values() if w >= 0.5)
        dead = sum(1 for w in self.current_weights.values() if w <= 0.3)
        lines.append(f"  有效因子: {active}/15 | 失效因子: {dead}")
        lines.append(f"  历史报告数: {len(self.review_history) if hasattr(self, 'review_history') else 'N/A'}")

        return "\n".join(lines)

# ============================================================
# 独立运行
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--review":
        # 月度复盘模式
        reviewer = MonthlyReview()
        print(reviewer.monthly_summary())
    elif len(sys.argv) >= 2 and sys.argv[1] == "--send":
        # 发送邮件模式
        password = sys.argv[2] if len(sys.argv) >= 3 else None
        report = load_latest_report()
        if report:
            html = generate_html_report(report)
            send_email(html, password, report_date=report.get("date"))
        else:
            print("[MAIL] 无可用报告")
    else:
        print("Usage: python report_mailer.py --send <email_password>")
        print("       python report_mailer.py --review")
