#!/usr/bin/env python
"""
邮件报告 + 每周复盘 + 自适应参数优化
=====================================
- 每日收盘后生成HTML报告
- 发送到用户QQ邮箱
- 每周六复盘并自适应调整因子权重
"""
import json, smtplib, os, math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
SMTP_CONFIG = {
    "host": "smtp.qq.com",
    "port": 465,
    "user": "1239617073@qq.com",
    "sender_name": "A股量化决策系统"
}

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_html_report(report_data):
    """生成HTML格式的邮件报告 — 增强版"""
    if not report_data:
        return "<p>报告生成失败</p>"

    timing = report_data.get("timing", {})
    plan = report_data.get("plan", {})
    scores = report_data.get("scores", [])

    # 关注列表
    watchlist = [s for s in scores if s.get("is_watchlist") and not s.get("is_holding")]
    holding_scores = [s for s in scores if s.get("is_holding")]
    stock_scores = [s for s in scores if s.get("is_stock")]

    pos_pct = int(timing.get('base_position', 0) * 100)
    signal_labels = {
        "S1_HS300_above_MA20":"沪深300在20日线上", "S2_HS300_MA60_up":"沪深300的60日线向上",
        "S3_NorthFlow_5d_positive":"北向资金5日净流入", "S4_Volume_active":"成交额>2万亿",
        "S5_LimitDown_low":"跌停<20家", "S6_Margin_increasing":"融资余额增加"
    }

    html = f"""
    <html><head><meta charset="utf-8"><style>
    body {{ font-family: 'Microsoft YaHei', Arial; background: #1a1a2e; color: #eee; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #16213e, #0f3460); padding: 25px; border-radius: 10px; margin-bottom: 20px; text-align: center; }}
    .header h1 {{ margin: 0; color: #e94560; }}
    .card {{ background: #16213e; border-radius: 10px; padding: 20px; margin-bottom: 15px; }}
    .card h2 {{ color: #e94560; border-bottom: 1px solid #0f3460; padding-bottom: 10px; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th {{ background: #0f3460; padding: 8px 10px; text-align: left; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #1a1a3e; }}
    .buy {{ color: #00ff88; font-weight: bold; }}
    .sell {{ color: #ff4757; font-weight: bold; }}
    .watch {{ color: #ffa502; }}
    .avoid {{ color: #666; text-decoration: line-through; }}
    .pass {{ color: #00ff88; }} .fail {{ color: #ff4757; }}
    .reason_good {{ color: #00cc66; font-size: 12px; }}
    .reason_bad {{ color: #ff6b7f; font-size: 12px; }}
    .kbx {{ display: inline-block; background: #0f3460; color: #ffa502; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 0 2px; }}
    .footer {{ text-align: center; color: #666; margin-top: 30px; font-size: 11px; }}
    .position-bar {{ background: #0f3460; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
    .position-fill {{ background: linear-gradient(90deg, #e94560, #ffa502, #00ff88); height: 100%; transition: width 0.5s; }}
    .limit {{ color: #ff4757; font-size: 12px; }}
    </style></head><body>

    <div class="header"><h1>A股量化决策系统 - 每日报告</h1><p>{datetime.now().strftime('%Y年%m月%d日')} | 收盘后自动生成</p></div>

    <!-- 一、大盘择时 -->
    <div class="card"><h2>一、大盘择时</h2>
    <p>看多信号: <b>{timing.get('bull_signals', 0)}/{timing.get('total_signals', 6)}</b> | 建议仓位: <b>{pos_pct}%</b></p>
    <div class="position-bar"><div class="position-fill" style="width:{pos_pct}%"></div></div>
    """

    for name, value in timing.get("signal_detail", {}).items():
        cls = "pass" if value else "fail"
        label = signal_labels.get(name, name)
        html += f'<span class="{cls}" style="margin-right:12px">{"✓" if value else "✗"} {label}</span>'

    html += f"<p>{timing.get('advice', '')}</p></div>"

    # 二、持仓实时
    hold_list = plan.get("hold_list", [])
    sell_list = plan.get("sell_list", [])
    if hold_list or sell_list:
        html += '<div class="card"><h2>二、当前持仓与操作建议</h2><table><tr><th>标的</th><th>建议</th><th>现价</th><th>盈亏</th><th>止损</th><th>止盈</th></tr>'
        for h in hold_list:
            html += f'<tr><td>{h["name"]}<br><small>{h["code"]}</small></td><td class="watch">持有观察</td><td>{h.get("current_price", "-")}</td><td style="color:{ "#00ff88" if h.get("pnl_pct",0)>0 else "#ff4757" }">{h.get("pnl_pct",0):+.1f}%</td><td>-</td><td>-</td></tr>'
        for s in sell_list:
            html += f'<tr><td>{s["name"]}<br><small>{s["code"]}</small></td><td class="sell">建议卖出</td><td>{s.get("current_price", "-")}</td><td style="color:#ff4757">{s.get("pnl_pct",0):+.1f}%</td><td>{"止损触发" if s.get("pnl_pct",0)<-8 else "评分下滑"}</td><td>-</td></tr>'
        html += '</table></div>'

    # 三、买入建议
    buy_list = plan.get("buy_list", [])
    if buy_list:
        html += '<div class="card"><h2>三、买入建议</h2><table><tr><th>标的</th><th>数量</th><th>价格</th><th>金额</th><th>止损</th><th>止盈</th><th>理由</th></tr>'
        for b in buy_list:
            sl = round(b["price"]*0.95, 3); tp = round(b["price"]*1.08, 3)
            reasons = "; ".join(b.get("reasons", [])[:2])
            html += f'<tr><td class="buy">{b["name"]}<br><small>{b["code"]}</small></td><td>{b["shares"]}股</td><td>{b["price"]:.3f}</td><td>{b["amount"]:.0f}元</td><td class="limit">{sl:.3f}</td><td style="color:#00ff88">{tp:.3f}</td><td style="font-size:11px">{reasons}</td></tr>'
        html += '</table></div>'

    # 四、关注ETF逐只分析
    if watchlist:
        html += '<div class="card"><h2>四、关注ETF逐只分析</h2>'
        for s in watchlist:
            ind = s.get("indicators", {})
            ret = s.get("returns", {})
            score = s["score"]
            rsi = ind.get("rsi", 50)
            is_overbought = rsi > 68
            is_extended = ind.get("consecutive_up", 0) >= 5
            action = "buy" if (score >= 65 and not is_overbought and not is_extended) else ("watch" if score >= 50 else "avoid")
            act_color = {"buy":"#00ff88","watch":"#ffa502","avoid":"#666"}[action]
            act_text = {"buy":"[可买入]","watch":"[观望]","avoid":"[回避]"}[action]
            html += f'<div style="border-left:3px solid {act_color}; padding:8px 12px; margin:8px 0; background:#111827;">'
            html += f'<b style="color:{act_color}">{act_text}</b> <b>{s["name"]}</b> <small>({s["code"]})</small> | {score}分 | RSI:{rsi:.0f} | 现价:{s["price"]:.4f}<br>'
            html += f'<small>5日:{ret.get("r5d",0):+.1f}% | 20日:{ret.get("r20d",0):+.1f}% | 60日:{ret.get("r60d",0):+.1f}% | 连涨:{ind.get("consecutive_up",0)}天</small>'
            if action == "buy":
                sl = round(s["price"]*0.95,4); tp = round(s["price"]*1.08,4)
                html += f'<br><small>止损:{sl:.4f}(-5%) | 止盈:{tp:.4f}(+8%)</small>'
            html += '</div>'
        html += '</div>'

    # 五、个股
    if stock_scores:
        html += '<div class="card"><h2>五、个股评分</h2>'
        for s in stock_scores:
            ind = s.get("indicators", {})
            ret = s.get("returns", {})
            html += f'<div style="border-left:3px solid #ffa502; padding:8px 12px; margin:4px 0; background:#111827;">'
            html += f'<b>{s["name"]}</b> <small>({s["code"]})</small> | {s["score"]}分 | RSI:{ind.get("rsi",0):.0f} | 现价:{s["price"]:.2f}<br>'
            html += f'<small>20日:{ret.get("r20d",0):+.1f}% | 60日:{ret.get("r60d",0):+.1f}% | 120日:{ret.get("r120d",0):+.1f}%</small></div>'
        html += '</div>'
    html += """
    <div class="footer">
        <p>本报告由A股量化决策系统自动生成 | 仅供辅助决策 | 不构成投资建议</p>
        <p>投资有风险，入市需谨慎</p>
    </div></body></html>
    """
    return html
    return html

def send_email(report_html, password):
    """发送邮件报告"""
    if not password:
        print("[MAIL] 未提供邮箱授权码，跳过发送", file=__import__('sys').stderr)
        return False

    try:
        from email.header import Header
        from email.utils import formataddr

        msg = MIMEMultipart('alternative')
        msg['Subject'] = Header(f"A股量化日报 - {datetime.now().strftime('%Y.%m.%d')}", 'utf-8')
        msg['From'] = formataddr(('A股量化系统', SMTP_CONFIG['user']))
        msg['To'] = SMTP_CONFIG['user']

        msg.attach(MIMEText(report_html, 'html', 'utf-8'))

        with smtplib.SMTP_SSL(SMTP_CONFIG['host'], SMTP_CONFIG['port']) as server:
            server.login(SMTP_CONFIG['user'], password)
            server.sendmail(SMTP_CONFIG['user'], SMTP_CONFIG['user'], msg.as_string())

        print(f"[MAIL] 报告已发送到 {SMTP_CONFIG['user']}", file=__import__('sys').stderr)
        return True
    except Exception as e:
        print(f"[MAIL] 发送失败: {e}", file=__import__('sys').stderr)
        return False

# ============================================================
# 每周复盘 + 自适应参数优化
# ============================================================
class WeeklyReview:
    """每周复盘引擎 —— 追踪因子表现，自适应调整权重"""

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
        """加载历史报告，计算因子IC"""
        reports = []
        for fname in sorted(os.listdir(REPORT_DIR)):
            if fname.startswith("report_") and fname.endswith(".json"):
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

    def weekly_summary(self):
        """生成周度复盘报告"""
        self.load_history()
        self.adapt_weights()

        lines = []
        lines.append("=" * 60)
        lines.append(f"  周度复盘报告 - {datetime.now().strftime('%Y.%m.%d')}")
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
        # 周度复盘模式
        reviewer = WeeklyReview()
        print(reviewer.weekly_summary())
    elif len(sys.argv) >= 2 and sys.argv[1] == "--send":
        # 发送邮件模式
        password = sys.argv[2] if len(sys.argv) >= 3 else None
        report = load_latest_report()
        if report:
            html = generate_html_report(report)
            send_email(html, password)
        else:
            print("[MAIL] 无可用报告")
    else:
        print("Usage: python report_mailer.py --send <email_password>")
        print("       python report_mailer.py --review")
