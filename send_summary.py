"""发送总结邮件"""
import json, smtplib, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DIR, 'email_config.json'), 'r') as f:
    config = json.load(f)

html = f'''
<html><head><meta charset="utf-8"><style>
body {{ font-family:"Microsoft YaHei",Arial; background:#1a1a2e; color:#eee; padding:20px; }}
.header {{ background:linear-gradient(135deg,#16213e,#0f3460); padding:25px; border-radius:10px; margin-bottom:20px; text-align:center; }}
.header h1 {{ margin:0; color:#e94560; }}
.card {{ background:#16213e; border-radius:10px; padding:20px; margin-bottom:15px; }}
.card h2 {{ color:#e94560; border-bottom:1px solid #0f3460; padding-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }}
th {{ background:#0f3460; padding:8px 10px; text-align:left; }}
td {{ padding:6px 10px; border-bottom:1px solid #1a1a3e; }}
.g {{ color:#00ff88; font-weight:bold; }} .r {{ color:#ff4757; }} .y {{ color:#ffa502; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; margin:0 3px; }}
.tb {{ background:#00ff8833; color:#00ff88; }} .tw {{ background:#ffa50233; color:#ffa502; }}
.ta {{ background:#ff475733; color:#ff4757; }} .th {{ background:#888833; color:#ffa502; }}
.footer {{ text-align:center; color:#666; margin-top:30px; font-size:11px; }}
.hl {{ background:#0f3460; border-left:3px solid #e94560; padding:15px; margin:10px 0; border-radius:5px; }}
</style></head><body>

<div class="header">
<h1>A股量化决策系统 - 2026.06.23 下午操作指南</h1>
<p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

<div class="card"><h2>一、大盘环境</h2>
<table>
<tr><th>择时信号</th><td class="r">1/6 看多 - 偏防御</td><th>建议仓位</th><td class="y">20% (上限30%)</td></tr>
<tr><td>沪深300在20日线上</td><td class="r">FAIL</td><td>沪深300的60日线向上</td><td class="r">FAIL</td></tr>
<tr><td>北向资金5日净流入</td><td class="r">FAIL</td><td>成交额>2万亿</td><td class="r">FAIL</td></tr>
<tr><td>跌停<20家</td><td class="g">PASS</td><td>融资余额增加</td><td class="r">FAIL</td></tr>
</table>
<p style="margin-top:10px"><b>盘中信号:</b> 科创50深V反转+1.48%，资金从有色周期流向科技成长。</p>
</div>

<div class="card"><h2>二、当前持仓</h2>
<table>
<tr><th>标的</th><th>成本</th><th>现价(约)</th><th>市值</th><th>浮盈</th><th>量化</th><th>建议</th></tr>
<tr>
<td><b>黄金ETF华夏</b><br><small>518850</small></td><td>9.091</td><td>8.700</td><td>1740元</td><td class="r">-4.3%</td><td>57/C</td>
<td><span class="tag th">持有不操作</span><br><small>RSI=28超卖，等周五PCE</small></td>
</tr>
<tr>
<td><b>新能源车ETF招商</b><br><small>159183</small></td><td>0.985</td><td>1.032</td><td>1032元</td><td class="g">+4.8%</td><td>-</td>
<td><span class="tag th">持有等止盈</span><br><small>止盈线1.064(+8%)，中报驱动</small></td>
</tr>
</table>
<div class="hl"><b>总资产: 4133元 | ETF市值: 2772元 | 可用资金: 1361元 | 现金占比: 32.9%</b></div>
</div>

<div class="card"><h2>三、关注ETF逐只分析</h2>
<table>
<tr><th>#</th><th>ETF</th><th>量化</th><th>RSI</th><th>5日</th><th>建议</th><th>核心逻辑</th></tr>
<tr>
<td>1</td><td><b>机器人ETF华夏 562500</b></td><td class="g">72/B</td><td>55</td><td>+2.1%</td>
<td><span class="tag tb">可买入</span></td>
<td><small>工信部标准+Walker C1+机构共识2026应用大年+RSI中性+未追高+Sortino 1.22</small></td>
</tr>
<tr>
<td>2</td><td><b>人工智能ETF华夏 515070</b></td><td class="g">79/A</td><td>58</td><td>+6.3%</td>
<td><span class="tag tw">观望</span></td>
<td><small>量化最高分但5日涨太多，等低开买点</small></td>
</tr>
<tr>
<td>3</td><td><b>芯片ETF国泰 512760</b></td><td class="y">57/C</td><td class="r">73</td><td class="r">+14.5%</td>
<td><span class="tag ta">回避</span></td>
<td><small>SEMI上调增速+长川预增110%但连涨6天RSI过热，等回调2.80-2.85</small></td>
</tr>
<tr>
<td>4</td><td><b>航空航天ETF华夏 159227</b></td><td>-</td><td>-</td><td>-</td>
<td><span class="tag ta">回避</span></td>
<td><small>SpaceX暴跌16%传导+机构看淡</small></td>
</tr>
<tr>
<td>5</td><td><b>纳斯达克100ETF 159659</b></td><td>-</td><td>-</td><td>-</td>
<td><span class="tag tw">等溢价</span></td>
<td><small>溢价5.8%>目标3-4%，等2.29-2.31</small></td>
</tr>
<tr>
<td>6</td><td><b>皇台酒业 000995</b></td><td class="r">43/D</td><td class="r">23</td><td class="r">-8.5%</td>
<td><span class="tag ta">回避</span></td>
<td><small>RSI极弱+20日-25.5%+无反转信号</small></td>
</tr>
</table>
</div>

<div class="card"><h2>四、今日操作建议</h2>
<div class="hl">
<h3 style="margin-top:0;color:#00ff88">唯一推荐: 买入机器人ETF华夏 562500</h3>
<table>
<tr><td>买入价</td><td><b>1.155</b></td><td>数量</td><td><b>400股</b></td></tr>
<tr><td>金额</td><td><b>462元</b></td><td>剩余现金</td><td><b>899元 (21.8%)</b></td></tr>
<tr><td>止损</td><td class="r"><b>1.097 (-5%, 亏23元)</b></td><td>止盈1</td><td class="g"><b>1.247 (+8%, 赚37元)</b></td></tr>
</table>
<p style="margin-top:10px"><b>理由:</b> 量化72分+RSI中性+新闻密集利好(工信部标准/优必选Walker C1/机构共识)+今天已跌过释放抛压</p>
</div>
<p><b>其余持仓: 全部不操作。</b> 黄金等周五PCE，新能源车等止盈1.064。</p>
</div>

<div class="card"><h2>五、今日关键新闻</h2>
<table>
<tr><th>方向</th><th>要点</th></tr>
<tr><td class="g">机器人</td><td>工信部征求人形机器人标准；优必选Walker C1发布；机构判断2026应用大年</td></tr>
<tr><td class="g">芯片</td><td>SEMI上调设备增速至23.5%；长川科技预增110%+；费城半导体历史新高</td></tr>
<tr><td class="g">AI</td><td>智谱市值破万亿；GLM-5.2追平国际水平；火山引擎Force大会今日开幕</td></tr>
<tr><td class="y">新能源</td><td>亿纬锂能储能景气兑现；6月电池排产+68%；中报业绩驱动板块回暖</td></tr>
<tr><td class="r">黄金</td><td>四大行上调贵金属保证金至140%；今晚美国PMI数据关键</td></tr>
<tr><td class="r">航天</td><td>SpaceX跌16%跌破发行价；短期情绪承压</td></tr>
</table>
</div>

<div class="card"><h2>机构资金流向</h2>
<table>
<tr><th>维度</th><th>TOP3</th></tr>
<tr><td>行业主力净流入</td><td class="g">每日自动拉取：半导体/AI/机器人/创新药/证券</td></tr>
<tr><td>龙虎榜机构净买</td><td>每日自动拉取机构席位大额买入标的</td></tr>
<tr><td>ETF份额增长</td><td>机构增持最多的ETF（申购放量）</td></tr>
<tr><td>北向资金偏好</td><td>外资重点流入的行业方向</td></tr>
</table>
<p style="font-size:11px; color:#888;">*收盘后数据更新，显示当日机构资金共识方向。不是跟单信号，是择时参考。</p>
</div>

<div class="footer">
<p>本报告由A股量化决策系统自动生成 | 仅供辅助决策 | 不构成投资建议</p>
<p>投资有风险 入市需谨慎 | 下次报告: 明日17:57</p>
</div>
</body></html>
'''

msg = MIMEMultipart('alternative')
msg['Subject'] = Header('A股量化日报 2026.06.23 - 下午操作指南', 'utf-8')
msg['From'] = formataddr(('A股量化系统', config['QQMAIL_USER']))
msg['To'] = config['QQMAIL_USER']
msg.attach(MIMEText(html, 'html', 'utf-8'))

with smtplib.SMTP_SSL('smtp.qq.com', 465) as server:
    server.login(config['QQMAIL_USER'], config['QQMAIL_AUTH_CODE'])
    server.sendmail(config['QQMAIL_USER'], config['QQMAIL_USER'], msg.as_string())

print('OK - Email sent to ' + config['QQMAIL_USER'])
