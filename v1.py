“””
股票自動掃描程式 - 基於EMA/MACD/成交量實時監控與買賣信號
分析基礎：
下跌趨勢特徵（02/23）：EMA均線空頭排列、DIF<DEA均為負值、MACD柱由正轉負、成交量放量下殺
上漲趨勢特徵（02/17、02/19、02/20）：EMA均線多頭排列、DIF>DEA均為正值/向上交叉、價格突破壓力
“””

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ─────────────────────────────────────────────────────────

# 頁面設定

# ─────────────────────────────────────────────────────────

st.set_page_config(
page_title=“📈 股票自動掃描系統”,
page_icon=“📈”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

st.markdown(”””

<style>
body { background-color: #0d0d0d; }
.stApp { background-color: #111111; color: #f0f0f0; }
.buy-signal { background: linear-gradient(90deg,#003300,#006600); border-left: 4px solid #00ff00;
              padding: 12px; border-radius: 6px; margin: 6px 0; color: #00ff00; font-weight: bold; }
.sell-signal { background: linear-gradient(90deg,#330000,#660000); border-left: 4px solid #ff4444;
               padding: 12px; border-radius: 6px; margin: 6px 0; color: #ff4444; font-weight: bold; }
.neutral-signal { background: linear-gradient(90deg,#1a1a1a,#2a2a2a); border-left: 4px solid #888;
                  padding: 12px; border-radius: 6px; margin: 6px 0; color: #aaa; }
.metric-card { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 4px; border: 1px solid #333; }
.signal-title { font-size: 18px; font-weight: bold; margin-bottom: 6px; }
</style>

“””, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────

# 技術指標計算

# ─────────────────────────────────────────────────────────

def calc_ema(series, period):
return series.ewm(span=period, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
ema_fast = calc_ema(close, fast)
ema_slow = calc_ema(close, slow)
dif = ema_fast - ema_slow
dea = calc_ema(dif, signal)
macd_bar = (dif - dea) * 2
return dif, dea, macd_bar

def calc_indicators(df):
c = df[‘Close’]
df[‘EMA5’]   = calc_ema(c, 5)
df[‘EMA10’]  = calc_ema(c, 10)
df[‘EMA20’]  = calc_ema(c, 20)
df[‘EMA30’]  = calc_ema(c, 30)
df[‘EMA60’]  = calc_ema(c, 60)
df[‘EMA120’] = calc_ema(c, 120)
df[‘EMA200’] = calc_ema(c, 200)
df[‘MA5’]    = c.rolling(5).mean()
df[‘MA15’]   = c.rolling(15).mean()
df[‘DIF’], df[‘DEA’], df[‘MACD_BAR’] = calc_macd(c)
df[‘VOL_MA5’] = df[‘Volume’].rolling(5).mean()
df[‘VOL_MA20’] = df[‘Volume’].rolling(20).mean()

```
# 動量
df['ROC'] = c.pct_change(5) * 100
# ATR 用於止損
df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
return df
```

# ─────────────────────────────────────────────────────────

# 核心策略：根據圖表歸納的買賣邏輯

# ─────────────────────────────────────────────────────────

def generate_signal(df, shares=10):
“””
買入條件（上漲趨勢特徵）：
1. EMA5 > EMA10 > EMA20（短期多頭排列）
2. DIF > DEA 且 DIF 由負轉正或 MACD柱 > 0
3. 成交量 > 前5日均量（放量突破）
4. 收盤價 > MA5 且 MA5 > MA15（短期均線多頭）

```
賣出條件（下跌趨勢特徵）：
  1. EMA5 < EMA10 < EMA20（短期空頭排列）
  2. DIF < DEA 且兩者均 < 0（MACD死叉且在負區）
  3. 成交量 > 均量（放量下殺）
  4. 收盤價 < MA5 且 MA5 < MA15（短期均線空頭）
"""
if len(df) < 30:
    return "觀望", None, None, None, {}

last = df.iloc[-1]
prev = df.iloc[-2]
price = float(last['Close'])
atr   = float(last['ATR']) if not np.isnan(last['ATR']) else price * 0.01

details = {
    "EMA排列": None,
    "MACD狀態": None,
    "成交量": None,
    "MA短期": None,
    "得分": 0,
}

buy_score  = 0
sell_score = 0

# ── EMA排列 ──
if last['EMA5'] > last['EMA10'] > last['EMA20']:
    buy_score  += 2
    details["EMA排列"] = "✅ 多頭排列 (EMA5>EMA10>EMA20)"
elif last['EMA5'] < last['EMA10'] < last['EMA20']:
    sell_score += 2
    details["EMA排列"] = "🔴 空頭排列 (EMA5<EMA10<EMA20)"
else:
    details["EMA排列"] = "⚪ 均線糾纏中"

# ── MACD ──
dif_cross_up   = prev['DIF'] < prev['DEA'] and last['DIF'] > last['DEA']
dif_cross_down = prev['DIF'] > prev['DEA'] and last['DIF'] < last['DEA']
macd_positive  = last['MACD_BAR'] > 0
dif_positive   = last['DIF'] > 0

if dif_cross_up:
    buy_score += 3
    details["MACD狀態"] = "✅ DIF金叉DEA（強力買入信號）"
elif last['DIF'] > last['DEA'] and macd_positive:
    buy_score += 2
    details["MACD狀態"] = "✅ MACD多頭（DIF>DEA，柱體正值）"
elif dif_cross_down:
    sell_score += 3
    details["MACD狀態"] = "🔴 DIF死叉DEA（強力賣出信號）"
elif last['DIF'] < last['DEA'] and last['DIF'] < 0 and last['DEA'] < 0:
    sell_score += 2
    details["MACD狀態"] = "🔴 MACD空頭（DIF<DEA<0，圖表典型下跌特徵）"
else:
    details["MACD狀態"] = "⚪ MACD中性"

# ── 成交量 ──
vol_ratio = last['Volume'] / last['VOL_MA5'] if last['VOL_MA5'] > 0 else 1
if vol_ratio > 1.3 and last['Close'] > last['Open']:
    buy_score += 2
    details["成交量"] = f"✅ 放量上漲（量比={vol_ratio:.1f}x）"
elif vol_ratio > 1.3 and last['Close'] < last['Open']:
    sell_score += 2
    details["成交量"] = f"🔴 放量下跌（量比={vol_ratio:.1f}x）"
else:
    details["成交量"] = f"⚪ 量能平穩（量比={vol_ratio:.1f}x）"

# ── MA短期 ──
if last['Close'] > last['MA5'] and last['MA5'] > last['MA15']:
    buy_score += 1
    details["MA短期"] = "✅ 收盤>MA5>MA15（短期多頭）"
elif last['Close'] < last['MA5'] and last['MA5'] < last['MA15']:
    sell_score += 1
    details["MA短期"] = "🔴 收盤<MA5<MA15（短期空頭）"
else:
    details["MA短期"] = "⚪ MA短期中性"

# ── 決策 ──
buy_price = stop_loss = target = None

if buy_score >= 5 and buy_score > sell_score:
    signal = "買入"
    buy_price  = round(price, 2)
    stop_loss  = round(price - 2 * atr, 2)     # 止損=買入價-2×ATR
    target     = round(price + 3 * atr, 2)      # 目標=買入價+3×ATR（1:1.5風報比）
    details["得分"] = f"買入{buy_score} / 賣出{sell_score}"
elif sell_score >= 5 and sell_score > buy_score:
    signal = "賣出"
    buy_price  = round(price, 2)
    stop_loss  = round(price + 2 * atr, 2)      # 空單止損=賣出價+2×ATR
    target     = round(price - 3 * atr, 2)
    details["得分"] = f"買入{buy_score} / 賣出{sell_score}"
else:
    signal = "觀望"
    details["得分"] = f"買入{buy_score} / 賣出{sell_score}"

return signal, buy_price, stop_loss, target, details
```

# ─────────────────────────────────────────────────────────

# 資料擷取

# ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_data(ticker, period=“5d”, interval=“5m”):
try:
df = yf.download(ticker, period=period, interval=interval,
auto_adjust=True, progress=False)
if df.empty:
return None
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = calc_indicators(df)
return df
except Exception as e:
return None

# ─────────────────────────────────────────────────────────

# 繪圖（K線 + 成交量 + MACD）

# ─────────────────────────────────────────────────────────

def plot_chart(df, ticker, signal, buy_price, stop_loss, target):
df_plot = df.tail(100).copy()

```
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.03,
    subplot_titles=(f"{ticker} K線圖", "成交量", "MACD(12,26,9)")
)

# K線
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['Open'], high=df_plot['High'],
    low=df_plot['Low'],   close=df_plot['Close'],
    name="K線", increasing_line_color='#00ff88',
    decreasing_line_color='#ff4444'
), row=1, col=1)

colors_ema = {'EMA5':'#00ff00','EMA10':'#ffff00','EMA20':'#ff8800',
              'EMA30':'#ff4444','EMA60':'#aa44ff','MA5':'#00ccff'}
for col, color in colors_ema.items():
    if col in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col],
            name=col, line=dict(color=color, width=1), opacity=0.8), row=1, col=1)

# 買賣標記
if signal == "買入" and buy_price:
    fig.add_hline(y=buy_price, line_color='#00ff00', line_dash='dot',
                  annotation_text=f"買入 {buy_price}", row=1, col=1)
    fig.add_hline(y=stop_loss, line_color='#ff4444', line_dash='dash',
                  annotation_text=f"止損 {stop_loss}", row=1, col=1)
    fig.add_hline(y=target, line_color='#00ffff', line_dash='dash',
                  annotation_text=f"目標 {target}", row=1, col=1)
elif signal == "賣出" and buy_price:
    fig.add_hline(y=buy_price, line_color='#ff4444', line_dash='dot',
                  annotation_text=f"賣出 {buy_price}", row=1, col=1)
    fig.add_hline(y=stop_loss, line_color='#ff8800', line_dash='dash',
                  annotation_text=f"止損 {stop_loss}", row=1, col=1)
    fig.add_hline(y=target, line_color='#00ffff', line_dash='dash',
                  annotation_text=f"目標 {target}", row=1, col=1)

# 成交量
vol_colors = ['#00ff88' if c >= o else '#ff4444'
              for c, o in zip(df_plot['Close'], df_plot['Open'])]
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'],
    name="成交量", marker_color=vol_colors, opacity=0.7), row=2, col=1)

# MACD
macd_colors = ['#00ff88' if v >= 0 else '#ff4444'
               for v in df_plot['MACD_BAR']]
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_BAR'],
    name="MACD柱", marker_color=macd_colors, opacity=0.7), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DIF'],
    name="DIF", line=dict(color='#ffaa00', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DEA'],
    name="DEA", line=dict(color='#00aaff', width=1.5)), row=3, col=1)

fig.update_layout(
    height=750, template='plotly_dark',
    paper_bgcolor='#111111', plot_bgcolor='#1a1a1a',
    legend=dict(orientation='h', y=1.02),
    xaxis_rangeslider_visible=False,
    margin=dict(l=50, r=50, t=60, b=30)
)
fig.update_xaxes(showgrid=True, gridcolor='#333')
fig.update_yaxes(showgrid=True, gridcolor='#333')

return fig
```

# ─────────────────────────────────────────────────────────

# 側邊欄

# ─────────────────────────────────────────────────────────

with st.sidebar:
st.markdown(”## ⚙️ 掃描設定”)

```
st.markdown("### 股票清單")
default_tickers = "0050.TW\n2330.TW\n2317.TW\n2454.TW\n2382.TW\nAAPL\nTSLA\nNVDA"
ticker_input = st.text_area("每行一個代碼", default_tickers, height=180)
tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

st.markdown("### 參數")
interval = st.selectbox("K棒週期", ["5m","15m","1h","1d"], index=0)
period_map = {"5m":"5d","15m":"5d","1h":"1mo","1d":"6mo"}
data_period = period_map[interval]
shares = st.number_input("交易股數", min_value=1, max_value=10000, value=100)
auto_refresh = st.checkbox("🔄 自動刷新（60秒）", value=False)
min_score_buy  = st.slider("最低買入得分（滿8）", 3, 8, 5)

st.markdown("---")
st.markdown("### 📊 策略說明")
st.markdown("""
```

**買入信號條件：**

- EMA5 > EMA10 > EMA20 多頭排列
- DIF > DEA（金叉加分）
- 放量上漲（量比>1.3）
- 收盤 > MA5 > MA15

**賣出信號條件：**

- EMA5 < EMA10 < EMA20 空頭排列
- DIF < DEA < 0（雙負死叉）
- 放量下跌（量比>1.3）
- 收盤 < MA5 < MA15

**止損設定：**

- 2×ATR（14日真實波幅）

**目標設定：**

- 3×ATR（風報比 1:1.5）
  “””)

# ─────────────────────────────────────────────────────────

# 主頁面

# ─────────────────────────────────────────────────────────

st.markdown(”# 📈 股票自動掃描系統”)
st.markdown(f”**更新時間：** {datetime.now().strftime(’%Y-%m-%d %H:%M:%S’)} | **週期：** {interval}”)

if auto_refresh:
time.sleep(0.5)
st.rerun()

# 掃描按鈕

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
scan_btn = st.button(“🔍 立即掃描”, use_container_width=True)

# ─── 掃描結果 ───

st.markdown(”—”)
st.markdown(”## 📋 掃描結果總覽”)

if scan_btn or auto_refresh:
results = []
prog = st.progress(0, text=“掃描中…”)

```
for i, ticker in enumerate(tickers):
    prog.progress((i+1)/len(tickers), text=f"掃描 {ticker}...")
    df = fetch_data(ticker, data_period, interval)
    if df is None or len(df) < 30:
        results.append({
            "代碼": ticker, "現價": "N/A", "信號": "無數據",
            "買入價": "-", "止損": "-", "目標": "-",
            "潛在盈利%": "-", "潛在虧損%": "-", "數據": None
        })
        continue
    
    signal, buy_price, stop_loss, target, details = generate_signal(df, shares)
    price = float(df.iloc[-1]['Close'])
    
    if buy_price and stop_loss and target:
        if signal == "買入":
            profit_pct = round((target - buy_price) / buy_price * 100, 2)
            loss_pct   = round((buy_price - stop_loss) / buy_price * 100, 2)
        else:
            profit_pct = round((buy_price - target) / buy_price * 100, 2)
            loss_pct   = round((stop_loss - buy_price) / buy_price * 100, 2)
    else:
        profit_pct = loss_pct = "-"

    results.append({
        "代碼": ticker,
        "現價": f"{price:.2f}",
        "信號": signal,
        "買入價": f"{buy_price:.2f}" if buy_price else "-",
        "止損": f"{stop_loss:.2f}" if stop_loss else "-",
        "目標": f"{target:.2f}" if target else "-",
        "潛在盈利%": f"+{profit_pct}%" if profit_pct != "-" else "-",
        "潛在虧損%": f"-{loss_pct}%" if loss_pct != "-" else "-",
        "數據": df,
        "詳情": details
    })

prog.empty()

# 分類顯示
buy_results  = [r for r in results if r["信號"] == "買入"]
sell_results = [r for r in results if r["信號"] == "賣出"]
hold_results = [r for r in results if r["信號"] not in ("買入","賣出")]

col1, col2, col3 = st.columns(3)
col1.metric("🟢 買入信號", len(buy_results))
col2.metric("🔴 賣出信號", len(sell_results))
col3.metric("⚪ 觀望/無數據", len(hold_results))

# ── 買入信號區 ──
if buy_results:
    st.markdown("### 🟢 買入信號")
    for r in buy_results:
        st.markdown(f"""
```

<div class="buy-signal">
  <div class="signal-title">🟢 {r['代碼']} — 買入信號</div>
  💰 現價：<b>{r['現價']}</b> &nbsp;|&nbsp;
  📥 建議買入：<b>{r['買入價']}</b> × {shares}股
  &nbsp;=&nbsp; <b>{float(r['買入價'].replace(',','')) * shares if r['買入價']!='-' else '-':.0f}</b> 元<br>
  🛑 止損價：<b>{r['止損']}</b> &nbsp;|&nbsp;
  🎯 目標價：<b>{r['目標']}</b><br>
  📈 潛在盈利：<b>{r['潛在盈利%']}</b> &nbsp;|&nbsp;
  📉 最大虧損：<b>{r['潛在虧損%']}</b>
</div>""", unsafe_allow_html=True)

```
# ── 賣出信號區 ──
if sell_results:
    st.markdown("### 🔴 賣出信號")
    for r in sell_results:
        st.markdown(f"""
```

<div class="sell-signal">
  <div class="signal-title">🔴 {r['代碼']} — 賣出信號</div>
  💰 現價：<b>{r['現價']}</b> &nbsp;|&nbsp;
  📤 建議賣出：<b>{r['買入價']}</b> × {shares}股<br>
  🛑 止損價（空單）：<b>{r['止損']}</b> &nbsp;|&nbsp;
  🎯 目標價：<b>{r['目標']}</b><br>
  📈 潛在盈利：<b>{r['潛在盈利%']}</b> &nbsp;|&nbsp;
  📉 最大虧損：<b>{r['潛在虧損%']}</b>
</div>""", unsafe_allow_html=True)

```
# ── 觀望區 ──
if hold_results:
    st.markdown("### ⚪ 觀望/無數據")
    cols = st.columns(min(len(hold_results), 4))
    for i, r in enumerate(hold_results):
        cols[i % 4].markdown(f"""
```

<div class="neutral-signal">
  <b>{r['代碼']}</b><br>
  現價：{r['現價']}<br>
  狀態：{r['信號']}
</div>""", unsafe_allow_html=True)

```
# ── 詳細圖表 ──
st.markdown("---")
st.markdown("## 📊 個股詳細分析")

all_sig = buy_results + sell_results + hold_results
valid   = [r for r in all_sig if r["數據"] is not None]

if valid:
    selected = st.selectbox(
        "選擇個股查看詳細圖表",
        [r["代碼"] for r in valid]
    )
    sel_r = next(r for r in valid if r["代碼"] == selected)
    df_sel = sel_r["數據"]
    sig, bp, sl, tg, det = generate_signal(df_sel, shares)

    # 顯示圖表
    fig = plot_chart(df_sel, selected, sig, bp, sl, tg)
    st.plotly_chart(fig, use_container_width=True)

    # 信號詳情
    st.markdown("#### 🔍 信號分析詳情")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown(f"**EMA排列：** {det.get('EMA排列','N/A')}")
        st.markdown(f"**MACD狀態：** {det.get('MACD狀態','N/A')}")
    with dc2:
        st.markdown(f"**成交量：** {det.get('成交量','N/A')}")
        st.markdown(f"**MA短期：** {det.get('MA短期','N/A')}")
    st.markdown(f"**綜合得分：** {det.get('得分','N/A')}")

    # 具體操作指令
    st.markdown("#### 📌 操作指令")
    last_price = float(df_sel.iloc[-1]['Close'])
    last_ema5  = float(df_sel.iloc[-1]['EMA5'])
    last_dif   = float(df_sel.iloc[-1]['DIF'])
    last_dea   = float(df_sel.iloc[-1]['DEA'])
    last_macd  = float(df_sel.iloc[-1]['MACD_BAR'])

    if sig == "買入" and bp:
        cost = bp * shares
        max_loss = abs(bp - sl) * shares
        gain = abs(tg - bp) * shares
        st.success(f"""
```

🟢 **買入指令**

📥 **立即以 {bp:.2f} 買入 {shares} 股**（總成本：{cost:,.0f} 元）
🛑 **止損：{sl:.2f}**（若跌破立即賣出，最大虧損約 {max_loss:,.0f} 元）
🎯 **目標：{tg:.2f}**（達到時分批賣出，預計獲利 {gain:,.0f} 元）

📊 觸發原因：EMA多頭排列 + DIF({last_dif:.3f}) {’>’ if last_dif>last_dea else ‘<’} DEA({last_dea:.3f}) + MACD柱({last_macd:.3f})
“””)
elif sig == “賣出” and bp:
max_loss = abs(sl - bp) * shares
gain = abs(bp - tg) * shares
st.error(f”””
🔴 **賣出指令（或減倉/做空）**

📤 **立即以 {bp:.2f} 賣出 {shares} 股**
🛑 **止損：{sl:.2f}**（若反彈超過此價，回補止損，最大虧損約 {max_loss:,.0f} 元）
🎯 **目標：{tg:.2f}**（達到時回補，預計獲利 {gain:,.0f} 元）

📊 觸發原因：EMA空頭排列 + DIF({last_dif:.3f}) {’<’ if last_dif<last_dea else ‘>’} DEA({last_dea:.3f}) + 雙線負值（圖表下跌特徵）
“””)
else:
st.info(f”””
⚪ **觀望指令**

目前信號不明確，建議等待以下確認再入場：

- 等待 EMA5 明確穿越 EMA10
- 等待 DIF 與 DEA 形成金叉或死叉
- 確認成交量配合方向（放量突破）
  “””)
  
  ```
    # 近期數據表
    st.markdown("#### 📋 近10根K棒數據")
    show_cols = ['Open','High','Low','Close','Volume','EMA5','EMA10','DIF','DEA','MACD_BAR']
    st.dataframe(
        df_sel[show_cols].tail(10).round(3).style.background_gradient(
            subset=['Close'], cmap='RdYlGn'),
        use_container_width=True
    )
  ```

else:
# 首次載入提示
st.info(”””
👆 請點擊「立即掃描」開始分析股票

**系統功能：**

- 🔍 同時掃描多支股票
- 📊 5分鐘K棒即時分析（與您提供的圖表相同週期）
- 🎯 自動計算買入/賣出價、止損、目標
- 📈 EMA多頭/空頭排列判斷
- 📉 MACD金叉/死叉+負值區域偵測
- 📦 成交量放量確認

**策略來源：** 基於您提供的7張圖表分析歸納

- 下跌特徵(02/23)：空頭EMA排列 + DIF/DEA雙負 + 放量下殺
- 上漲特徵(02/17-02/20)：多頭EMA排列 + MACD金叉 + 放量突破
  “””)

st.markdown(”—”)
st.markdown(”””

<div style="text-align:center; color:#555; font-size:12px;">
⚠️ 本程式僅供參考，不構成投資建議。股票投資有風險，請自行評估並謹慎決策。
</div>
""", unsafe_allow_html=True)
