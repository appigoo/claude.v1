"""
股票自動掃描程式 v2.0
新增功能：回測引擎、勝率統計、實時統計儀表板、資金曲線、交易記錄
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ═══════════════════════════════════════════════════════════

# 頁面設定

# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="📈 股票智能掃描系統 v2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""

<style>
.stApp { background-color: #121212; color: #f0f0f0; }
.block-container { padding-top: 1rem; }
.buy-signal {
    background: linear-gradient(135deg,#001a00,#003300);
    border-left: 5px solid #00ff44;
    padding: 14px 16px; border-radius: 8px; margin: 8px 0;
    color: #00ff88; font-weight: bold; font-size:14px;
}
.sell-signal {
    background: linear-gradient(135deg,#1a0000,#330000);
    border-left: 5px solid #ff3333;
    padding: 14px 16px; border-radius: 8px; margin: 8px 0;
    color: #ff6666; font-weight: bold; font-size:14px;
}
.neutral-signal {
    background: #1c1c1c; border-left: 4px solid #555;
    padding: 10px 14px; border-radius: 6px; margin: 6px 0; color: #aaa;
}
.stat-card {
    background: #1a1a2e; border: 1px solid #333;
    border-radius: 10px; padding: 16px; text-align: center; margin: 4px;
}
.stat-value { font-size: 28px; font-weight: bold; margin: 4px 0; }
.stat-label { font-size: 12px; color: #888; }
.section-title {
    font-size: 20px; font-weight: bold;
    border-bottom: 2px solid #333; padding-bottom: 6px; margin: 20px 0 12px 0;
}
</style>

""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════

# 技術指標計算

# ═══════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    dif = calc_ema(close, fast) - calc_ema(close, slow)
    dea = calc_ema(dif, signal)
    return dif, dea, (dif - dea) * 2

def calc_indicators(df):
    c = df['Close']
    for p in [5, 10, 20, 30, 60, 120, 200]:
        df[f'EMA{p}'] = calc_ema(c, p)
    df['MA5']  = c.rolling(5).mean()
    df['MA15'] = c.rolling(15).mean()
    df['DIF'], df['DEA'], df['MACD_BAR'] = calc_macd(c)
    df['VOL_MA5']  = df['Volume'].rolling(5).mean()
    df['VOL_MA20'] = df['Volume'].rolling(20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['ROC'] = c.pct_change(5) * 100
    return df

# ═══════════════════════════════════════════════════════════

# 信號評分（單根K棒）

# ═══════════════════════════════════════════════════════════

def score_bar(row, prev_row):
    buy_score = sell_score = 0
    if row['EMA5'] > row['EMA10'] > row['EMA20']:
        buy_score += 2
    elif row['EMA5'] < row['EMA10'] < row['EMA20']:
        sell_score += 2

    dif_cross_up   = prev_row['DIF'] < prev_row['DEA'] and row['DIF'] > row['DEA']
    dif_cross_down = prev_row['DIF'] > prev_row['DEA'] and row['DIF'] < row['DEA']

    if dif_cross_up:
        buy_score += 3
    elif row['DIF'] > row['DEA'] and row['MACD_BAR'] > 0:
        buy_score += 2
    if dif_cross_down:
        sell_score += 3
    elif row['DIF'] < row['DEA'] and row['DIF'] < 0 and row['DEA'] < 0:
        sell_score += 2

    vol_ratio = row['Volume'] / row['VOL_MA5'] if row['VOL_MA5'] > 0 else 1
    if vol_ratio > 1.3 and row['Close'] > row['Open']:
        buy_score += 2
    elif vol_ratio > 1.3 and row['Close'] < row['Open']:
        sell_score += 2

    if row['Close'] > row['MA5'] and row['MA5'] > row['MA15']:
        buy_score += 1
    elif row['Close'] < row['MA5'] and row['MA5'] < row['MA15']:
        sell_score += 1

    return buy_score, sell_score

def generate_signal(df, shares=10):
    if len(df) < 30:
        return "觀望", None, None, None, {}
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = float(last['Close'])
    atr   = float(last['ATR']) if not np.isnan(last['ATR']) else price * 0.01

    buy_score, sell_score = score_bar(last, prev)
    details = {
        "EMA排列": "✅ 多頭排列" if buy_score >= 2 else ("🔴 空頭排列" if sell_score >= 2 else "⚪ 糾纏"),
        "MACD狀態": f"DIF={last['DIF']:.3f}  DEA={last['DEA']:.3f}",
        "成交量":   f"量比={last['Volume']/last['VOL_MA5']:.1f}x" if last['VOL_MA5'] > 0 else "N/A",
        "MA短期":   f"MA5={last['MA5']:.2f}  MA15={last['MA15']:.2f}",
        "得分":     f"買入{buy_score} / 賣出{sell_score}",
    }
    buy_price = stop_loss = target = None
    if buy_score >= 5 and buy_score > sell_score:
        signal    = "買入"
        buy_price = round(price, 2)
        stop_loss = round(price - 2 * atr, 2)
        target    = round(price + 3 * atr, 2)
    elif sell_score >= 5 and sell_score > buy_score:
        signal    = "賣出"
        buy_price = round(price, 2)
        stop_loss = round(price + 2 * atr, 2)
        target    = round(price - 3 * atr, 2)
    else:
        signal = "觀望"
    return signal, buy_price, stop_loss, target, details

# ═══════════════════════════════════════════════════════════

# ★ 回測引擎

# ═══════════════════════════════════════════════════════════

def run_backtest(df, initial_capital=100000, shares_per_trade=100,
                 atr_stop=2.0, atr_target=3.0, min_score=5):
    df = df.copy().reset_index()
    results   = []
    capital   = initial_capital
    eq_curve  = [capital]
    eq_times  = [0]

    in_trade  = False
    trade_dir = None
    entry_px  = stop_px = target_px = 0.0
    entry_time = entry_idx = None

    for i in range(30, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]
        ts   = row.get('Datetime', row.get('Date', i))

        # ── 持倉中檢查出場 ──
        if in_trade:
            hi = float(row['High'])
            lo = float(row['Low'])
            exited = False; exit_px = 0.0; exit_type = ""

            if trade_dir == 'long':
                if lo <= stop_px:
                    exit_px, exit_type = stop_px, "止損"
                    exited = True
                elif hi >= target_px:
                    exit_px, exit_type = target_px, "獲利"
                    exited = True
            else:
                if hi >= stop_px:
                    exit_px, exit_type = stop_px, "止損"
                    exited = True
                elif lo <= target_px:
                    exit_px, exit_type = target_px, "獲利"
                    exited = True

            if exited:
                pnl = (exit_px - entry_px) * shares_per_trade if trade_dir == 'long' \
                      else (entry_px - exit_px) * shares_per_trade
                capital += pnl
                results.append({
                    "方向":     "做多" if trade_dir == 'long' else "做空",
                    "進場時間": entry_time,
                    "出場時間": ts,
                    "持倉K棒":  i - entry_idx,
                    "進場價":   round(entry_px, 2),
                    "出場價":   round(exit_px,  2),
                    "止損價":   round(stop_px,  2),
                    "目標價":   round(target_px, 2),
                    "出場原因": exit_type,
                    "盈虧(元)": round(pnl, 2),
                    "盈虧%":    round(pnl / (entry_px * shares_per_trade) * 100, 2),
                    "資金餘額": round(capital, 2),
                })
                in_trade = False

        # ── 無持倉：尋找進場 ──
        if not in_trade:
            try:
                atr_val = float(row['ATR'])
                if np.isnan(atr_val) or atr_val == 0:
                    continue
            except:
                continue

            buy_s, sell_s = score_bar(row, prev)
            cl = float(row['Close'])

            if buy_s >= min_score and buy_s > sell_s:
                in_trade   = True; trade_dir = 'long'
                entry_px   = cl
                stop_px    = cl - atr_stop   * atr_val
                target_px  = cl + atr_target * atr_val
                entry_time = ts; entry_idx = i

            elif sell_s >= min_score and sell_s > buy_s:
                in_trade   = True; trade_dir = 'short'
                entry_px   = cl
                stop_px    = cl + atr_stop   * atr_val
                target_px  = cl - atr_target * atr_val
                entry_time = ts; entry_idx = i

        eq_curve.append(capital)
        eq_times.append(i)

    return pd.DataFrame(results), eq_curve, eq_times, capital

def calc_stats(trades_df, initial_capital, final_capital):
    if trades_df.empty:
        return {}
    wins  = trades_df[trades_df['盈虧(元)'] > 0]
    loses = trades_df[trades_df['盈虧(元)'] <= 0]
    total = len(trades_df)

    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_win  = wins['盈虧(元)'].mean()  if len(wins)  > 0 else 0
    avg_loss = loses['盈虧(元)'].mean() if len(loses) > 0 else 0
    pf = abs(wins['盈虧(元)'].sum() / loses['盈虧(元)'].sum()) \
         if loses['盈虧(元)'].sum() != 0 else float('inf')

    equity = trades_df['資金餘額'].values
    peak   = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak * 100).min()

    expectancy = win_rate/100 * avg_win + (1 - win_rate/100) * avg_loss

    streak = trades_df['盈虧(元)'].apply(lambda x: 1 if x > 0 else -1).values
    max_ws = max_ls = wc = lc = 0
    for s in streak:
        if s == 1: wc += 1; lc = 0
        else:      lc += 1; wc = 0
        max_ws = max(max_ws, wc); max_ls = max(max_ls, lc)

    return {
        "總交易次數":   total,
        "獲利次數":     len(wins),
        "虧損次數":     len(loses),
        "勝率%":        round(win_rate, 1),
        "平均獲利":     round(avg_win,  2),
        "平均虧損":     round(avg_loss, 2),
        "獲利因子":     round(pf,       2),
        "總盈虧":       round(trades_df['盈虧(元)'].sum(), 2),
        "總報酬%":      round((final_capital-initial_capital)/initial_capital*100, 2),
        "最大回撤%":    round(max_dd, 2),
        "期望值":       round(expectancy, 2),
        "最長連贏":     max_ws,
        "最長連虧":     max_ls,
        "最大單筆獲利": round(trades_df['盈虧(元)'].max(), 2),
        "最大單筆虧損": round(trades_df['盈虧(元)'].min(), 2),
    }

# ═══════════════════════════════════════════════════════════

# 資料擷取

# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_data(ticker, period="5d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty: return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return calc_indicators(df)
    except: return None

@st.cache_data(ttl=300)
def fetch_backtest_data(ticker, period="60d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty: return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return calc_indicators(df)
    except: return None

# ═══════════════════════════════════════════════════════════

# 繪圖

# ═══════════════════════════════════════════════════════════

def plot_main_chart(df, ticker, signal, buy_price, stop_loss, target, trades_df=None):
    df_plot = df.tail(120).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02,
                        subplot_titles=(f"{ticker} 5分K", "成交量", "MACD(12,26,9)"))

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name="K線",
        increasing_line_color='#00e676', decreasing_line_color='#ff1744'
    ), row=1, col=1)

    for col, color in [('EMA5','#00ff00'),('EMA10','#ffff00'),('EMA20','#ff8800'),
                       ('EMA30','#ff4466'),('EMA60','#cc44ff'),('MA5','#00cfff')]:
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=col,
                                     line=dict(color=color, width=1.2), opacity=0.85), row=1, col=1)

    for val, color, dash, label in [
        (buy_price, '#00ff44', 'dot',  f"{'買入' if signal=='買入' else '賣出'} {buy_price}"),
        (stop_loss, '#ff4444', 'dash', f"止損 {stop_loss}"),
        (target,    '#00e5ff', 'dash', f"目標 {target}"),
    ]:
        if val:
            fig.add_hline(y=val, line_color=color, line_dash=dash,
                          annotation_text=label, annotation_font_color=color, row=1, col=1)

    if trades_df is not None and not trades_df.empty:
        try:
            longs  = trades_df[trades_df['方向']=='做多']
            shorts = trades_df[trades_df['方向']=='做空']
            fig.add_trace(go.Scatter(x=longs['進場時間'], y=longs['進場價'],
                                     mode='markers', name='做多進場',
                                     marker=dict(symbol='triangle-up',   size=10, color='#00ff88')), row=1, col=1)
            fig.add_trace(go.Scatter(x=shorts['進場時間'], y=shorts['進場價'],
                                     mode='markers', name='做空進場',
                                     marker=dict(symbol='triangle-down', size=10, color='#ff4466')), row=1, col=1)
        except: pass

    vol_colors = ['#00e676' if c >= o else '#ff1744'
                  for c, o in zip(df_plot['Close'], df_plot['Open'])]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'],
                         name="成交量", marker_color=vol_colors, opacity=0.75), row=2, col=1)

    macd_colors = ['#00e676' if v >= 0 else '#ff1744' for v in df_plot['MACD_BAR']]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_BAR'],
                         name="MACD", marker_color=macd_colors, opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DIF'], name="DIF",
                             line=dict(color='#ffaa00', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DEA'], name="DEA",
                             line=dict(color='#00aaff', width=1.5)), row=3, col=1)
    fig.add_hline(y=0, line_color='#444', line_dash='dot', row=3, col=1)

    fig.update_layout(height=780, template='plotly_dark',
                      paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
                      legend=dict(orientation='h', y=1.02, font=dict(size=11)),
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=55, r=55, t=50, b=20))
    fig.update_xaxes(showgrid=True, gridcolor='#222', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#222', zeroline=False)
    return fig

def plot_equity_curve(eq_curve, init_cap, stats):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(eq_curve))), y=eq_curve,
        fill='tozeroy', fillcolor='rgba(0,200,100,0.08)',
        line=dict(color='#00e676', width=2), name="資金曲線"
    ))
    fig.add_hline(y=init_cap, line_color='#555', line_dash='dash',
                  annotation_text=f"初始資金 {init_cap:,.0f}")
    fig.update_layout(
        title=f"📈 資金曲線  |  最終: {eq_curve[-1]:,.0f}  |  報酬: {stats.get('總報酬%',0):+.1f}%",
        height=300, template='plotly_dark',
        paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
        margin=dict(l=50, r=30, t=50, b=30),
        xaxis_title="K棒序號", yaxis_title="資金(元)"
    )
    return fig

def plot_pnl_distribution(trades_df):
    wins  = trades_df[trades_df['盈虧(元)'] > 0]['盈虧(元)']
    loses = trades_df[trades_df['盈虧(元)'] <= 0]['盈虧(元)']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins,  name="獲利", marker_color='#00e676', opacity=0.75, nbinsx=20))
    fig.add_trace(go.Histogram(x=loses, name="虧損", marker_color='#ff1744', opacity=0.75, nbinsx=20))
    fig.add_vline(x=0, line_color='#fff', line_dash='dash')
    fig.update_layout(title="盈虧分佈直方圖", barmode='overlay', height=280,
                      template='plotly_dark', paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
                      margin=dict(l=40, r=20, t=40, b=30))
    return fig

def plot_monthly_pnl(trades_df):
    df = trades_df.copy()
    try:
        df['月份'] = pd.to_datetime(df['出場時間']).dt.to_period('M').astype(str)
    except: return None
    monthly = df.groupby('月份')['盈虧(元)'].sum().reset_index()
    colors  = ['#00e676' if v >= 0 else '#ff1744' for v in monthly['盈虧(元)']]
    fig = go.Figure(go.Bar(x=monthly['月份'], y=monthly['盈虧(元)'],
                           marker_color=colors,
                           text=monthly['盈虧(元)'].apply(lambda x: f"{x:+,.0f}"),
                           textposition='outside'))
    fig.update_layout(title="月度盈虧統計", height=280, template='plotly_dark',
                      paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
                      margin=dict(l=40, r=20, t=40, b=30))
    return fig

# ═══════════════════════════════════════════════════════════

# 側邊欄

# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ 系統設定")
    page = st.radio("功能模組", ["📡 實時掃描", "🔬 回測分析", "📊 多股比較"])
    st.markdown("—")

    st.markdown("### 股票清單")
    default_tickers = "TSLA\nNIO.TW\nTSLL\nXPEV\nAMZN\nNVDA\nMETA\nAAPL\nGOOGL\nAAPL\nNVDA\nMSFT\nTSM\nGLD\nBTC-USD\nQQQ"
    ticker_input = st.text_area("每行一個代碼", default_tickers, height=170)
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

    st.markdown("### 交易參數")
    interval    = st.selectbox("K棒週期", ["5m","15m","1h","1d"], index=0)
    shares      = st.number_input("交易股數", 1, 100000, 100)
    atr_stop    = st.slider("止損ATR倍數",  1.0, 4.0, 2.0, 0.5)
    atr_target  = st.slider("目標ATR倍數",  1.0, 6.0, 3.0, 0.5)
    min_score   = st.slider("最低信號得分", 3, 8, 5)

    st.markdown("### 回測設定")
    bt_period    = st.selectbox("回測週期", ["30d","60d","3mo","6mo","1y"], index=1)
    init_capital = st.number_input("初始資金(元)", 10000, 10000000, 100000, 10000)
    auto_refresh = st.checkbox("🔄 自動刷新(60秒)", False)

    period_map  = {"5m":"5d","15m":"10d","1h":"1mo","1d":"6mo"}
    data_period = period_map.get(interval, "5d")

    st.markdown("---")
    st.markdown("""<div style="font-size:12px;color:#666;">
📌 策略邏輯<br>
買入：EMA多頭排列 + MACD金叉 + 放量上漲<br>
賣出：EMA空頭排列 + MACD死叉+負值 + 放量下跌<br>
止損：ATR倍數可調｜目標：ATR倍數可調

</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════

# 標題欄

# ═══════════════════════════════════════════════════════════

st.markdown(f"""

<div style="background:linear-gradient(90deg,#0d0d2e,#1a1a3e);
     padding:16px 24px;border-radius:10px;margin-bottom:16px;border:1px solid #333;">
  <span style="font-size:26px;font-weight:bold;">📈 股票智能掃描系統 v2.0</span>
  <span style="float:right;color:#888;font-size:13px;">
    🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; 週期：{interval}
  </span>
</div>""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(1)
    st.rerun()

# ═══════════════════════════════════════════════════════════

# ① 實時掃描

# ═══════════════════════════════════════════════════════════

if page == "📡 實時掃描":
    scan_btn = st.button("🔍 立即掃描", type="primary")

    if scan_btn or auto_refresh:
        results = []
        prog = st.progress(0, text="掃描中...")
        for i, ticker in enumerate(tickers):
            prog.progress((i+1)/len(tickers), text=f"分析 {ticker}...")
            df = fetch_data(ticker, data_period, interval)
            if df is None or len(df) < 30:
                results.append({"代碼":ticker,"現價":"N/A","信號":"無數據",
                                "買入價":"-","止損":"-","目標":"-","盈利%":"-","虧損%":"-",
                                "數據":None,"詳情":{}})
                continue
            sig, bp, sl, tg, det = generate_signal(df, shares)
            price = float(df.iloc[-1]['Close'])
            if bp and sl and tg:
                p_pct = round((tg-bp)/bp*100,2) if sig=="買入" else round((bp-tg)/bp*100,2)
                l_pct = round((bp-sl)/bp*100,2) if sig=="買入" else round((sl-bp)/bp*100,2)
            else:
                p_pct = l_pct = "-"
            results.append({"代碼":ticker,"現價":f"{price:.2f}","信號":sig,
                            "買入價":f"{bp:.2f}" if bp else "-",
                            "止損":f"{sl:.2f}" if sl else "-",
                            "目標":f"{tg:.2f}" if tg else "-",
                            "盈利%":f"+{p_pct}%" if p_pct!="-" else "-",
                            "虧損%":f"-{l_pct}%" if l_pct!="-" else "-",
                            "數據":df,"詳情":det})
        prog.empty()

        buys  = [r for r in results if r["信號"]=="買入"]
        sells = [r for r in results if r["信號"]=="賣出"]
        holds = [r for r in results if r["信號"] not in ("買入","賣出")]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🟢 買入信號", len(buys))
        c2.metric("🔴 賣出信號", len(sells))
        c3.metric("⚪ 觀望", len(holds))
        c4.metric("📊 掃描總數", len(tickers))

        if buys:
            st.markdown('<div class="section-title">🟢 買入信號</div>', unsafe_allow_html=True)
            for r in buys:
                try: cost = f"{float(r['買入價'])*shares:,.0f}"
                except: cost = "-"
                st.markdown(f"""
<div class="buy-signal">
  🟢 <b>{r['代碼']}</b> — 買入信號<br>
  💰 現價：<b>{r['現價']}</b> &nbsp;|&nbsp;
  📥 建議買入：<b>{r['買入價']}</b> × {shares}股 = <b>{cost}</b> 元<br>
  🛑 止損：<b>{r['止損']}</b> &nbsp;|&nbsp; 🎯 目標：<b>{r['目標']}</b><br>
  📈 潛在盈利：<b>{r['盈利%']}</b> &nbsp;|&nbsp; 📉 最大虧損：<b>{r['虧損%']}</b>
</div>""", unsafe_allow_html=True)

        if sells:
            st.markdown('<div class="section-title">🔴 賣出信號</div>', unsafe_allow_html=True)
            for r in sells:
                st.markdown(f"""
<div class="sell-signal">
  🔴 <b>{r['代碼']}</b> — 賣出信號<br>
  💰 現價：<b>{r['現價']}</b> &nbsp;|&nbsp;
  📤 建議賣出：<b>{r['買入價']}</b> × {shares}股<br>
  🛑 空單止損：<b>{r['止損']}</b> &nbsp;|&nbsp; 🎯 目標：<b>{r['目標']}</b><br>
  📈 潛在盈利：<b>{r['盈利%']}</b> &nbsp;|&nbsp; 📉 最大虧損：<b>{r['虧損%']}</b>
</div>""", unsafe_allow_html=True)

        if holds:
            st.markdown('<div class="section-title">⚪ 觀望中</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(holds), 4))
            for i, r in enumerate(holds):
                cols[i%4].markdown(f"""
<div class="neutral-signal">
  <b>{r['代碼']}</b> | {r['現價']}<br>
  <span style="color:#888;font-size:12px;">{r['信號']}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📊 個股詳細分析</div>', unsafe_allow_html=True)
        valid = [r for r in results if r["數據"] is not None]
        if valid:
            sel  = st.selectbox("選擇個股", [r["代碼"] for r in valid])
            sr   = next(r for r in valid if r["代碼"]==sel)
            df_s = sr["數據"]
            sig2, bp2, sl2, tg2, det2 = generate_signal(df_s, shares)

            st.plotly_chart(plot_main_chart(df_s, sel, sig2, bp2, sl2, tg2), use_container_width=True)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**EMA排列：** {det2.get('EMA排列')}")
                st.markdown(f"**MACD：** {det2.get('MACD狀態')}")
            with d2:
                st.markdown(f"**成交量：** {det2.get('成交量')}")
                st.markdown(f"**MA短期：** {det2.get('MA短期')}")
            st.markdown(f"**綜合得分：** `{det2.get('得分')}`")

            last = df_s.iloc[-1]
            if sig2 == "買入" and bp2:
                gain = abs(tg2-bp2)*shares; loss = abs(bp2-sl2)*shares
                st.success(f"""
🟢 **操作指令 → 立即以 {bp2:.2f} 買入 {shares} 股**

- 📥 總成本：**{bp2*shares:,.0f}** 元
- 🛑 止損：**{sl2:.2f}**（最大虧損 {loss:,.0f} 元）
- 🎯 目標：**{tg2:.2f}**（預期獲利 {gain:,.0f} 元）
- 📊 DIF={float(last['DIF']):.3f}  DEA={float(last['DEA']):.3f}  MACD柱={float(last['MACD_BAR']):.3f}
  """)
            elif sig2 == "賣出" and bp2:
                gain = abs(bp2-tg2)*shares; loss = abs(sl2-bp2)*shares
                st.error(f"""
🔴 **操作指令 → 立即以 {bp2:.2f} 賣出/做空 {shares} 股**
- 🛑 止損：**{sl2:.2f}**（最大虧損 {loss:,.0f} 元）
- 🎯 目標：**{tg2:.2f}**（預期獲利 {gain:,.0f} 元）
- 📊 DIF={float(last['DIF']):.3f}  DEA={float(last['DEA']):.3f}（雙負空頭特徵）
  """)
            else:
                st.info("⚪ **觀望** — 等待EMA排列明確 + MACD金/死叉 + 量能配合")
          
            show_cols = ['Open','High','Low','Close','Volume','EMA5','EMA10','DIF','DEA','MACD_BAR']
            st.markdown("**近15根K棒數據**")
            st.dataframe(df_s[show_cols].tail(15).round(3), use_container_width=True)
      
        else:
            st.info('👆 點擊「立即掃描」開始實時分析所有股票')

# ═══════════════════════════════════════════════════════════

# ② 回測分析

# ═══════════════════════════════════════════════════════════

elif page == "🔬 回測分析":
    st.markdown('<div class="section-title">🔬 策略回測引擎</div>', unsafe_allow_html=True)
    bt_ticker = st.selectbox("選擇回測股票", tickers)
    bt_btn    = st.button("▶️ 執行回測", type="primary")

    if bt_btn:
        with st.spinner(f"正在回測 {bt_ticker} ({bt_period})..."):
            df_bt = fetch_backtest_data(bt_ticker, bt_period, interval)

        if df_bt is None or len(df_bt) < 50:
            st.error("數據不足，請換個股票或延長週期")
        else:
            trades_df, eq_curve, eq_times, final_cap = run_backtest(
                df_bt, init_capital, shares, atr_stop, atr_target, min_score)
            stats = calc_stats(trades_df, init_capital, final_cap)

            if not stats:
                st.warning("回測期間未產生有效交易，請調整參數或延長週期")
            else:
                # ── 6大核心指標 ──
                st.markdown("### 📊 核心績效指標")
                s1,s2,s3,s4,s5,s6 = st.columns(6)
                wc = "#00ff88" if stats['勝率%'] >= 50 else "#ff4444"
                rc = "#00ff88" if stats['總報酬%'] >= 0 else "#ff4444"
                pc = "#00ff88" if stats['獲利因子'] >= 1 else "#ff4444"

                s1.markdown(f"""<div class="stat-card">
                    <div class="stat-label">勝率</div>
                    <div class="stat-value" style="color:{wc}">{stats['勝率%']}%</div>
                    <div class="stat-label">{stats['獲利次數']}勝/{stats['虧損次數']}敗</div>
                </div>""", unsafe_allow_html=True)
                s2.markdown(f"""<div class="stat-card">
                    <div class="stat-label">總報酬</div>
                    <div class="stat-value" style="color:{rc}">{stats['總報酬%']:+.1f}%</div>
                    <div class="stat-label">{stats['總盈虧']:+,.0f} 元</div>
                </div>""", unsafe_allow_html=True)
                s3.markdown(f"""<div class="stat-card">
                    <div class="stat-label">獲利因子</div>
                    <div class="stat-value" style="color:{pc}">{stats['獲利因子']}</div>
                    <div class="stat-label">共 {stats['總交易次數']} 次交易</div>
                </div>""", unsafe_allow_html=True)
                s4.markdown(f"""<div class="stat-card">
                    <div class="stat-label">最大回撤</div>
                    <div class="stat-value" style="color:#ff8844">{stats['最大回撤%']:.1f}%</div>
                    <div class="stat-label">風險指標</div>
                </div>""", unsafe_allow_html=True)
                s5.markdown(f"""<div class="stat-card">
                    <div class="stat-label">期望值/筆</div>
                    <div class="stat-value" style="color:#ffcc00">{stats['期望值']:+,.0f}</div>
                    <div class="stat-label">元/交易</div>
                </div>""", unsafe_allow_html=True)
                s6.markdown(f"""<div class="stat-card">
                    <div class="stat-label">最終資金</div>
                    <div class="stat-value" style="color:#00cfff">{final_cap:,.0f}</div>
                    <div class="stat-label">初始 {init_capital:,.0f}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── 詳細統計 ──
                st.markdown("### 📋 詳細統計數據")
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.metric("平均獲利/筆",   f"{stats['平均獲利']:+,.2f} 元")
                    st.metric("平均虧損/筆",   f"{stats['平均虧損']:+,.2f} 元")
                    st.metric("最大單筆獲利",  f"{stats['最大單筆獲利']:+,.2f} 元")
                with tc2:
                    st.metric("最大單筆虧損",  f"{stats['最大單筆虧損']:+,.2f} 元")
                    st.metric("最長連贏",      f"{stats['最長連贏']} 筆")
                    st.metric("最長連虧",      f"{stats['最長連虧']} 筆")
                with tc3:
                    rr = abs(stats['平均獲利']/stats['平均虧損']) if stats['平均虧損'] != 0 else 0
                    st.metric("平均風報比",    f"1 : {rr:.2f}")
                    st.metric("止損ATR倍數",   f"{atr_stop}x")
                    st.metric("目標ATR倍數",   f"{atr_target}x")

                # ── 資金曲線 ──
                st.markdown("### 📈 資金曲線")
                st.plotly_chart(plot_equity_curve(eq_curve, init_capital, stats),
                                use_container_width=True)

                # ── 盈虧分佈 + 月度 ──
                if not trades_df.empty:
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        st.plotly_chart(plot_pnl_distribution(trades_df), use_container_width=True)
                    with ch2:
                        fig_mo = plot_monthly_pnl(trades_df)
                        if fig_mo:
                            st.plotly_chart(fig_mo, use_container_width=True)

                # ── 勝率圓餅圖 ──
                st.markdown("### 🥧 勝敗分佈")
                pc1, pc2 = st.columns(2)
                with pc1:
                    fig_pie = go.Figure(go.Pie(
                        labels=['獲利交易', '虧損交易'],
                        values=[stats['獲利次數'], stats['虧損次數']],
                        marker_colors=['#00e676','#ff1744'],
                        hole=0.4,
                        textinfo='label+percent'
                    ))
                    fig_pie.update_layout(title=f"勝率 {stats['勝率%']}%",
                                          height=280, template='plotly_dark',
                                          paper_bgcolor='#0d0d0d',
                                          margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)
                with pc2:
                    # 方向分佈
                    if not trades_df.empty:
                        dir_grp = trades_df.groupby('方向')['盈虧(元)'].agg(['count','sum']).reset_index()
                        fig_dir = go.Figure(go.Bar(
                            x=dir_grp['方向'],
                            y=dir_grp['sum'],
                            text=dir_grp.apply(lambda r: f"{int(r['count'])}次\n{r['sum']:+,.0f}元", axis=1),
                            textposition='outside',
                            marker_color=['#00e676' if v >= 0 else '#ff1744' for v in dir_grp['sum']]
                        ))
                        fig_dir.update_layout(title="多空方向盈虧",
                                              height=280, template='plotly_dark',
                                              paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
                                              margin=dict(l=30,r=20,t=40,b=30))
                        st.plotly_chart(fig_dir, use_container_width=True)

                # ── K線圖（含回測標記）──
                st.markdown("### 📉 近期K線（含回測進場點）")
                df_recent = fetch_data(bt_ticker, data_period, interval)
                if df_recent is not None:
                    sig_r, bp_r, sl_r, tg_r, _ = generate_signal(df_recent)
                    st.plotly_chart(plot_main_chart(df_recent, bt_ticker, sig_r, bp_r, sl_r, tg_r, trades_df),
                                    use_container_width=True)

                # ── 交易記錄表 ──
                if not trades_df.empty:
                    st.markdown("### 📜 完整交易記錄")
                    disp_cols = ['方向','進場時間','出場時間','持倉K棒',
                                 '進場價','出場價','止損價','目標價','出場原因','盈虧(元)','盈虧%','資金餘額']

                    def hl(row):
                        c = '#002200' if row['盈虧(元)'] > 0 else '#220000'
                        return [f'background-color:{c}']*len(row)

                    st.dataframe(
                        trades_df[disp_cols].style.apply(hl, axis=1).format({
                            '盈虧(元)':'{:+,.2f}','盈虧%':'{:+.2f}%',
                            '資金餘額':'{:,.0f}','進場價':'{:.2f}','出場價':'{:.2f}'}),
                        use_container_width=True, height=420
                    )
                    csv = trades_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("⬇️ 下載交易記錄 CSV", csv,
                                       f"{bt_ticker}_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                                       "text/csv")
    else:
        st.info('👆 選擇股票後點擊「執行回測」，系統將自動模擬所有歷史交易並統計勝率')

# ═══════════════════════════════════════════════════════════

# ③ 多股比較

# ═══════════════════════════════════════════════════════════

elif page == "📊 多股比較":
    st.markdown('<div class="section-title">📊 多股回測勝率比較</div>', unsafe_allow_html=True)
    compare_btn = st.button("🔄 開始全部回測比較", type="primary")

    if compare_btn:
        compare_results = []
        prog2 = st.progress(0, text="批量回測中...")

        for i, ticker in enumerate(tickers):
            prog2.progress((i+1)/len(tickers), text=f"回測 {ticker}...")
            df_c = fetch_backtest_data(ticker, bt_period, interval)
            if df_c is None or len(df_c) < 50:
                compare_results.append({"股票":ticker,"狀態":"數據不足"}); continue
            t_df, eq, _, fc = run_backtest(df_c, init_capital, shares, atr_stop, atr_target, min_score)
            s = calc_stats(t_df, init_capital, fc)
            if not s:
                compare_results.append({"股票":ticker,"狀態":"無信號"}); continue
            compare_results.append({
                "股票":      ticker, "狀態":"✅",
                "勝率%":     s['勝率%'],
                "總報酬%":   s['總報酬%'],
                "總交易":    s['總交易次數'],
                "獲利因子":  s['獲利因子'],
                "最大回撤%": s['最大回撤%'],
                "期望值(元)":s['期望值'],
            })
        prog2.empty()

        cr_df    = pd.DataFrame(compare_results)
        valid_cr = cr_df[cr_df['狀態']=="✅"].copy()

        if not valid_cr.empty:
            valid_cr = valid_cr.sort_values('勝率%', ascending=False)

            st.markdown("### 🏆 勝率排行榜")

            def color_row(row):
                wr = row.get('勝率%', 50)
                if wr >= 60: return ['background-color:#002200']*len(row)
                elif wr >= 50: return ['background-color:#111800']*len(row)
                else: return ['background-color:#1a0000']*len(row)

            fmt = {'勝率%':'{:.1f}%','總報酬%':'{:+.1f}%',
                   '最大回撤%':'{:.1f}%','獲利因子':'{:.2f}','期望值(元)':'{:+,.0f}'}
            st.dataframe(
                valid_cr.style.apply(color_row, axis=1).format(fmt,
                                                               subset=[c for c in fmt if c in valid_cr.columns]),
                use_container_width=True, height=350)

            # 視覺化
            ch_a, ch_b = st.columns(2)
            with ch_a:
                fig_wr = go.Figure(go.Bar(
                    x=valid_cr['股票'], y=valid_cr['勝率%'],
                    marker_color=['#00e676' if w>=50 else '#ff4444' for w in valid_cr['勝率%']],
                    text=valid_cr['勝率%'].apply(lambda x: f"{x:.1f}%"), textposition='outside'))
                fig_wr.add_hline(y=50, line_color='#fff', line_dash='dash',
                                 annotation_text="50%基準線")
                fig_wr.update_layout(title="各股勝率比較", height=300,
                                     template='plotly_dark', paper_bgcolor='#0d0d0d',
                                     plot_bgcolor='#151520', margin=dict(l=30,r=20,t=40,b=30))
                st.plotly_chart(fig_wr, use_container_width=True)

            with ch_b:
                fig_ret = go.Figure(go.Bar(
                    x=valid_cr['股票'], y=valid_cr['總報酬%'],
                    marker_color=['#00e676' if r>=0 else '#ff4444' for r in valid_cr['總報酬%']],
                    text=valid_cr['總報酬%'].apply(lambda x: f"{x:+.1f}%"), textposition='outside'))
                fig_ret.add_hline(y=0, line_color='#fff', line_dash='dash')
                fig_ret.update_layout(title="各股總報酬比較", height=300,
                                      template='plotly_dark', paper_bgcolor='#0d0d0d',
                                      plot_bgcolor='#151520', margin=dict(l=30,r=20,t=40,b=30))
                st.plotly_chart(fig_ret, use_container_width=True)

            # 散點圖：勝率 vs 報酬
            fig_sc = go.Figure(go.Scatter(
                x=valid_cr['勝率%'], y=valid_cr['總報酬%'],
                mode='markers+text',
                text=valid_cr['股票'], textposition='top center',
                marker=dict(
                    size=valid_cr['總交易'].apply(lambda x: max(8, min(30, x))),
                    color=valid_cr['獲利因子'],
                    colorscale='RdYlGn', showscale=True,
                    colorbar=dict(title="獲利因子")
                )
            ))
            fig_sc.add_vline(x=50, line_color='#555', line_dash='dash')
            fig_sc.add_hline(y=0,  line_color='#555', line_dash='dash')
            fig_sc.update_layout(
                title="勝率 vs 報酬 散點圖（泡泡大小=交易次數，顏色=獲利因子）",
                xaxis_title="勝率(%)", yaxis_title="總報酬(%)",
                height=380, template='plotly_dark',
                paper_bgcolor='#0d0d0d', plot_bgcolor='#151520',
                margin=dict(l=50,r=30,t=50,b=40))
            st.plotly_chart(fig_sc, use_container_width=True)

            # 綜合推薦
            st.markdown("### 🎯 綜合推薦（勝率≥55% 且 報酬>0）")
            top = valid_cr[(valid_cr['勝率%']>=55) & (valid_cr['總報酬%']>0)]
            if not top.empty:
                for _, row in top.iterrows():
                    st.markdown(f"""
<div class="buy-signal">
  🏆 <b>{row['股票']}</b> &nbsp;—&nbsp;
  勝率：<b>{row['勝率%']:.1f}%</b> &nbsp;|&nbsp;
  報酬：<b>{row['總報酬%']:+.1f}%</b> &nbsp;|&nbsp;
  獲利因子：<b>{row['獲利因子']:.2f}</b> &nbsp;|&nbsp;
  期望值：<b>{row['期望值(元)']:+,.0f}元/筆</b> &nbsp;|&nbsp;
  交易：<b>{int(row['總交易'])}次</b>
</div>""", unsafe_allow_html=True)
            else:
                st.info("目前無股票同時滿足勝率≥55%且報酬>0，建議調整參數或延長回測週期")

            # 下載比較結果
            csv = valid_cr.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("⬇️ 下載比較結果 CSV", csv,
                               f"compare_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.warning("所有股票回測均無有效結果，請延長回測週期或調低最低得分")
    else:
        st.info('👆 點擊「開始全部回測比較」，系統將對所有股票執行回測並排出勝率榜')

# ═══════════════════════════════════════════════════════════

# 頁腳

# ═══════════════════════════════════════════════════════════

st.markdown("—")
st.markdown("""

<div style="text-align:center;color:#444;font-size:12px;padding:10px;">
⚠️ 本系統僅供研究參考，回測績效不代表未來表現。股市有風險，投資請謹慎評估。<br>
過去勝率 ≠ 未來勝率 &nbsp;|&nbsp; 請結合基本面與市場環境綜合判斷
</div>""", unsafe_allow_html=True)
