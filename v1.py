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
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════════
# 頁面設定與樣式
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
# 信號評分邏輯
# ═══════════════════════════════════════════════════════════

def score_bar(row, prev_row):
    buy_score = sell_score = 0
    # 趨勢判斷
    if row['EMA5'] > row['EMA10'] > row['EMA20']:
        buy_score += 2
    elif row['EMA5'] < row['EMA10'] < row['EMA20']:
        sell_score += 2

    # MACD 交叉
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

    # 量價配合
    vol_ratio = row['Volume'] / row['VOL_MA5'] if row['VOL_MA5'] > 0 else 1
    if vol_ratio > 1.3 and row['Close'] > row['Open']:
        buy_score += 2
    elif vol_ratio > 1.3 and row['Close'] < row['Open']:
        sell_score += 2

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
        "EMA排列": "✅ 多頭" if buy_score >= 2 else ("🔴 空頭" if sell_score >= 2 else "⚪ 糾纏"),
        "MACD狀態": f"DIF={last['DIF']:.3f}",
        "成交量":   f"量比={last['Volume']/last['VOL_MA5']:.1f}x" if last['VOL_MA5'] > 0 else "N/A",
        "MA短期":   f"MA5={last['MA5']:.2f}",
        "得分":     f"買{buy_score}/賣{sell_score}",
    }
    
    buy_price = stop_loss = target = None
    if buy_score >= 5 and buy_score > sell_score:
        signal, buy_price = "買入", round(price, 2)
        stop_loss, target = round(price - 2*atr, 2), round(price + 3*atr, 2)
    elif sell_score >= 5 and sell_score > buy_score:
        signal, buy_price = "賣出", round(price, 2)
        stop_loss, target = round(price + 2*atr, 2), round(price - 3*atr, 2)
    else:
        signal = "觀望"
        
    return signal, buy_price, stop_loss, target, details

# ═══════════════════════════════════════════════════════════
# 回測與統計引擎
# ═══════════════════════════════════════════════════════════

def run_backtest(df, initial_capital=100000, shares_per_trade=100, atr_stop=2.0, atr_target=3.0, min_score=5):
    df_bt = df.copy().reset_index()
    results, capital = [], initial_capital
    eq_curve, in_trade = [capital], False
    trade_dir, entry_px, stop_px, target_px, entry_time, entry_idx = None, 0, 0, 0, None, 0

    for i in range(30, len(df_bt)):
        row, prev = df_bt.iloc[i], df_bt.iloc[i-1]
        ts = row.get('Datetime', row.get('Date', i))

        if in_trade:
            hi, lo, exited = float(row['High']), float(row['Low']), False
            if trade_dir == 'long':
                if lo <= stop_px: exit_px, exit_type, exited = stop_px, "止損", True
                elif hi >= target_px: exit_px, exit_type, exited = target_px, "獲利", True
            else:
                if hi >= stop_px: exit_px, exit_type, exited = stop_px, "止損", True
                elif lo <= target_px: exit_px, exit_type, exited = target_px, "獲利", True

            if exited:
                pnl = (exit_px - entry_px) * shares_per_trade if trade_dir == 'long' else (entry_px - exit_px) * shares_per_trade
                capital += pnl
                results.append({"方向": "做多" if trade_dir == 'long' else "做空", "進場時間": entry_time, "出場時間": ts, "進場價": round(entry_px, 2), "出場價": round(exit_px, 2), "盈虧(元)": round(pnl, 2), "資金餘額": round(capital, 2)})
                in_trade = False

        if not in_trade:
            atr_val = float(row['ATR'])
            if np.isnan(atr_val) or atr_val == 0: continue
            buy_s, sell_s = score_bar(row, prev)
            cl = float(row['Close'])
            if buy_s >= min_score:
                in_trade, trade_dir, entry_px, stop_px, target_px, entry_time, entry_idx = True, 'long', cl, cl - atr_stop * atr_val, cl + atr_target * atr_val, ts, i
            elif sell_s >= min_score:
                in_trade, trade_dir, entry_px, stop_px, target_px, entry_time, entry_idx = True, 'short', cl, cl + atr_stop * atr_val, cl - atr_target * atr_val, ts, i
        eq_curve.append(capital)
    return pd.DataFrame(results), eq_curve, capital

def calc_stats(trades_df, initial_capital, final_capital):
    if trades_df.empty: return {}
    wins = trades_df[trades_df['盈虧(元)'] > 0]
    total = len(trades_df)
    win_rate = len(wins) / total * 100
    equity = trades_df['資金餘額'].values
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak * 100).min() if (peak > 0).all() else 0
    return {
        "總交易次數": total, "勝率%": round(win_rate, 1), "獲利因子": round(abs(wins['盈虧(元)'].sum() / trades_df[trades_df['盈虧(元)']<0]['盈虧(元)'].sum()), 2) if not trades_df[trades_df['盈虧(元)']<0].empty else 99,
        "總報酬%": round((final_capital-initial_capital)/initial_capital*100, 2), "最大回撤%": round(max_dd, 2), "最終資金": final_capital,
        "獲利次數": len(wins), "虧損次數": total - len(wins), "總盈虧": round(final_capital - initial_capital, 2)
    }

# ═══════════════════════════════════════════════════════════
# 數據獲取與繪圖
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_data(ticker, period="5d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return calc_indicators(df)
    except: return None

def plot_main_chart(df, ticker, signal, buy_price, stop_loss, target):
    df_p = df.tail(100)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)
    for ma, color in [('EMA5','#00ff00'),('EMA20','#ff8800')]:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p[ma], name=ma, line=dict(width=1)), row=1, col=1)
    if buy_price:
        fig.add_hline(y=buy_price, line_color="white", line_dash="dot", row=1, col=1)
        fig.add_hline(y=stop_loss, line_color="red", line_dash="dash", row=1, col=1)
    fig.add_trace(go.Bar(x=df_p.index, y=df_p['Volume'], name="成交量", marker_color="gray"), row=2, col=1)
    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ═══════════════════════════════════════════════════════════
# 主程序流程
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ 設定")
    page = st.radio("功能", ["📡 實時掃描", "🔬 回測分析"])
    ticker_input = st.text_area("代碼 (每行一個)", "2330.TW\nNVDA\nAAPL\nTSLA", height=120)
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    interval = st.selectbox("週期", ["5m", "15m", "1h", "1d"], index=0)
    shares = st.number_input("每筆股數", 1, 10000, 100)
    min_score = st.slider("最低信號分", 3, 7, 5)

st.title("📈 股票智能掃描系統 v2.0")

if page == "📡 實時掃描":
    if st.button("🔍 開始掃描"):
        cols = st.columns(len(tickers))
        results = []
        for ticker in tickers:
            df = fetch_data(ticker, "5d", interval)
            if df is not None:
                sig, bp, sl, tg, det = generate_signal(df, shares)
                results.append({"ticker": ticker, "sig": sig, "bp": bp, "sl": sl, "tg": tg, "df": df, "det": det})
        
        # 顯示信號卡片
        for r in results:
            if r['sig'] == "買入":
                st.markdown(f'<div class="buy-signal">🟢 {r["ticker"]} | 建議買入: {r["bp"]} | 止損: {r["sl"]}</div>', unsafe_allow_html=True)
            elif r['sig'] == "賣出":
                st.markdown(f'<div class="sell-signal">🔴 {r["ticker"]} | 建議賣出: {r["bp"]} | 止損: {r["sl"]}</div>', unsafe_allow_html=True)
        
        if results:
            sel = st.selectbox("查看詳細圖表", [r['ticker'] for r in results])
            curr = next(r for r in results if r['ticker'] == sel)
            st.plotly_chart(plot_main_chart(curr['df'], sel, curr['sig'], curr['bp'], curr['sl'], curr['tg']), use_container_width=True)

elif page == "🔬 回測分析":
    target_tk = st.selectbox("選擇回測對象", tickers)
    if st.button("▶️ 執行回測"):
        df_bt = fetch_data(target_tk, "60d", interval)
        if df_bt is not None:
            trades, curve, final = run_backtest(df_bt, 100000, shares, 2.0, 3.0, min_score)
            stats = calc_stats(trades, 100000, final)
            
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("勝率", f"{stats['勝率%']}%")
                c2.metric("總報酬", f"{stats['總報酬%']}%")
                c3.metric("獲利因子", stats['獲利因子'])
                c4.metric("總交易", stats['總交易次數'])
                
                st.markdown("### 資金曲線")
                fig_curve = go.Figure(go.Scatter(y=curve, mode='lines', fill='tozeroy', line=dict(color='#00ff88')))
                fig_curve.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_curve, use_container_width=True)
                
                st.markdown("### 交易明細")
                st.dataframe(trades, use_container_width=True)
            else:
                st.warning("此區間無交易信號")
