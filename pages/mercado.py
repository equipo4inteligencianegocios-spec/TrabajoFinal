"""pages/mercado.py — Datos del Mercado · Componente 1 (Yahoo Finance)"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.data import EMPRESAS, C, theme, get_ohlcv, add_indicators

def show():
    ticker = st.session_state.get("ticker", "FSM")
    emp    = EMPRESAS[ticker]

    st.title("📈 Datos del Mercado")
    st.caption(f"**Componente 1 · Yahoo Finance API** · {ticker} — {emp['nombre']}")

    # ── Controles ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        rango = st.radio("Período", ["1M","3M","6M","1A","2A"], horizontal=True, index=2)
    with c2:
        tipo_graf = st.selectbox("Tipo de gráfico", ["Velas (OHLC)", "Línea de cierre", "Área"])
    with c3:
        mostrar_vol = st.checkbox("Mostrar volumen", value=True)

    dias_map = {"1M":21,"3M":63,"6M":126,"1A":252,"2A":504}
    dias = dias_map[rango]

    df  = get_ohlcv(ticker, dias)
    df  = add_indicators(df)

    # ── KPIs del día ─────────────────────────────────────────────
    p_hoy  = df["Close"].iloc[-1]
    p_ant  = df["Close"].iloc[-2]
    chg    = (p_hoy / p_ant - 1) * 100
    hi52   = df["High"].max()
    lo52   = df["Low"].min()
    vol_m  = df["Volume"].iloc[-1] / 1e6

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Precio de cierre",    f"${p_hoy:,.4f}",  f"{chg:+.2f}% vs ayer")
    k2.metric("Apertura",            f"${df['Open'].iloc[-1]:,.4f}")
    k3.metric("Máximo 52 sem.",      f"${hi52:,.4f}")
    k4.metric("Mínimo 52 sem.",      f"${lo52:,.4f}")
    k5.metric("Volumen (millones)",  f"{vol_m:.2f}M")

    # ── Gráfico principal ─────────────────────────────────────────
    rows = 3 if mostrar_vol else 2
    row_heights = [0.55, 0.25, 0.20] if mostrar_vol else [0.65, 0.35]
    subplot_titles = [f"{ticker} · Precio", "RSI 14", "Volumen"] if mostrar_vol \
                else [f"{ticker} · Precio", "RSI 14"]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.04)

    # Panel 1: precio
    if tipo_graf == "Velas (OHLC)":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            name="OHLC",
            increasing_line_color=C["accent"],
            decreasing_line_color=C["red"],
            increasing_fillcolor=C["accent"],
            decreasing_fillcolor=C["red"],
        ), row=1, col=1)
    elif tipo_graf == "Línea de cierre":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Cierre",
            line=dict(color=C["accent"], width=2),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Cierre",
            line=dict(color=C["accent"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
        ), row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20",
        line=dict(color=C["gold"],   width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA 50",
        line=dict(color=C["blue"],   width=1.2, dash="dot")), row=1, col=1)
    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Superior",
        line=dict(color=C["purple"], width=0.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Inferior",
        line=dict(color=C["purple"], width=0.8, dash="dash"),
        fill="tonexty", fillcolor="rgba(155,109,255,0.04)"), row=1, col=1)

    # Panel 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI 14",
        line=dict(color=C["gold"], width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=C["red"],    row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=C["accent"], row=2, col=1)

    # Panel 3: Volumen
    if mostrar_vol:
        colors_vol = [C["accent"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else C["red"]
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volumen",
                             marker_color=colors_vol, opacity=0.6), row=3, col=1)

    theme(fig)
    fig.update_layout(height=600, title_text="", showlegend=True,
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=-0.06))
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",          row=2, col=1, range=[0, 100])
    if mostrar_vol:
        fig.update_yaxes(title_text="Volumen", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── MACD subplot independiente ────────────────────────────────
    with st.expander("📊 MACD — Detalle"):
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD",
            line=dict(color=C["accent"], width=1.6)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Señal",
            line=dict(color=C["gold"],   width=1.4, dash="dot")))
        colors_h = [C["accent"] if v >= 0 else C["red"] for v in df["MACD_hist"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"],
            name="Histograma", marker_color=colors_h, opacity=0.7))
        theme(fig_macd)
        fig_macd.update_layout(height=220, title_text="MACD",
                                legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_macd, use_container_width=True)

    # ── Tabla OHLCV últimos 10 días ───────────────────────────────
    st.subheader("📋 Últimas sesiones · OHLCV")
    df_show = df[["Open","High","Low","Close","Volume"]].tail(10).copy()
    df_show.index = df_show.index.strftime("%d %b %Y")
    df_show["Variación %"] = df_show["Close"].pct_change() * 100

    def _cv(v):
        try: return f"color:{C['accent']}" if float(v)>=0 else f"color:{C['red']}"
        except: return ""

    st.dataframe(
        df_show.style
               .applymap(_cv, subset=["Variación %"])
               .format({"Open":"${:.4f}","High":"${:.4f}","Low":"${:.4f}",
                        "Close":"${:.4f}","Volume":"{:,.0f}","Variación %":"{:+.2f}%"})
               .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
        use_container_width=True,
    )
