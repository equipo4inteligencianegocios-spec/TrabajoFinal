"""pages/backtesting.py — Componente 3 · Backtesting con VectorBT"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data import (EMPRESAS, MODELOS_CLASIFICACION, MODELOS_REGRESION,
                        TODOS_MODELOS, C, theme, get_ohlcv, get_backtest_metrics)

def show():
    ticker = st.session_state.get("ticker", "FSM")
    emp    = EMPRESAS[ticker]

    st.title("⚖ Backtesting · VectorBT")
    st.caption(f"**Componente 3** · Evaluación histórica de estrategias · "
               f"Empresa: {ticker} — {emp['nombre']}")

    st.info(
        "📌 El backtesting evalúa retrospectivamente qué tan rentable habría sido "
        "seguir las señales de cada modelo en el pasado. "
        "**En producción**: usar `vectorbt` con las señales reales generadas por cada modelo."
    )

    # ── Controles ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        empresa_bt = st.selectbox("Empresa para backtesting",
            list(EMPRESAS.keys()),
            index=list(EMPRESAS.keys()).index(ticker),
            format_func=lambda t: f"{t} · {EMPRESAS[t]['nombre'][:28]}")
    with c2:
        periodo = st.selectbox("Período", ["6 meses","1 año","2 años"], index=1)

    dias_map = {"6 meses": 126, "1 año": 252, "2 años": 504}
    dias = dias_map[periodo]

    # ── Tabla comparativa todos los modelos ───────────────────────
    st.subheader(f"📊 Comparativa de Modelos · {empresa_bt} · {periodo}")

    all_metrics = []
    for m in TODOS_MODELOS:
        mt = get_backtest_metrics(empresa_bt, m["id"])
        all_metrics.append({
            "Comp.":         m["comp"],
            "Modelo":        m["label"],
            "Módulo":        "Clasificación" if m in MODELOS_CLASIFICACION else "Regresión",
            "Retorno Total": mt["total_return"],
            "Sharpe":        mt["sharpe"],
            "Sortino":       mt["sortino"],
            "Calmar":        mt["calmar"],
            "Max Drawdown":  mt["max_drawdown"],
            "Win Rate":      mt["win_rate"],
            "Trades":        mt["total_trades"],
        })

    df_cmp = pd.DataFrame(all_metrics).sort_values("Retorno Total", ascending=False)

    def _cr(v):
        try:
            val = float(v)
            return f"color:{C['accent']};font-weight:700" if val>0 else f"color:{C['red']};font-weight:700"
        except: return ""

    st.dataframe(
        df_cmp.style
              .applymap(_cr, subset=["Retorno Total","Max Drawdown"])
              .format({"Retorno Total":"{:+.2%}","Sharpe":"{:.3f}","Sortino":"{:.3f}",
                       "Calmar":"{:.3f}","Max Drawdown":"{:.2%}","Win Rate":"{:.1%}","Trades":"{:.0f}"})
              .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
        use_container_width=True, hide_index=True,
    )

    # ── Gráfico de barras: retorno por modelo ─────────────────────
    st.subheader("📈 Retorno Total por Modelo")
    fig_ret = go.Figure(go.Bar(
        x=df_cmp["Modelo"],
        y=df_cmp["Retorno Total"],
        marker_color=[C["accent"] if v>0 else C["red"] for v in df_cmp["Retorno Total"]],
        text=[f"{v:+.1%}" for v in df_cmp["Retorno Total"]],
        textposition="outside",
        textfont=dict(color=C["text"], size=12),
    ))
    theme(fig_ret)
    fig_ret.add_hline(y=0, line_color=C["border"])
    fig_ret.update_layout(height=280, showlegend=False, yaxis_tickformat=".0%",
                          yaxis_title="Retorno Total")
    st.plotly_chart(fig_ret, use_container_width=True)

    st.divider()

    # ── Equity curves simuladas ───────────────────────────────────
    st.subheader(f"📉 Curvas de Capital (Equity Curves) · {empresa_bt}")
    st.caption("Evolución del capital $10,000 si se siguieran las señales de cada modelo")

    modelo_sel = st.multiselect(
        "Modelos a comparar",
        [m["label"] for m in TODOS_MODELOS],
        default=[m["label"] for m in TODOS_MODELOS[:4]],
    )

    df_price = get_ohlcv(empresa_bt, dias)
    fig_eq   = go.Figure()

    all_colors = [C["accent"],C["blue"],C["gold"],C["purple"],C["red"],C["dim"],C["muted"],C["text"]]
    label_to_id = {m["label"]: m["id"] for m in TODOS_MODELOS}
    label_to_color = {m["label"]: all_colors[i] for i, m in enumerate(TODOS_MODELOS)}

    for label in modelo_sel:
        mid    = label_to_id[label]
        mt     = get_backtest_metrics(empresa_bt, mid)
        np.random.seed(abs(hash(empresa_bt + mid + "eq")) % 9999)
        # Simular equity curve con el retorno total
        n      = len(df_price)
        rets   = np.random.normal(mt["total_return"]/n, 0.018, n)
        equity = 10_000 * np.cumprod(1 + rets)
        fig_eq.add_trace(go.Scatter(
            x=df_price.index, y=equity,
            name=f"{label} ({mt['total_return']:+.1%})",
            line=dict(color=label_to_color[label], width=1.8),
        ))

    # Buy & Hold como referencia
    bh = 10_000 * df_price["Close"] / df_price["Close"].iloc[0]
    fig_eq.add_trace(go.Scatter(
        x=df_price.index, y=bh,
        name="Buy & Hold", line=dict(color=C["muted"], width=1.2, dash="dot"),
    ))
    theme(fig_eq)
    fig_eq.update_layout(height=360, title_text="",
                         yaxis_title="Capital (USD)",
                         legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ── Mejor modelo + recomendación ─────────────────────────────
    st.subheader(f"🏆 Mejor Modelo · {empresa_bt}")
    best = df_cmp.iloc[0]
    clr  = C["accent"] if best["Retorno Total"] > 0 else C["red"]

    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("🥇 Mejor modelo",    str(best["Modelo"]))
    b2.metric("Retorno Total",      f"{best['Retorno Total']:+.2%}")
    b3.metric("Sharpe Ratio",       f"{best['Sharpe']:.3f}")
    b4.metric("Win Rate",           f"{best['Win Rate']:.1%}")
    b5.metric("Max Drawdown",       f"{best['Max Drawdown']:.2%}")

    st.success(
        f"✅ El modelo **{best['Modelo']}** obtuvo el mayor retorno ({best['Retorno Total']:+.2%}) "
        f"para **{empresa_bt}** en el período seleccionado, con un Sharpe de **{best['Sharpe']:.2f}** "
        f"y Win Rate de **{best['Win Rate']:.1%}**. Se recomienda usar como modelo primario."
    )

    # ── Drawdown chart del mejor modelo ──────────────────────────
    st.subheader(f"📉 Drawdown · {best['Modelo']} · {empresa_bt}")
    np.random.seed(abs(hash(empresa_bt + str(best["Modelo"]) + "dd")) % 9999)
    n       = len(df_price)
    eq_best = 10_000 * np.cumprod(1 + np.random.normal(
        best["Retorno Total"]/n, 0.016, n))
    rolling_max = pd.Series(eq_best).cummax()
    drawdown    = (pd.Series(eq_best) - rolling_max) / rolling_max

    fig_dd = go.Figure(go.Scatter(
        x=df_price.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(255,77,106,0.12)",
        line=dict(color=C["red"], width=1.5),
        name="Drawdown",
    ))
    theme(fig_dd)
    fig_dd.update_layout(height=220, yaxis_tickformat=".0%",
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)
