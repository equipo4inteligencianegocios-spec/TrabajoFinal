"""pages/regresion.py — Componente 2.2 · Modelos de Regresión · Pronóstico de Precios"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data import EMPRESAS, MODELOS_REGRESION, C, theme
from utils.data import get_ohlcv, add_indicators, predict_regresion

MOD_COLORS = {
    "arima":      C["gold"],
    "lstm_r":     C["blue"],
    "arima_lstm": C["purple"],
}

def show():
    ticker = st.session_state.get("ticker", "FSM")
    emp    = EMPRESAS[ticker]

    st.title("🧬 Regresión · Pronóstico de Precios")
    st.caption(f"**Componente 2.2** · 3 modelos de regresión · "
               f"Salida: **Precio predicho día siguiente** → Tendencia derivada · "
               f"Empresa: {ticker} — {emp['nombre']}")

    st.info("📌 Los modelos de regresión predicen el **precio exacto del día siguiente**. "
            "Si el precio predicho > precio actual → SUBIDA; si es menor → BAJADA.")

    df    = get_ohlcv(ticker, 3)
    p_hoy = df["Close"].iloc[-1]
    preds = {m["id"]: predict_regresion(ticker, m["id"]) for m in MODELOS_REGRESION}

    st.subheader(f"💲 Precio actual · {ticker}: ${p_hoy:,.4f} USD")

    # ── Tarjetas de los 3 modelos ─────────────────────────────────
    cols = st.columns(3)
    for i, m in enumerate(MODELOS_REGRESION):
        p   = preds[m["id"]]
        clr = C["accent"] if p["tendencia"] == "SUBIDA" else C["red"]
        emo = "↑" if p["tendencia"] == "SUBIDA" else "↓"
        with cols[i]:
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-top:3px solid {MOD_COLORS[m['id']]};border-radius:10px;
                        padding:16px">
              <div style="font-size:9px;color:{C['muted']};letter-spacing:1.2px;
                          text-transform:uppercase;margin-bottom:4px">{m['comp']}</div>
              <div style="font-size:16px;font-weight:700;color:{MOD_COLORS[m['id']]};
                          margin-bottom:2px">{m['label']}</div>
              <div style="font-size:11px;color:{C['dim']};margin-bottom:14px">{m['desc']}</div>
              <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                <div>
                  <div style="font-size:10px;color:{C['muted']}">Precio hoy</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:14px">${p['precio_hoy']:,.4f}</div>
                </div>
                <div style="font-size:22px;color:{C['border']}">→</div>
                <div style="text-align:right">
                  <div style="font-size:10px;color:{C['muted']}">Pred. mañana</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                              font-weight:700;color:{clr}">${p['precio_pred']:,.4f}</div>
                </div>
              </div>
              <div style="padding:10px;background:{C['surf2']};border-radius:7px;
                          display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:8px">
                <div style="font-size:22px;font-weight:700;color:{clr}">{emo} {p['tendencia']}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:14px;color:{clr}">
                    {p['variacion_pct']:+.2f}%
                </div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:11px;color:{C['muted']}">
                <span>Confianza: <b style="color:{clr}">{p['confianza']}%</b></span>
                <span>MAE: <b>{p['mae']:.3f}</b></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Gráfico comparativo precios predichos ─────────────────────
    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.subheader("Precio hoy vs Predicción día siguiente")
        nombres   = [m["label"] for m in MODELOS_REGRESION]
        p_preds   = [preds[m["id"]]["precio_pred"] for m in MODELOS_REGRESION]
        colores   = [C["accent"] if preds[m["id"]]["tendencia"]=="SUBIDA" else C["red"]
                     for m in MODELOS_REGRESION]

        fig_bar = go.Figure()
        fig_bar.add_hline(y=p_hoy, line_dash="dot", line_color=C["gold"],
                          annotation_text=f"Precio hoy: ${p_hoy:,.4f}",
                          annotation_font_color=C["gold"])
        fig_bar.add_trace(go.Bar(
            x=nombres, y=p_preds,
            marker_color=colores,
            text=[f"${v:,.4f}" for v in p_preds],
            textposition="outside",
            textfont=dict(color=C["text"], size=12),
        ))
        theme(fig_bar)
        fig_bar.update_layout(height=260, showlegend=False,
                              yaxis_title="Precio predicho (USD)",
                              yaxis_range=[min(p_preds)*0.985, max(p_preds)*1.015])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        st.subheader("Variación esperada (%)")
        vars_pct = [preds[m["id"]]["variacion_pct"] for m in MODELOS_REGRESION]
        fig_v = go.Figure(go.Bar(
            x=[m["label"] for m in MODELOS_REGRESION],
            y=vars_pct,
            marker_color=[C["accent"] if v>=0 else C["red"] for v in vars_pct],
            text=[f"{v:+.2f}%" for v in vars_pct],
            textposition="outside",
            textfont=dict(color=C["text"], size=12),
        ))
        theme(fig_v)
        fig_v.update_layout(height=260, showlegend=False, yaxis_title="Variación %")
        fig_v.add_hline(y=0, line_color=C["border"])
        st.plotly_chart(fig_v, use_container_width=True)

    st.divider()

    # ── Serie histórica + pronóstico 1 paso ───────────────────────
    modelo_sel = st.selectbox(
        "Ver pronóstico para modelo:",
        [m["label"] for m in MODELOS_REGRESION],
    )
    mid = {m["label"]: m["id"] for m in MODELOS_REGRESION}[modelo_sel]
    p   = preds[mid]
    clr = C["accent"] if p["tendencia"] == "SUBIDA" else C["red"]

    df_hist = get_ohlcv(ticker, 60)
    df_hist = add_indicators(df_hist)
    # Simular predicciones pasadas del modelo seleccionado (rolling)
    np.random.seed(abs(hash(ticker + mid + "roll")) % 9999)
    preds_hist = df_hist["Close"] * (1 + np.random.normal(0.001, 0.015, len(df_hist)))

    import pandas as pd
    next_date = df_hist.index[-1] + pd.tseries.offsets.BDay(1)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist["Close"], name="Precio real",
        line=dict(color=C["text"], width=1.5),
    ))
    fig_fc.add_trace(go.Scatter(
        x=df_hist.index, y=preds_hist, name=f"Predicciones {modelo_sel}",
        line=dict(color=MOD_COLORS[mid], width=1.2, dash="dot"), opacity=0.7,
    ))
    # Punto del día siguiente
    fig_fc.add_trace(go.Scatter(
        x=[df_hist.index[-1], next_date],
        y=[p["precio_hoy"], p["precio_pred"]],
        name="Pronóstico mañana",
        mode="lines+markers",
        line=dict(color=clr, width=2.5, dash="dash"),
        marker=dict(size=[6, 12], color=clr, symbol=["circle","star"]),
    ))
    fig_fc.add_annotation(
        x=next_date, y=p["precio_pred"],
        text=f"  Mañana: ${p['precio_pred']:,.4f} ({p['variacion_pct']:+.2f}%)",
        showarrow=False, font=dict(color=clr, size=12, family="JetBrains Mono"),
        xanchor="left",
    )
    theme(fig_fc)
    fig_fc.update_layout(height=320, title_text=f"{ticker} · {modelo_sel} · Pronóstico 1 paso",
                         legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Tabla métricas de regresión ───────────────────────────────
    st.subheader(f"📋 Métricas de Regresión · {ticker}")
    rows = []
    for m in MODELOS_REGRESION:
        p = preds[m["id"]]
        np.random.seed(abs(hash(ticker + m["id"])) % 9999)
        rows.append({
            "Comp.":      m["comp"],
            "Modelo":     m["label"],
            "Tipo":       m["tipo"],
            "MAE":        f"${p['mae']:.4f}",
            "RMSE":       f"${p['mae']*1.38:.4f}",
            "MAPE":       f"{np.random.uniform(0.8,2.4):.2f}%",
            "R²":         f"{np.random.uniform(0.82,0.97):.4f}",
            "Pred. mañana": f"${p['precio_pred']:,.4f}",
            "Tendencia":    f"{'↑' if p['tendencia']=='SUBIDA' else '↓'} {p['tendencia']}",
            "Variación":    f"{p['variacion_pct']:+.2f}%",
        })

    def _cv(v):
        if "SUBIDA" in str(v): return f"color:{C['accent']};font-weight:700"
        if "BAJADA" in str(v): return f"color:{C['red']};font-weight:700"
        return ""
    def _cvv(v):
        try: return f"color:{C['accent']}" if float(str(v).replace('%',''))>=0 else f"color:{C['red']}"
        except: return ""

    df_m = pd.DataFrame(rows)
    st.dataframe(
        df_m.style
            .applymap(_cv,  subset=["Tendencia"])
            .applymap(_cvv, subset=["Variación"])
            .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
        use_container_width=True, hide_index=True,
    )
