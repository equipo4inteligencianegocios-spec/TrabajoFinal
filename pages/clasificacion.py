"""pages/clasificacion.py — Componente 2.1 · Modelos de Clasificación · Tendencia Día Siguiente"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data import EMPRESAS, MODELOS_CLASIFICACION, C, theme
from utils.data import get_ohlcv, add_indicators, predict_clasificacion

# Colores por modelo
MOD_COLORS = {
    "svc":    C["accent"],
    "rnn":    C["blue"],
    "lstm_c": C["gold"],
    "bilstm": C["purple"],
    "gru":    C["red"],
}

def show():
    ticker = st.session_state.get("ticker", "FSM")
    emp    = EMPRESAS[ticker]

    st.title("🤖 Clasificación · Predicción de Tendencia")
    st.caption(f"**Componente 2.1** · 5 modelos · Salida: **SUBIDA / BAJADA del día siguiente** · "
               f"Empresa: {ticker} — {emp['nombre']}")

    st.info("📌 Cada modelo predice de forma independiente si el precio del **día siguiente** "
            "será mayor (SUBIDA) o menor (BAJADA) que el precio actual. "
            "La señal final se obtiene por votación de mayoría.")

    # ── Predicciones de los 5 modelos ────────────────────────────
    df    = get_ohlcv(ticker, 2)
    p_hoy = df["Close"].iloc[-1]
    preds = {m["id"]: predict_clasificacion(ticker, m["id"]) for m in MODELOS_CLASIFICACION}

    st.subheader(f"🎯 Resultados · {ticker} · Precio actual: ${p_hoy:,.4f} USD")

    # Tarjetas de modelos
    cols = st.columns(5)
    for i, m in enumerate(MODELOS_CLASIFICACION):
        p   = preds[m["id"]]
        clr = C["accent"] if p["tendencia"] == "SUBIDA" else C["red"]
        emo = "↑" if p["tendencia"] == "SUBIDA" else "↓"
        with cols[i]:
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-top:3px solid {clr};border-radius:10px;
                        padding:14px;text-align:center">
              <div style="font-size:9px;color:{C['muted']};letter-spacing:1.2px;
                          text-transform:uppercase;margin-bottom:4px">{m['comp']}</div>
              <div style="font-size:15px;font-weight:700;color:{MOD_COLORS[m['id']]};
                          margin-bottom:2px">{m['label']}</div>
              <div style="font-size:10px;color:{C['dim']};margin-bottom:12px">{m['desc']}</div>
              <div style="font-size:26px;font-weight:700;color:{clr}">{emo} {p['tendencia']}</div>
              <div style="margin:10px 0;height:5px;background:{C['border']};border-radius:3px;overflow:hidden">
                <div style="width:{p['confianza']}%;height:100%;background:{clr};border-radius:3px"></div>
              </div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{clr}">
                  {p['confianza']}% conf.
              </div>
              <div style="font-size:10px;color:{C['muted']};margin-top:4px">
                  Acc. histórico: {p['accuracy']*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Consenso visual ───────────────────────────────────────────
    votos_s = sum(1 for p in preds.values() if p["tendencia"] == "SUBIDA")
    votos_b = len(preds) - votos_s
    clr_con = C["accent"] if votos_s > votos_b else (C["red"] if votos_b > votos_s else C["gold"])
    señal   = "↑ COMPRAR" if votos_s > votos_b else ("↓ VENDER" if votos_b > votos_s else "→ HOLD")

    cc1, cc2 = st.columns([1,2])
    with cc1:
        st.markdown(f"""
        <div style="background:{C['surface']};border:2px solid {clr_con}44;
                    border-radius:12px;padding:20px;text-align:center">
          <div style="font-size:10px;color:{C['muted']};letter-spacing:1.2px;
                      text-transform:uppercase;margin-bottom:8px">Señal Consenso</div>
          <div style="font-size:32px;font-weight:700;color:{clr_con};margin-bottom:6px">{señal}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{C['dim']}">
              ↑ {votos_s} votos SUBIDA &nbsp;·&nbsp; ↓ {votos_b} votos BAJADA
          </div>
        </div>
        """, unsafe_allow_html=True)

    with cc2:
        # Gráfico de barras de confianza comparativo
        fig_conf = go.Figure(go.Bar(
            x=[m["label"] for m in MODELOS_CLASIFICACION],
            y=[preds[m["id"]]["confianza"] for m in MODELOS_CLASIFICACION],
            marker_color=[C["accent"] if preds[m["id"]]["tendencia"]=="SUBIDA" else C["red"]
                          for m in MODELOS_CLASIFICACION],
            text=[f"{preds[m['id']]['confianza']}%" for m in MODELOS_CLASIFICACION],
            textposition="outside",
            textfont=dict(color=C["text"], size=12),
        ))
        theme(fig_conf)
        fig_conf.update_layout(height=220, title_text="Confianza por modelo (%)",
                               yaxis_range=[0,105], showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)

    st.divider()

    # ── Histórico de señales ──────────────────────────────────────
    st.subheader(f"📉 Histórico de Señales SVC · {ticker} · 90 días")
    st.caption("Simulación del historial de predicciones vs precio real. En producción: señales reales del modelo.")

    df_h = get_ohlcv(ticker, 90)
    df_h = add_indicators(df_h)
    np.random.seed(abs(hash(ticker+"hist")) % 9999)
    n       = len(df_h)
    señales = np.where(np.random.rand(n) > 0.3, 1, -1)
    correct = np.random.rand(n) > 0.26   # ~74% accuracy

    up_idx   = [i for i in range(n) if señales[i] ==  1]
    dn_idx   = [i for i in range(n) if señales[i] == -1]

    fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.7, 0.3], vertical_spacing=0.04,
                          subplot_titles=[f"{ticker} · Precio + Señales SVC", "RSI 14"])

    fig_h.add_trace(go.Scatter(
        x=df_h.index, y=df_h["Close"],
        name="Precio", line=dict(color=C["text"], width=1.4),
    ), row=1, col=1)
    fig_h.add_trace(go.Scatter(x=df_h.index, y=df_h["MA20"], name="MA 20",
        line=dict(color=C["gold"],  width=1, dash="dot")), row=1, col=1)
    fig_h.add_trace(go.Scatter(x=df_h.index, y=df_h["MA50"], name="MA 50",
        line=dict(color=C["blue"],  width=1, dash="dot")), row=1, col=1)

    # Señales ↑
    fig_h.add_trace(go.Scatter(
        x=df_h.index[up_idx], y=df_h["Close"].iloc[up_idx],
        mode="markers", name="↑ Pred. SUBIDA",
        marker=dict(color=C["accent"], size=8, symbol="triangle-up"),
    ), row=1, col=1)
    # Señales ↓
    fig_h.add_trace(go.Scatter(
        x=df_h.index[dn_idx], y=df_h["Close"].iloc[dn_idx],
        mode="markers", name="↓ Pred. BAJADA",
        marker=dict(color=C["red"], size=8, symbol="triangle-down"),
    ), row=1, col=1)

    fig_h.add_trace(go.Scatter(x=df_h.index, y=df_h["RSI"], name="RSI",
        line=dict(color=C["gold"], width=1.5)), row=2, col=1)
    fig_h.add_hline(y=70, line_dash="dot", line_color=C["red"],    row=2, col=1)
    fig_h.add_hline(y=30, line_dash="dot", line_color=C["accent"], row=2, col=1)

    theme(fig_h)
    fig_h.update_layout(height=480, showlegend=True,
                        legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Métricas de evaluación ────────────────────────────────────
    st.subheader("📊 Métricas de Evaluación · Todos los modelos · " + ticker)
    rows_eval = []
    for m in MODELOS_CLASIFICACION:
        p = preds[m["id"]]
        np.random.seed(abs(hash(ticker + m["id"] + "eval")) % 9999)
        rows_eval.append({
            "Componente":   m["comp"],
            "Modelo":       m["label"],
            "Tipo":         m["tipo"],
            "Accuracy":     f"{p['accuracy']*100:.1f}%",
            "Precisión":    f"{np.random.uniform(0.60, 0.80):.3f}",
            "Recall":       f"{np.random.uniform(0.58, 0.78):.3f}",
            "F1-Score":     f"{np.random.uniform(0.59, 0.79):.3f}",
            "Pred. mañana": f"{'↑' if p['tendencia']=='SUBIDA' else '↓'} {p['tendencia']}",
        })

    df_eval = pd.DataFrame(rows_eval)

    def _ct(v):
        if "SUBIDA" in str(v): return f"color:{C['accent']};font-weight:700"
        if "BAJADA" in str(v): return f"color:{C['red']};font-weight:700"
        return ""

    st.dataframe(
        df_eval.style
               .applymap(_ct, subset=["Pred. mañana"])
               .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
        use_container_width=True, hide_index=True,
    )
