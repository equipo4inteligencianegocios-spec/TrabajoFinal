"""pages/dashboard.py — Dashboard General · Resumen de las 6 empresas"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.data import EMPRESAS, MODELOS_CLASIFICACION, MODELOS_REGRESION, C, theme
from utils.data import get_ohlcv, get_all_predictions, consenso

def show():
    ticker = st.session_state.get("ticker", "FSM")
    emp    = EMPRESAS[ticker]

    st.title("📊 Dashboard General")
    st.caption("Componente 4 · Capa de Salida · "
               f"Empresa activa: **{ticker}** — {emp['nombre']} · "
               "Predicción: **día siguiente**")

    # ── Señales día siguiente · las 6 empresas ───────────────────
    st.subheader("🔮 Señal Día Siguiente · Las 6 Empresas Mineras")
    st.caption("Consenso de los 8 modelos (Comp. 2.1 + 2.2)")

    cols = st.columns(3)
    for idx, (tick, info) in enumerate(EMPRESAS.items()):
        df    = get_ohlcv(tick, 3)
        p_hoy = df["Close"].iloc[-1]
        p_ant = df["Close"].iloc[-2]
        chg   = (p_hoy / p_ant - 1) * 100
        preds = get_all_predictions(tick)
        rec   = consenso(preds)
        votos_s = rec["votos_subida"]
        total   = rec["total"]

        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-top:3px solid {rec['color']};border-radius:10px;
                        padding:16px;margin-bottom:14px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:17px;
                              font-weight:700;color:{info['color']}">{tick}</div>
                  <div style="font-size:11px;color:{C['muted']};margin-top:2px">{info['nombre'][:30]}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                              font-weight:600">${p_hoy:,.3f}</div>
                  <div style="font-size:11px;color:{'#00d4aa' if chg>=0 else '#ff4d6a'}">{chg:+.2f}% hoy</div>
                </div>
              </div>
              <div style="padding:10px 14px;background:{C['surf2']};border-radius:8px;
                          display:flex;justify-content:space-between;align-items:center">
                <div>
                  <div style="font-size:9px;color:{C['muted']};letter-spacing:1px;
                              text-transform:uppercase;margin-bottom:3px">Señal · día siguiente</div>
                  <div style="font-size:20px;font-weight:700;color:{rec['color']}">
                      {rec['emoji']} {rec['señal']}
                  </div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:10px;color:{C['muted']}">Consenso</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                              color:{rec['color']}">{votos_s}/{total} modelos</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Gráfico comparativo normalizado ──────────────────────────
    st.subheader("📈 Evolución Comparada · 90 días · Base 100")
    fig = go.Figure()
    for tick, info in EMPRESAS.items():
        df   = get_ohlcv(tick, 90)
        norm = df["Close"] / df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm.values, name=tick,
            line=dict(color=info["color"], width=1.8),
            hovertemplate=f"<b>{tick}</b> %{{y:.1f}}<extra></extra>",
        ))
    theme(fig)
    fig.update_layout(height=320, yaxis_title="Índice (base 100)",
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Detalle empresa activa ────────────────────────────────────
    st.subheader(f"🔬 Detalle de Predicciones · {ticker} · 8 Modelos")
    preds = get_all_predictions(ticker)
    rows  = []
    for mid, p in preds.items():
        if p["tipo"] == "clasificacion":
            detalle = f"Confianza {p['confianza']}%  |  Accuracy histórico {p['accuracy']*100:.1f}%"
        else:
            detalle = (f"Hoy ${p['precio_hoy']:,.4f}  →  Pred ${p['precio_pred']:,.4f}  "
                       f"({p['variacion_pct']:+.2f}%)  |  MAE {p['mae']:.3f}")
        rows.append({
            "Comp.":      p["comp"],
            "Modelo":     p["label"],
            "Módulo":     "Clasificación" if p["tipo"]=="clasificacion" else "Regresión",
            "Tendencia":  f"{'↑' if p['tendencia']=='SUBIDA' else '↓'} {p['tendencia']}",
            "Detalle":    detalle,
            "Señal":      "↑ COMPRAR" if p["tendencia"]=="SUBIDA" else "↓ VENDER",
        })

    df_tab = pd.DataFrame(rows)

    def _ct(v):
        if "SUBIDA"  in str(v) or "COMPRAR" in str(v): return f"color:{C['accent']};font-weight:700"
        if "BAJADA"  in str(v) or "VENDER"  in str(v): return f"color:{C['red']};font-weight:700"
        return ""

    st.dataframe(
        df_tab.style
              .applymap(_ct, subset=["Tendencia","Señal"])
              .set_properties(**{"background-color":C["surface"],"color":C["text"],"border-color":C["border"]}),
        use_container_width=True, hide_index=True,
    )

    rec = consenso(preds)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Señal consenso",    f"{rec['emoji']} {rec['señal']}")
    k2.metric("Votos SUBIDA",      f"{rec['votos_subida']}/{rec['total']} modelos")
    k3.metric("Precio actual",     f"${get_ohlcv(ticker,2)['Close'].iloc[-1]:,.4f} USD")
    k4.metric("Modelos evaluados", "8  (5 clas. + 3 reg.)")
