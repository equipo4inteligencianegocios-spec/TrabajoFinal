"""pages/portafolio.py — Componente 4 · Portafolio + Recomendaciones + Broker"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data import EMPRESAS, TODOS_MODELOS, C, theme
from utils.data import get_ohlcv, get_all_predictions, consenso

# Portafolio de ejemplo (reemplazar con datos reales IB)
PORTAFOLIO_BASE = {
    "FSM":  {"acciones": 500,  "precio_entrada": 3.90},
    "BVN":  {"acciones": 150,  "precio_entrada": 9.20},
    "ABX":  {"acciones": 100,  "precio_entrada": 16.80},
    "SCCO": {"acciones": 20,   "precio_entrada": 88.00},
}

def show():
    ticker = st.session_state.get("ticker", "FSM")

    st.title("💼 Portafolio y Recomendaciones")
    st.caption("**Componente 4 · Capa de Salida** · Optimización del portafolio del día · "
               "Conexión Interactive Brokers API")

    tabs = st.tabs([
        "📋 Portafolio actual",
        "🎯 Recomendaciones del día",
        "📤 Envío de órdenes al Broker",
        "📊 Optimización del portafolio",
    ])

    # ════════════════════════════════════
    # TAB 1: Portafolio actual
    # ════════════════════════════════════
    with tabs[0]:
        st.subheader("Posiciones actuales · Interactive Brokers")
        rows = []
        total_val = 0
        for tick, pos in PORTAFOLIO_BASE.items():
            df      = get_ohlcv(tick, 2)
            p_hoy   = df["Close"].iloc[-1]
            p_ant   = df["Close"].iloc[-2]
            valor   = p_hoy * pos["acciones"]
            pnl     = (p_hoy - pos["precio_entrada"]) * pos["acciones"]
            pnl_pct = (p_hoy / pos["precio_entrada"] - 1) * 100
            chg_hoy = (p_hoy / p_ant - 1) * 100
            total_val += valor
            rows.append({
                "Ticker":          tick,
                "Empresa":         EMPRESAS[tick]["nombre"][:28],
                "Acciones":        pos["acciones"],
                "P. Entrada":      pos["precio_entrada"],
                "P. Actual":       round(p_hoy, 4),
                "Valor USD":       round(valor, 2),
                "P&L USD":         round(pnl, 2),
                "P&L %":           round(pnl_pct, 2),
                "Cambio hoy %":    round(chg_hoy, 2),
            })

        df_port = pd.DataFrame(rows)

        def _cpnl(v):
            try: return f"color:{C['accent']}" if float(v)>=0 else f"color:{C['red']}"
            except: return ""

        st.dataframe(
            df_port.style
                   .applymap(_cpnl, subset=["P&L USD","P&L %","Cambio hoy %"])
                   .format({"P. Entrada":"${:.4f}","P. Actual":"${:.4f}",
                            "Valor USD":"${:,.2f}","P&L USD":"${:+,.2f}",
                            "P&L %":"{:+.2f}%","Cambio hoy %":"{:+.2f}%"})
                   .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
            use_container_width=True, hide_index=True,
        )

        kk1,kk2,kk3 = st.columns(3)
        total_pnl = sum(r["P&L USD"] for r in rows)
        kk1.metric("Valor total portafolio", f"${total_val:,.2f} USD")
        kk2.metric("P&L total",              f"${total_pnl:+,.2f} USD",
                   delta_color="normal")
        kk3.metric("Posiciones abiertas",    f"{len(rows)} empresas")

        # Donut
        fig_d = go.Figure(go.Pie(
            labels=df_port["Ticker"], values=df_port["Valor USD"],
            hole=0.55,
            marker=dict(colors=[EMPRESAS[t]["color"] for t in df_port["Ticker"]]),
        ))
        theme(fig_d)
        fig_d.update_layout(
            height=280,
            annotations=[dict(text=f"${total_val:,.0f}", x=0.5, y=0.5,
                              font=dict(size=14, color=C["accent"],
                                        family="JetBrains Mono"), showarrow=False)],
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # ════════════════════════════════════
    # TAB 2: Recomendaciones del día
    # ════════════════════════════════════
    with tabs[1]:
        st.subheader("🎯 Recomendaciones · Todas las empresas · Día siguiente")
        st.caption("Basadas en consenso de los 8 modelos (Comp. 2.1 + 2.2)")

        recom_rows = []
        for tick, info in EMPRESAS.items():
            df    = get_ohlcv(tick, 2)
            p_hoy = df["Close"].iloc[-1]
            preds = get_all_predictions(tick)
            rec   = consenso(preds)
            # Precio predicho promedio (regresores)
            reg_preds = [preds[m["id"]]["precio_pred"] for m in
                         [x for x in TODOS_MODELOS if x["id"] in ["arima","lstm_r","arima_lstm"]]]
            p_pred_avg = np.mean(reg_preds)

            if rec["señal"] == "COMPRAR":
                operacion = "COMPRA (inversión)"
                descripcion = f"Mayoría de modelos ({rec['votos_subida']}/{rec['total']}) predicen subida"
            elif rec["señal"] == "VENDER":
                operacion = "VENTA / SHORT"
                descripcion = f"Mayoría de modelos predicen bajada. Considerar short selling"
            else:
                operacion = "HOLD (conservar)"
                descripcion = "Señales mixtas. Mantener posición actual"

            recom_rows.append({
                "Ticker":      tick,
                "P. Actual":   round(p_hoy, 4),
                "P. Pred.":    round(p_pred_avg, 4),
                "Var. est.":   round((p_pred_avg/p_hoy-1)*100, 2),
                "Consenso":    f"{rec['emoji']} {rec['señal']}",
                "Votos ↑":    f"{rec['votos_subida']}/{rec['total']}",
                "Operación":   operacion,
                "Descripción": descripcion,
            })

        df_rec = pd.DataFrame(recom_rows)

        def _crec(v):
            if "COMPRA" in str(v) or "COMPRAR" in str(v): return f"color:{C['accent']};font-weight:700"
            if "VENTA"  in str(v) or "VENDER"  in str(v): return f"color:{C['red']};font-weight:700"
            if "HOLD"   in str(v):                        return f"color:{C['gold']};font-weight:700"
            return ""

        def _cvar(v):
            try: return f"color:{C['accent']}" if float(str(v).replace('%',''))>=0 else f"color:{C['red']}"
            except: return ""

        st.dataframe(
            df_rec.style
                  .applymap(_crec, subset=["Consenso","Operación"])
                  .applymap(_cvar, subset=["Var. est."])
                  .format({"P. Actual":"${:.4f}","P. Pred.":"${:.4f}","Var. est.":"{:+.2f}%"})
                  .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
            use_container_width=True, hide_index=True,
        )

        # Gráfico de señales
        fig_s = go.Figure()
        for _, row in df_rec.iterrows():
            clr = C["accent"] if "COMPRAR" in row["Consenso"] else \
                  C["red"]    if "VENDER"  in row["Consenso"] else C["gold"]
            fig_s.add_trace(go.Bar(
                x=[row["Ticker"]], y=[row["Var. est."]],
                marker_color=clr, showlegend=False,
                text=f"{row['Var. est.']:+.2f}%", textposition="outside",
                textfont=dict(color=clr, size=12),
            ))
        fig_s.add_hline(y=0, line_color=C["border"])
        theme(fig_s)
        fig_s.update_layout(height=240, yaxis_title="Variación estimada (%)",
                            title_text="Variación de precio estimada por empresa")
        st.plotly_chart(fig_s, use_container_width=True)

    # ════════════════════════════════════
    # TAB 3: Envío de órdenes al Broker
    # ════════════════════════════════════
    with tabs[2]:
        st.subheader("📤 Envío de Órdenes · Interactive Brokers API")
        st.warning("⚠ Modo demostración. En producción: usar `ib_insync` para órdenes reales.")

        col_form, col_prev = st.columns([1,1])

        with col_form:
            st.markdown("**Nueva orden**")
            ord_tick   = st.selectbox("Ticker", list(EMPRESAS.keys()))
            ord_accion = st.selectbox("Tipo de orden", ["COMPRA (BUY)", "VENTA (SELL)", "SHORT SELL"])
            ord_tipo   = st.selectbox("Tipo de ejecución", ["MARKET","LIMIT","STOP LIMIT"])
            ord_cant   = st.number_input("Cantidad de acciones", min_value=1, value=100)

            df_ord = get_ohlcv(ord_tick, 2)
            p_ref  = df_ord["Close"].iloc[-1]

            ord_precio = st.number_input("Precio límite (USD)", value=round(p_ref, 4),
                                         disabled=(ord_tipo=="MARKET"), format="%.4f")
            ord_sl     = st.number_input("Stop-Loss (USD)", value=round(p_ref*0.95, 4), format="%.4f")
            ord_tp     = st.number_input("Take-Profit (USD)", value=round(p_ref*1.07, 4), format="%.4f")

            enviar = st.button("📤 Enviar orden al broker", use_container_width=True)
            if enviar:
                st.success(f"✅ Orden simulada enviada: **{ord_accion}** {ord_cant} acciones "
                           f"de **{ord_tick}** a ${ord_precio:.4f} · SL: ${ord_sl:.4f} · TP: ${ord_tp:.4f}")

        with col_prev:
            st.markdown("**Vista previa de la orden**")
            clr_o = C["accent"] if "COMPRA" in ord_accion else C["red"]
            total_est = ord_cant * (p_ref if ord_tipo=="MARKET" else ord_precio)
            pnl_tp = (ord_tp - (p_ref if ord_tipo=="MARKET" else ord_precio)) * ord_cant
            pnl_sl = (ord_sl - (p_ref if ord_tipo=="MARKET" else ord_precio)) * ord_cant

            st.markdown(f"""
            <div style="background:{C['surface']};border:2px solid {clr_o}44;
                        border-radius:12px;padding:18px">
              <div style="font-size:10px;color:{C['muted']};letter-spacing:1.2px;
                          text-transform:uppercase;margin-bottom:6px">Orden · {ord_accion}</div>
              <div style="font-size:24px;font-weight:700;color:{clr_o};margin-bottom:14px">{ord_tick}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
                <div style="background:{C['surf2']};border-radius:7px;padding:10px">
                  <div style="font-size:10px;color:{C['muted']}">Cantidad</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:15px">{ord_cant:,}</div>
                </div>
                <div style="background:{C['surf2']};border-radius:7px;padding:10px">
                  <div style="font-size:10px;color:{C['muted']}">Total estimado</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:15px">${total_est:,.2f}</div>
                </div>
                <div style="background:{C['surf2']};border-radius:7px;padding:10px">
                  <div style="font-size:10px;color:{C['muted']}">Stop-Loss</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                              color:{C['red']}">${ord_sl:.4f} (${pnl_sl:+,.2f})</div>
                </div>
                <div style="background:{C['surf2']};border-radius:7px;padding:10px">
                  <div style="font-size:10px;color:{C['muted']}">Take-Profit</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                              color:{C['accent']}">${ord_tp:.4f} (${pnl_tp:+,.2f})</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════
    # TAB 4: Optimización del portafolio
    # ════════════════════════════════════
    with tabs[3]:
        st.subheader("📊 Optimización del Portafolio del Día")
        st.caption("Pesos sugeridos basados en señales de modelos y métricas de riesgo")

        # Calcular señal y volatilidad para cada empresa
        opt_rows = []
        for tick, info in EMPRESAS.items():
            df_o  = get_ohlcv(tick, 60)
            vol   = df_o["Close"].pct_change().std() * np.sqrt(252)
            preds = get_all_predictions(tick)
            rec   = consenso(preds)
            score = rec["ratio"]   # 0..1, mayor = más alcista
            opt_rows.append({
                "Ticker":       tick,
                "Señal":        f"{rec['emoji']} {rec['señal']}",
                "Score":        round(score, 3),
                "Volatilidad":  round(vol, 4),
                "Peso actual":  f"{PORTAFOLIO_BASE.get(tick,{}).get('acciones',0) * get_ohlcv(tick,2)['Close'].iloc[-1] / 1:.0f}",
                "Peso sugerido %": 0,  # se calcula abajo
            })

        # Peso sugerido: proporcional al score, reducir vol alta
        df_opt   = pd.DataFrame(opt_rows)
        raw_w    = df_opt["Score"] / (df_opt["Volatilidad"] + 0.01)
        df_opt["Peso sugerido %"] = (raw_w / raw_w.sum() * 100).round(1)

        def _cw(v):
            if "COMPRAR" in str(v): return f"color:{C['accent']};font-weight:700"
            if "VENDER"  in str(v): return f"color:{C['red']};font-weight:700"
            if "HOLD"    in str(v): return f"color:{C['gold']}"
            return ""

        st.dataframe(
            df_opt[["Ticker","Señal","Score","Volatilidad","Peso sugerido %"]]
                  .style
                  .applymap(_cw, subset=["Señal"])
                  .format({"Score":"{:.3f}","Volatilidad":"{:.2%}","Peso sugerido %":"{:.1f}%"})
                  .set_properties(**{"background-color":C["surface"],"color":C["text"]}),
            use_container_width=True, hide_index=True,
        )

        # Donut sugerido
        fig_opt = go.Figure(go.Pie(
            labels=df_opt["Ticker"],
            values=df_opt["Peso sugerido %"],
            hole=0.5,
            marker=dict(colors=[EMPRESAS[t]["color"] for t in df_opt["Ticker"]]),
        ))
        theme(fig_opt)
        fig_opt.update_layout(height=300,
                              title_text="Distribución sugerida del portafolio",
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_opt, use_container_width=True)
