"""
app.py — InvestAI v2
Sistema Predictivo de Acciones Mineras con Operaciones en Perú
Componentes: Yahoo Finance API → Modelos ML/DL → Backtesting VectorBT → Recomendaciones
"""
import streamlit as st

st.set_page_config(
    page_title="InvestAI · Mineras Perú",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background-color: #080c10; }
section[data-testid="stSidebar"] { background-color: #0d1420; border-right: 1px solid #1e2d42; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #e8edf5; }
[data-testid="metric-container"] {
    background:#0d1420; border:1px solid #1e2d42; border-radius:10px; padding:14px 18px;
}
[data-testid="metric-container"] label { color:#5a6b80 !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#00d4aa; font-family:'JetBrains Mono',monospace;
}
.stTabs [data-baseweb="tab-list"] { background:#0d1420; border-bottom:1px solid #1e2d42; }
.stTabs [data-baseweb="tab"]      { background:transparent; color:#8899aa; }
.stTabs [aria-selected="true"]    { background:rgba(0,212,170,0.1)!important; color:#00d4aa!important; }
.stSelectbox>div>div, .stMultiSelect>div>div { background:#111927; border:1px solid #1e2d42; }
.stButton>button { background:linear-gradient(135deg,#00d4aa,#00b891);
    color:#041a14; font-weight:700; border:none; border-radius:7px; }
.stButton>button:hover { opacity:.88; }
hr { border-color:#1e2d42; }
.stDataFrame { border:1px solid #1e2d42; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
from utils.data import EMPRESAS, C

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 18px">
      <div style="width:40px;height:40px;background:linear-gradient(135deg,#00d4aa,#0087ff);
                  border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:20px">⛏</div>
      <div>
        <div style="font-size:17px;font-weight:700;font-family:Georgia,serif">InvestAI</div>
        <div style="font-size:9px;color:#5a6b80;letter-spacing:1.5px;text-transform:uppercase">
            Mineras · Perú · v2.0
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Selector de empresa activa (global) ───────────────────────
    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.4px;'
                'text-transform:uppercase;color:#5a6b80;margin-bottom:6px">'
                'Empresa activa</p>', unsafe_allow_html=True)

    ticker = st.selectbox(
        "empresa", list(EMPRESAS.keys()),
        format_func=lambda t: f"{t}  ·  {EMPRESAS[t]['nombre'][:24]}",
        label_visibility="collapsed",
    )
    emp = EMPRESAS[ticker]
    st.markdown(f"""
    <div style="padding:10px 12px;background:#111927;border:1px solid #1e2d42;
                border-left:3px solid {emp['color']};border-radius:8px;margin-bottom:18px">
        <div style="font-size:13px;font-weight:600;color:{emp['color']}">{ticker}</div>
        <div style="font-size:11px;color:#8899aa">{emp['nombre']}</div>
        <div style="font-size:10px;color:#5a6b80;margin-top:2px">{emp['pais']} · Minería {emp['sector']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.4px;'
                'text-transform:uppercase;color:#5a6b80;margin-bottom:6px">'
                'Módulos del sistema</p>', unsafe_allow_html=True)

    pagina = st.radio("nav", [
        "📊  Dashboard General",
        "📈  Datos del Mercado",
        "🤖  Clasificación · Tendencia Día Siguiente",
        "🧬  Regresión · Pronóstico de Precios",
        "⚖   Backtesting · VectorBT",
        "💼  Portafolio y Recomendaciones",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#5a6b80;line-height:2">
        <b style="color:#8899aa">Componente 1 · Fuentes de datos</b><br>
        📉 Yahoo Finance API &nbsp;<span style="color:#00d4aa">●</span><br>
        🏦 Interactive Brokers &nbsp;<span style="color:#f0b429">●</span>
    </div>
    """, unsafe_allow_html=True)

# ── Session state: ticker global ─────────────────────────────────
st.session_state["ticker"] = ticker

# ── Enrutador ─────────────────────────────────────────────────────
match pagina:
    case "📊  Dashboard General":
        from pages import dashboard;     dashboard.show()
    case "📈  Datos del Mercado":
        from pages import mercado;       mercado.show()
    case "🤖  Clasificación · Tendencia Día Siguiente":
        from pages import clasificacion; clasificacion.show()
    case "🧬  Regresión · Pronóstico de Precios":
        from pages import regresion;     regresion.show()
    case "⚖   Backtesting · VectorBT":
        from pages import backtesting;   backtesting.show()
    case "💼  Portafolio y Recomendaciones":
        from pages import portafolio;    portafolio.show()
