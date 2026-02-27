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
<style> */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Fondos blancos ─────────────────────────────────── */
.stApp                          { background-color: #FFFFFF; }
section[data-testid="stSidebar"]{ background-color: #F8F9FA;
                                   border-right: 1px solid #DEE2E6; }

/* ── Tipografía ──────────────────────────────────────── */
html, body, [class*="css"]      { font-family: 'DM Sans', sans-serif;
                                   color: #212529; }

/* ── Ocultar navegación automática ──────────────────── */
[data-testid="stSidebarNav"]    { display: none; }

/* ── Métricas ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #6C757D !important;
    font-size: 12px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #212529;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 2px solid #DEE2E6;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6C757D;
}
.stTabs [aria-selected="true"] {
    background: rgba(13,110,253,0.06) !important;
    color: #0D6EFD !important;
    border-bottom: 2px solid #0D6EFD !important;
}

/* ── Inputs y selectbox ──────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #FFFFFF;
    border: 1px solid #DEE2E6;
    color: #212529;
}

/* ── Botones: fondo blanco con borde delineado ───────── */
.stButton > button {
    background: #FFFFFF;
    color: #0D6EFD;
    font-weight: 600;
    border: 2px solid #0D6EFD;
    border-radius: 7px;
    padding: 8px 20px;
}
.stButton > button:hover {
    background: #0D6EFD;
    color: #FFFFFF;
}

/* ── Botón de acción principal (confirmar/enviar) ────── */
.stButton > button[kind="primary"] {
    background: #FFFFFF;
    color: #198754;
    border: 2px solid #198754;
}
.stButton > button[kind="primary"]:hover {
    background: #198754;
    color: #FFFFFF;
}

/* ── Alertas con colores claros/semitransparentes ─────── */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid;
}
div[data-baseweb="notification"] {
    background: rgba(220,53,69,0.08) !important;  /* rojo claro */
    border-color: #DC3545 !important;
}

/* ── Divisores y bordes ──────────────────────────────── */
hr { border-color: #DEE2E6; }
.stDataFrame { border: 1px solid #DEE2E6; border-radius: 8px; }

/* ── Sidebar texto ───────────────────────────────────── */
section[data-testid="stSidebar"] * { color: #212529; }
section[data-testid="stSidebar"] .stRadio label { color: #212529 !important; } */
            [data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
from utils.data import EMPRESAS, C

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 18px">
      <div style="width:40px;height:40px;
                  background:linear-gradient(135deg,#0D6EFD,#198754);
                  border-radius:9px;display:flex;align-items:center;
                  justify-content:center;font-size:20px">⛏</div>
      <div>
        <div style="font-size:17px;font-weight:700;font-family:Georgia,serif;
                    color:#212529">InvestAI</div>
        <div style="font-size:9px;color:#6C757D;letter-spacing:1.5px;
                    text-transform:uppercase">Mineras · Perú · v2.0</div>
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
    <div style="padding:10px 12px;background:#FFFFFF;
                border:1px solid #DEE2E6;
                border-left:3px solid {emp['color']};
                border-radius:8px;margin-bottom:18px;
                box-shadow:0 1px 3px rgba(0,0,0,0.06)">
        <div style="font-size:13px;font-weight:600;color:{emp['color']}">{ticker}</div>
        <div style="font-size:11px;color:#212529">{emp['nombre']}</div>
        <div style="font-size:10px;color:#6C757D;margin-top:2px">
            {emp['pais']} · Minería {emp['sector']}
        </div>
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
    <div style="font-size:11px;color:#6C757D;line-height:2">
        <b style="color:#212529">Componente 1 · Fuentes de datos</b><br>
        📉 Yahoo Finance API &nbsp;
        <span style="color:#198754;font-weight:700">● Activo</span><br>
        🏦 Interactive Brokers &nbsp;
        <span style="color:#FFC107;font-weight:700">● Standby</span>
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
