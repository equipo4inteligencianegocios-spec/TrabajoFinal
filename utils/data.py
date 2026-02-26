"""
utils/data.py
─────────────────────────────────────────────────────────────────
Configuración central: empresas, modelos, tema Plotly y capa de datos.

PRODUCCIÓN → descomentar el bloque yfinance y eliminar los datos simulados.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# 1. PALETA Y TEMA PLOTLY
# ══════════════════════════════════════════════════════════════════
C = {
    "bg":      "#080c10",
    "surface": "#0d1420",
    "surf2":   "#111927",
    "border":  "#1e2d42",
    "accent":  "#00d4aa",
    "blue":    "#0087ff",
    "gold":    "#f0b429",
    "red":     "#ff4d6a",
    "purple":  "#9b6dff",
    "text":    "#e8edf5",
    "muted":   "#5a6b80",
    "dim":     "#8899aa",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(family="DM Sans, sans-serif", color=C["text"], size=12),
    xaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"],
               tickfont=dict(color=C["muted"], size=11)),
    yaxis=dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"],
               tickfont=dict(color=C["muted"], size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                font=dict(color=C["dim"], size=11)),
    margin=dict(l=50, r=20, t=44, b=44),
    hovermode="x unified",
)

def theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_BASE)
    return fig


# ══════════════════════════════════════════════════════════════════
# 2. EMPRESAS (según documento especificación)
# ══════════════════════════════════════════════════════════════════
EMPRESAS = {
    "FSM":      {"nombre": "Fortuna Silver Mines",          "pais": "Canadá / Perú",    "sector": "Plata",   "color": C["accent"]},
    "VOLCABC1": {"nombre": "Volcan Compañía Minera S.A.A.", "pais": "Perú",              "sector": "Zinc",    "color": C["blue"]},
    "BVN":      {"nombre": "Cía. de Minas Buenaventura",   "pais": "Perú",              "sector": "Oro",     "color": C["gold"]},
    "ABX":      {"nombre": "Barrick Gold Corporation",      "pais": "Canadá / Perú",    "sector": "Oro",     "color": C["purple"]},
    "BHP":      {"nombre": "BHP Billiton Limited",          "pais": "Australia / Perú", "sector": "Diversif","color": C["red"]},
    "SCCO":     {"nombre": "Southern Copper Corporation",   "pais": "USA / Perú",       "sector": "Cobre",   "color": C["dim"]},
}

# ══════════════════════════════════════════════════════════════════
# 3. MODELOS (según Componentes 2.1 y 2.2 del documento)
# ══════════════════════════════════════════════════════════════════
MODELOS_CLASIFICACION = [
    {"id": "svc",    "label": "SVC",        "desc": "Support Vector Classifier",            "comp": "2.1.1", "tipo": "Machine Learning"},
    {"id": "rnn",    "label": "Simple RNN", "desc": "Red Neuronal Recurrente Simple",        "comp": "2.1.2", "tipo": "Deep Learning"},
    {"id": "lstm_c", "label": "LSTM",       "desc": "LSTM Classifier",                       "comp": "2.1.3", "tipo": "Deep Learning"},
    {"id": "bilstm", "label": "BiLSTM",     "desc": "Bidirectional LSTM Classifier",         "comp": "2.1.4", "tipo": "Deep Learning"},
    {"id": "gru",    "label": "GRU",        "desc": "Gated Recurrent Unit Classifier",       "comp": "2.1.5", "tipo": "Deep Learning"},
]

MODELOS_REGRESION = [
    {"id": "arima",      "label": "ARIMA",      "desc": "Autoregressive Integrated Moving Average", "comp": "2.2.1", "tipo": "Series de Tiempo"},
    {"id": "lstm_r",     "label": "LSTM",        "desc": "LSTM Regressor",                           "comp": "2.2.2", "tipo": "Deep Learning"},
    {"id": "arima_lstm", "label": "ARIMA-LSTM",  "desc": "Ensamble ARIMA + LSTM Regressor",          "comp": "2.2.3", "tipo": "Ensemble"},
]

TODOS_MODELOS = MODELOS_CLASIFICACION + MODELOS_REGRESION


# ══════════════════════════════════════════════════════════════════
# 4. CAPA DE INGESTA DE DATOS — Componente 1
#    PRODUCCIÓN: descomentar bloque yfinance
# ══════════════════════════════════════════════════════════════════
def get_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Descarga datos OHLCV desde Yahoo Finance.

    PRODUCCIÓN — descomentar:
    ─────────────────────────
    import yfinance as yf
    yf_ticker = "VOLCABC1.LM" if ticker == "VOLCABC1" else ticker
    df = yf.download(yf_ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    return df

    PROTOTIPO — datos simulados:
    """
    np.random.seed(abs(hash(ticker)) % 9999)
    precios_base = {"FSM": 4.20, "VOLCABC1": 0.45, "BVN": 9.80,
                    "ABX": 17.50, "BHP": 52.00, "SCCO": 95.00}
    base = precios_base.get(ticker, 20.0)
    ret  = np.random.normal(0.0003, 0.021, days)
    cls  = base * np.cumprod(1 + ret)
    hi   = cls * (1 + np.abs(np.random.normal(0, 0.007, days)))
    lo   = cls * (1 - np.abs(np.random.normal(0, 0.007, days)))
    op   = np.roll(cls, 1); op[0] = base
    vol  = np.random.randint(300_000, 4_000_000, days).astype(float)
    idx  = pd.date_range(end=datetime.today(), periods=days, freq="B")
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo,
                         "Close": cls, "Volume": vol}, index=idx)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI-14, MACD, MA20, MA50 sobre un DataFrame OHLCV."""
    df = df.copy()
    # MA
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    # Bollinger
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    df["BB_upper"] = df["BB_mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["BB_mid"] - 2 * df["Close"].rolling(20).std()
    return df


# ══════════════════════════════════════════════════════════════════
# 5. PREDICCIONES SIMULADAS (reemplazar por modelos reales)
# ══════════════════════════════════════════════════════════════════
def _seed(ticker: str, model_id: str) -> int:
    return abs(hash(ticker + model_id)) % 99999

def predict_clasificacion(ticker: str, model_id: str) -> dict:
    """
    PRODUCCIÓN: cargar modelo entrenado (.pkl / .h5) y ejecutar predict().
    Retorna: tendencia DÍA SIGUIENTE ('SUBIDA' | 'BAJADA') + confianza.
    """
    np.random.seed(_seed(ticker, model_id))
    df     = get_ohlcv(ticker, 3)
    p_hoy  = float(df["Close"].iloc[-1])
    conf   = float(np.random.uniform(0.52, 0.97))
    tend   = "SUBIDA" if conf > 0.5 else "BAJADA"
    # Precisiones por modelo (simuladas según tipo)
    acc_map = {"svc": 0.741, "rnn": 0.658, "lstm_c": 0.712,
               "bilstm": 0.728, "gru": 0.703}
    return {
        "tipo": "clasificacion", "tendencia": tend,
        "confianza": round(conf * 100, 1),
        "accuracy": acc_map.get(model_id, 0.70),
        "precio_hoy": round(p_hoy, 4),
    }

def predict_regresion(ticker: str, model_id: str) -> dict:
    """
    PRODUCCIÓN: cargar modelo entrenado y ejecutar predict() sobre la ventana.
    Retorna: precio predicho DÍA SIGUIENTE + tendencia derivada.
    """
    np.random.seed(_seed(ticker, model_id))
    df      = get_ohlcv(ticker, 3)
    p_hoy   = float(df["Close"].iloc[-1])
    var_pct = float(np.random.normal(0.004, 0.019))
    p_pred  = p_hoy * (1 + var_pct)
    tend    = "SUBIDA" if p_pred > p_hoy else "BAJADA"
    mae_map = {"arima": 0.082, "lstm_r": 0.061, "arima_lstm": 0.048}
    return {
        "tipo": "regresion", "tendencia": tend,
        "precio_hoy": round(p_hoy, 4),
        "precio_pred": round(p_pred, 4),
        "variacion_pct": round(var_pct * 100, 2),
        "confianza": round(min(abs(var_pct / 0.019) * 45 + 55, 97), 1),
        "mae": mae_map.get(model_id, 0.07),
    }

def get_all_predictions(ticker: str) -> dict:
    """Retorna predicciones de los 8 modelos para un ticker."""
    out = {}
    for m in MODELOS_CLASIFICACION:
        out[m["id"]] = {**predict_clasificacion(ticker, m["id"]), "label": m["label"], "comp": m["comp"]}
    for m in MODELOS_REGRESION:
        out[m["id"]] = {**predict_regresion(ticker, m["id"]),     "label": m["label"], "comp": m["comp"]}
    return out

def consenso(preds: dict) -> dict:
    """Calcula la señal de consenso (COMPRAR / VENDER / HOLD) por votación."""
    votos_s = sum(1 for p in preds.values() if p["tendencia"] == "SUBIDA")
    total   = len(preds)
    ratio   = votos_s / total
    if   ratio >= 0.625: senal, color, emoji = "COMPRAR", C["accent"], "↑"
    elif ratio <= 0.375: senal, color, emoji = "VENDER",  C["red"],    "↓"
    else:                senal, color, emoji = "HOLD",    C["gold"],   "→"
    return {"señal": senal, "color": color, "emoji": emoji,
            "votos_subida": votos_s, "total": total, "ratio": ratio}


# ══════════════════════════════════════════════════════════════════
# 6. BACKTESTING SIMULADO (reemplazar por VectorBT real)
# ══════════════════════════════════════════════════════════════════
def get_backtest_metrics(ticker: str, model_id: str) -> dict:
    """
    PRODUCCIÓN:
        import vectorbt as vbt
        entries = signals_series == "SUBIDA"
        exits   = signals_series == "BAJADA"
        pf = vbt.Portfolio.from_signals(price_series, entries, exits)
        return {"total_return": pf.total_return(), "sharpe": pf.sharpe_ratio(), ...}
    """
    np.random.seed(_seed(ticker, model_id))
    return {
        "total_return":  round(float(np.random.uniform(-0.05, 0.55)), 4),
        "sharpe":        round(float(np.random.uniform(0.4,  2.2)),  3),
        "sortino":       round(float(np.random.uniform(0.5,  2.8)),  3),
        "max_drawdown":  round(float(np.random.uniform(-0.25, -0.04)), 4),
        "win_rate":      round(float(np.random.uniform(0.42, 0.78)),  4),
        "total_trades":  int(np.random.randint(18, 72)),
        "calmar":        round(float(np.random.uniform(0.3,  2.0)),  3),
    }
