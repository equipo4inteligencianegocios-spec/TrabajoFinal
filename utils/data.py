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
    import yfinance as yf
    yf_ticker = "VOLCABC1.LM" if ticker == "VOLCABC1" else ticker
    df = yf.download(yf_ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    
    # ── Aplanar MultiIndex de columnas (yfinance >= 0.2.x) ────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    """
    PROTOTIPO — datos simulados:
    
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
    """

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI, MACD, MA20, MA50, Bollinger sobre DataFrame OHLCV estándar."""
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
# 4. DATASET MULTI-ACTIVO  ← NUEVO BLOQUE, VA AQUÍ
# ══════════════════════════════════════════════════════════════════

# Tickers que conforman el dataset multi-activo del notebook
TICKERS_MULTIACTIVO = [
    "SCCO", "VOLCABC1", "FSM", "BVN", "ABX", "BHP",
    "BZF",
    "BAP", "BVL", "COPX", "GLD", "SLV",
    "HGF", "SIF", "ZINCL",
]

# Solo los 4 que aparecen en las 11 features del scaler
TICKERS_FEATURES = ["HG=F", "ABX", "FSM"]


def build_multiactivo_df(ticker: str, days: int = 120) -> pd.DataFrame:
    tickers_a_descargar = list(set([ticker] + TICKERS_FEATURES))
    dfs = []
    for tick in tickers_a_descargar:
        try:
            df_t = get_ohlcv(tick, days)
            df_t.columns = [f"{col}_{tick}" for col in df_t.columns]
            dfs.append(df_t)
            # ── DEBUG temporal ────────────────────────────────────
            print(f"[DEBUG] {tick}: {len(df_t)} filas | "
                  f"desde {df_t.index[0].date()} hasta {df_t.index[-1].date()}")
        except Exception as e:
            print(f"[AVISO] No se pudo descargar {tick}: {e}")

    if not dfs:
        raise ValueError("No se pudo descargar ningún ticker")

    # ── DEBUG: ver índices antes de concat ────────────────────────
    for d in dfs:
        print(f"[DEBUG] Columnas: {list(d.columns[:3])} | filas: {len(d)}")

    df_multi = pd.concat(dfs, axis=1)
    print(f"[DEBUG] Después de concat: {df_multi.shape} | NaN totales: {df_multi.isna().sum().sum()}")

    df_multi = df_multi.ffill().bfill()
    print(f"[DEBUG] Después de ffill/bfill: NaN restantes: {df_multi.isna().sum().sum()}")

    df_multi = df_multi.dropna()
    print(f"[DEBUG] Después de dropna: {len(df_multi)} filas")

    if len(df_multi) == 0:
        raise ValueError(f"DataFrame multi-activo vacío para {ticker}")

    return df_multi


def add_features_multiactivo(df_multi: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calcula las 11 features en el orden exacto del scaler:
    Spread_OC, Momentum_5, RSI, Momentum_10, Range_Norm,
    Range_HL, High_HG=F, Low_ABX, MACD, Open_FSM, Trend
    """
    df = df_multi.copy()
    t  = ticker

    df[f"Spread_OC_{t}"]   = df[f"Close_{t}"] - df[f"Open_{t}"]
    df[f"Momentum_5_{t}"]  = df[f"Close_{t}"].pct_change(5)

    delta = df[f"Close_{t}"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df[f"RSI_{t}"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    df[f"Momentum_10_{t}"] = df[f"Close_{t}"].pct_change(10)
    df[f"Range_Norm_{t}"]  = (df[f"High_{t}"] - df[f"Low_{t}"]) / df[f"Close_{t}"]
    df[f"Range_HL_{t}"]    = df[f"High_{t}"] - df[f"Low_{t}"]

    ema12 = df[f"Close_{t}"].ewm(span=12).mean()
    ema26 = df[f"Close_{t}"].ewm(span=26).mean()
    df[f"MACD_{t}"] = ema12 - ema26

    df["Trend"] = (df[f"Close_{t}"] > df[f"Close_{t}"].shift(1)).astype(int)

    return df.dropna()


# ══════════════════════════════════════════════════════════════════
# 5. PREDICCIONES 
# ══════════════════════════════════════════════════════════════════
def _seed(ticker: str, model_id: str) -> int:
    return abs(hash(ticker + model_id)) % 99999

def predict_clasificacion(ticker: str, model_id: str) -> dict:
    import joblib
    import tensorflow as tf
    import os

    base        = os.path.join(os.path.dirname(__file__), "..", "models")
    scaler_path = os.path.join(base, f"scaler_{ticker}.pkl")
    model_file_map = {
        "svc":    os.path.join(base, f"svc_{ticker}.pkl"),
        "rnn":    os.path.join(base, f"rnn_{ticker}.h5"),
        "lstm_c": os.path.join(base, f"lstm_c_{ticker}.h5"),
        "bilstm": os.path.join(base, f"bilstm_{ticker}.h5"),
        "gru":    os.path.join(base, f"gru_{ticker}.h5"),
    }
    model_path = model_file_map.get(model_id, "")

    # ── Fallback si no existe el modelo ──────────────────────────
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[AVISO] Modelo no encontrado: {model_path} → simulación")
        np.random.seed(_seed(ticker, model_id))
        conf    = float(np.random.uniform(0.52, 0.97))
        acc_map = {"svc": 0.741, "rnn": 0.658, "lstm_c": 0.712,
                   "bilstm": 0.728, "gru": 0.703}
        df_tmp  = get_ohlcv(ticker, 2)
        return {
            "tipo":       "clasificacion",
            "tendencia":  "SUBIDA" if conf > 0.5 else "BAJADA",
            "confianza":  round(conf * 100, 1),
            "accuracy":   acc_map.get(model_id, 0.70),
            "precio_hoy": round(float(df_tmp["Close"].iloc[-1]), 4),
        }

    # ── Producción ────────────────────────────────────────────────
    df_multi = build_multiactivo_df(ticker, days=120)
    df_multi = add_features_multiactivo(df_multi, ticker)
    p_hoy    = float(df_multi[f"Close_{ticker}"].iloc[-1])

    feature_cols = [
        f"Spread_OC_{ticker}",
        f"Momentum_5_{ticker}",
        f"RSI_{ticker}",
        f"Momentum_10_{ticker}",
        f"Range_Norm_{ticker}",
        f"Range_HL_{ticker}",
        "High_HG=F",
        "Low_ABX",
        f"MACD_{ticker}",
        "Open_FSM",
        "Trend",
    ]

    X        = df_multi[feature_cols].values
    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    acc_map  = {"svc": 0.741, "rnn": 0.658, "lstm_c": 0.712,
                "bilstm": 0.728, "gru": 0.703}

    if model_id == "svc":
        model = joblib.load(model_path)
        pred  = model.predict(X_scaled[-1:])[0]
        proba = model.predict_proba(X_scaled[-1:])[0]
        conf  = float(proba.max())
        tend  = "SUBIDA" if pred == 1 else "BAJADA"
    else:
        lookback = 60
        model    = tf.keras.models.load_model(model_path)
        X_seq    = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
        proba    = float(model.predict(X_seq, verbose=0)[0][0])
        conf     = proba if proba > 0.5 else 1 - proba
        tend     = "SUBIDA" if proba > 0.5 else "BAJADA"

    return {
        "tipo":       "clasificacion",
        "tendencia":  tend,
        "confianza":  round(conf * 100, 1),
        "accuracy":   acc_map.get(model_id, 0.70),
        "precio_hoy": round(p_hoy, 4),
    }

def predict_regresion(ticker: str, model_id: str) -> dict:
    import joblib
    import tensorflow as tf
    from statsmodels.tsa.arima.model import ARIMA

    df    = get_ohlcv(ticker, 120)
    df    = add_indicators(df)
    df    = df.dropna()
    p_hoy = float(df["Close"].iloc[-1])

    mae_map = {"arima": 0.082, "lstm_r": 0.061, "arima_lstm": 0.048}

    if model_id == "arima":
        # ARIMA no requiere scaler, trabaja directamente con la serie
        serie  = df["Close"].values
        model  = ARIMA(serie, order=(5, 1, 0))  # orden según tu notebook
        result = model.fit()
        p_pred = float(result.forecast(steps=1)[0])

    elif model_id == "lstm_r":
        lookback = 60
        feature_cols = ["Close", "RSI", "MACD", "MA20", "MA50"]
        scaler_X = joblib.load(f"models/scaler_X_{ticker}.pkl")
        scaler_y = joblib.load(f"models/scaler_y_{ticker}.pkl")
        model    = tf.keras.models.load_model(f"models/lstm_r_{ticker}.h5")
        X = df[feature_cols].values
        X_scaled = scaler_X.transform(X)
        X_seq    = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
        y_pred_scaled = model.predict(X_seq, verbose=0)
        p_pred   = float(scaler_y.inverse_transform(y_pred_scaled)[0][0])

    elif model_id == "arima_lstm":
        # Ensamble: promedio ponderado ARIMA + LSTM
        pred_arima = ...   # mismo bloque ARIMA de arriba
        pred_lstm  = ...   # mismo bloque LSTM de arriba
        p_pred = 0.4 * pred_arima + 0.6 * pred_lstm   # pesos del ensamble

    var_pct = (p_pred / p_hoy - 1) * 100
    tend    = "SUBIDA" if p_pred > p_hoy else "BAJADA"

    return {
        "tipo":          "regresion",
        "tendencia":     tend,
        "precio_hoy":    round(p_hoy, 4),
        "precio_pred":   round(p_pred, 4),
        "variacion_pct": round(var_pct, 2),
        "confianza":     round(min(abs(var_pct / 2) * 10 + 55, 97), 1),
        "mae":           mae_map.get(model_id, 0.07),
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


def _generate_historical_signals(ticker: str, model_id: str, df: pd.DataFrame) -> pd.Series:
    """
    Genera señales históricas rolling para backtesting.
    En producción: ejecutar el modelo sobre ventanas deslizantes del histórico.
    Por ahora: señal simple basada en cruce de medias (MA20 vs MA50).
    """
    df = add_indicators(df)
    # Señal: SUBIDA cuando MA20 cruza arriba de MA50, BAJADA cuando cruza abajo
    señal = pd.Series(index=df.index, dtype=str)
    señal[:] = "HOLD"
    señal[df["MA20"] > df["MA50"]] = "SUBIDA"
    señal[df["MA20"] < df["MA50"]] = "BAJADA"
    return señal
# ══════════════════════════════════════════════════════════════════
# 6. BACKTESTING
# ══════════════════════════════════════════════════════════════════
def get_backtest_metrics(ticker: str, model_id: str) -> dict:
    import vectorbt as vbt

    df = get_ohlcv(ticker, 365)

    # ── NO intentar asignar freq al índice ────────────────────────
    # Solo convertir a DatetimeIndex limpio
    df.index = pd.to_datetime(df.index)

    price   = df["Close"]
    signals = _generate_historical_signals(ticker, model_id, df)
    entries = signals == "SUBIDA"
    exits   = signals == "BAJADA"

    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=10_000,
        freq="1D",   # ← pasar como string "1D" en lugar de "B"
    )
    return {
        "total_return": float(pf.total_return()),
        "sharpe":       float(pf.sharpe_ratio()),
        "sortino":      float(pf.sortino_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate":     float(pf.trades.win_rate()),
        "total_trades": int(pf.trades.count()),
        "calmar":       float(pf.calmar_ratio()),
    }
