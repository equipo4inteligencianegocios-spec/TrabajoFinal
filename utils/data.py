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
    "bg":      "#FFFFFF",   # fondo blanco
    "surface": "#FFFFFF",   # superficies blancas
    "surf2":   "#F8F9FA",   # gris muy claro para cards
    "border":  "#DEE2E6",   # borde gris claro
    "accent":  "#198754",   # verde oscuro (subida)
    "blue":    "#0D6EFD",   # azul texto/botones
    "gold":    "#FFC107",   # amarillo advertencia
    "red":     "#DC3545",   # rojo bajada
    "purple":  "#6F42C1",   # púrpura (opcional)
    "text":    "#212529",   # negro texto principal
    "muted":   "#6C757D",   # gris texto secundario
    "dim":     "#ADB5BD",   # gris claro texto terciario
}

PLOTLY_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="DM Sans, sans-serif", color="#212529", size=12),
    xaxis=dict(
        gridcolor="#DEE2E6", zeroline=False, linecolor="#DEE2E6",
        tickfont=dict(color="#6C757D", size=11),
    ),
    yaxis=dict(
        gridcolor="#DEE2E6", zeroline=False, linecolor="#DEE2E6",
        tickfont=dict(color="#6C757D", size=11),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#DEE2E6",
        font=dict(color="#212529", size=11),
    ),
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
    "FSM":      {"nombre": "Fortuna Silver Mines",          "pais": "Canadá / Perú",    "sector": "Plata",    "color": "#198754"},  # verde
    "VOLCABC1": {"nombre": "Volcan Compañía Minera S.A.A.", "pais": "Perú",              "sector": "Zinc",     "color": "#0D6EFD"},  # azul
    "BVN":      {"nombre": "Cía. de Minas Buenaventura",   "pais": "Perú",              "sector": "Oro",      "color": "#B8860B"},  # dorado oscuro
    "ABX":      {"nombre": "Barrick Gold Corporation",      "pais": "Canadá / Perú",    "sector": "Oro",      "color": "#6F42C1"},  # púrpura
    "BHP":      {"nombre": "BHP Billiton Limited",          "pais": "Australia / Perú", "sector": "Diversif", "color": "#DC3545"},  # rojo
    "SCCO":     {"nombre": "Southern Copper Corporation",   "pais": "USA / Perú",       "sector": "Cobre",    "color": "#6C757D"},  # gris
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
    import json
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "models")

    # ── Rutas según estructura de carpetas del ZIP ────────────────
    model_file_map = {
        "svc":    os.path.join(base, "clasificacion", "svc",    ticker, "model.pkl"),
        "rnn":    os.path.join(base, "clasificacion", "rnn",    ticker, "model.h5"),
        "lstm_c": os.path.join(base, "clasificacion", "lstm_c", ticker, "model.h5"),
        "bilstm": os.path.join(base, "clasificacion", "bilstm", ticker, "model.h5"),
        "gru":    os.path.join(base, "clasificacion", "gru",    ticker, "model.h5"),
    }
    meta_file_map = {
        "svc":    os.path.join(base, "clasificacion", "svc",    ticker, "meta.json"),
        "rnn":    os.path.join(base, "clasificacion", "rnn",    ticker, "meta.json"),
        "lstm_c": os.path.join(base, "clasificacion", "lstm_c", ticker, "meta.json"),
        "bilstm": os.path.join(base, "clasificacion", "bilstm", ticker, "meta.json"),
        "gru":    os.path.join(base, "clasificacion", "gru",    ticker, "meta.json"),
    }

    model_path = model_file_map.get(model_id, "")
    meta_path  = meta_file_map.get(model_id, "")

    # ── Fallback si no existe el modelo ──────────────────────────
    if not os.path.exists(model_path):
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

    # ── Leer features exactas del meta.json ──────────────────────
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta["features"]
        accuracy     = meta.get("accuracy_test", 0.70)
    else:
        # Fallback: features estándar si no hay meta.json
        feature_cols = ["Open", "High", "Low", "Close", "Volume",
                        "MA20", "MA50", "RSI", "MACD", "MACD_signal",
                        "MACD_hist", "BB_mid", "BB_upper", "BB_lower"]
        accuracy = 0.70

    # ── Preparar datos con add_indicators() estándar ─────────────
    df    = get_ohlcv(ticker, 120)
    df    = add_indicators(df)
    df    = df.dropna()
    p_hoy = float(df["Close"].iloc[-1])
    X     = df[feature_cols].values

    # ── Producción: SVC Pipeline (scaler ya incluido) ─────────────
    if model_id == "svc":
        model = joblib.load(model_path)
        # Pipeline predice directamente, sin scaler externo
        pred  = model.predict(X[-1:])[0]
        proba = model.predict_proba(X[-1:])[0]
        conf  = float(proba.max())
        tend  = "SUBIDA" if pred == 1 else "BAJADA"

    # ── Deep Learning: RNN / LSTM / BiLSTM / GRU ─────────────────
    else:
        lookback = 60
        model    = tf.keras.models.load_model(model_path)
        # Para DL sí necesitamos escalar manualmente
        from sklearn.preprocessing import MinMaxScaler
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)  # ← reemplazar con scaler guardado cuando estén disponibles
        X_seq    = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
        proba    = float(model.predict(X_seq, verbose=0)[0][0])
        conf     = proba if proba > 0.5 else 1 - proba
        tend     = "SUBIDA" if proba > 0.5 else "BAJADA"

    return {
        "tipo":       "clasificacion",
        "tendencia":  tend,
        "confianza":  round(conf * 100, 1),
        "accuracy":   accuracy,
        "precio_hoy": round(p_hoy, 4),
    }

def predict_regresion(ticker: str, model_id: str) -> dict:
    import joblib
    import json
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "models")

    # ── Rutas según estructura de carpetas ────────────────────────
    model_file_map = {
        "arima":      os.path.join(base, "regresion", "arima",      ticker, "model.pkl"),
        "lstm_r":     os.path.join(base, "regresion", "lstm_r",     ticker, "model.keras"),
        "arima_lstm": os.path.join(base, "regresion", "arima_lstm", ticker, "model.pkl"),
    }
    meta_file_map = {
        "arima":      os.path.join(base, "regresion", "arima",      ticker, "meta.json"),
        "lstm_r":     os.path.join(base, "regresion", "lstm_r",     ticker, "meta.json"),
        "arima_lstm": os.path.join(base, "regresion", "arima_lstm", ticker, "meta.json"),
    }
    scaler_path = os.path.join(base, "regresion", "lstm_r", ticker, "scaler.joblib")

    model_path = model_file_map.get(model_id, "")
    meta_path  = meta_file_map.get(model_id, "")
    mae_map    = {"arima": 0.082, "lstm_r": 0.061, "arima_lstm": 0.048}

    # ── Fallback si no existe el modelo ──────────────────────────
    if not os.path.exists(model_path):
        print(f"[AVISO] Modelo regresión no encontrado: {model_path} → simulación")
        np.random.seed(_seed(ticker, model_id))
        df_tmp  = get_ohlcv(ticker, 2)
        p_hoy   = float(df_tmp["Close"].iloc[-1])
        var_pct = float(np.random.normal(0.004, 0.019))
        p_pred  = p_hoy * (1 + var_pct)
        tend    = "SUBIDA" if p_pred > p_hoy else "BAJADA"
        return {
            "tipo":          "regresion",
            "tendencia":     tend,
            "precio_hoy":    round(p_hoy, 4),
            "precio_pred":   round(p_pred, 4),
            "variacion_pct": round(var_pct * 100, 2),
            "confianza":     round(min(abs(var_pct / 0.019) * 45 + 55, 97), 1),
            "mae":           mae_map.get(model_id, 0.07),
        }

    # ── Leer meta.json para features y métricas ───────────────────
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        mae = meta.get("mae_test", meta.get("mae", mae_map.get(model_id, 0.07)))
    else:
        mae = mae_map.get(model_id, 0.07)

    # ── Datos base ────────────────────────────────────────────────
    df    = get_ohlcv(ticker, 120)
    df    = add_indicators(df)
    df    = df.dropna()
    p_hoy = float(df["Close"].iloc[-1])

    # ── ARIMA (2.2.1) ─────────────────────────────────────────────
    if model_id == "arima":
        model  = joblib.load(model_path)
        p_pred = float(model.forecast(steps=1).iloc[0])

    # ── LSTM Regressor (2.2.2) ────────────────────────────────────
    elif model_id == "lstm_r":
        import tensorflow as tf

        model  = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Preparar secuencia temporal
        feature_cols = ["Open", "High", "Low", "Close", "Volume",
                        "MA20", "MA50", "RSI", "MACD", "MACD_signal",
                        "MACD_hist", "BB_mid", "BB_upper", "BB_lower"]

        # Usar solo las features que existen en el meta.json si está disponible
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "features" in meta:
                feature_cols = meta["features"]

        input_shape = model.input_shape  # (None, lookback, n_features)
        lookback    = input_shape[1]
        X        = df[feature_cols].values
        X_scaled = scaler.transform(X)
        X_seq    = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))

        y_pred   = model.predict(X_seq, verbose=0)

        # Inverse transform: el scaler fue entrenado sobre Close
        # Reconstruir un array del mismo shape que el scaler espera
        dummy        = np.zeros((1, len(feature_cols)))
        close_idx    = feature_cols.index("Close")
        dummy[0, close_idx] = y_pred[0][0]
        p_pred = float(scaler.inverse_transform(dummy)[0, close_idx])

    # ── ARIMA-LSTM Ensamble (2.2.3) ───────────────────────────────
    elif model_id == "arima_lstm":
        # El model.pkl del ensamble puede ser:
        # a) Un objeto con forecast() → ARIMA fitted
        # b) Un dict con pesos y referencias
        # c) Un objeto custom
        # Intentamos como ARIMA primero, luego como dict
        try:
            model  = joblib.load(model_path)

            # Caso a: tiene método forecast → es ARIMA o similar
            if hasattr(model, "forecast"):
                p_pred = float(model.forecast(steps=1).iloc[0])

            # Caso b: es un dict con pesos del ensamble
            elif isinstance(model, dict):
                peso_arima = model.get("peso_arima", 0.4)
                peso_lstm  = model.get("peso_lstm",  0.6)
                # Cargar ARIMA
                arima_path = os.path.join(base, "regresion", "arima", ticker, "model.pkl")
                arima_m    = joblib.load(arima_path)
                pred_arima = float(arima_m.forecast(steps=1).iloc[0])
                # Cargar LSTM
                import tensorflow as tf
                lstm_path  = os.path.join(base, "regresion", "lstm_r", ticker, "model.keras")
                lstm_m     = tf.keras.models.load_model(lstm_path)
                scaler     = joblib.load(scaler_path)
                feature_cols = ["Open", "High", "Low", "Close", "Volume",
                                "MA20", "MA50", "RSI", "MACD", "MACD_signal",
                                "MACD_hist", "BB_mid", "BB_upper", "BB_lower"]
                lookback   = 60
                X          = df[feature_cols].values
                X_scaled   = scaler.transform(X)
                X_seq      = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
                y_pred     = lstm_m.predict(X_seq, verbose=0)
                dummy      = np.zeros((1, len(feature_cols)))
                close_idx  = feature_cols.index("Close")
                dummy[0, close_idx] = y_pred[0][0]
                pred_lstm  = float(scaler.inverse_transform(dummy)[0, close_idx])
                p_pred     = peso_arima * pred_arima + peso_lstm * pred_lstm

            # Caso c: tipo desconocido → fallback simulado
            else:
                print(f"[AVISO] arima_lstm tipo desconocido: {type(model)} → simulación")
                np.random.seed(_seed(ticker, model_id))
                var_pct = float(np.random.normal(0.004, 0.019))
                p_pred  = p_hoy * (1 + var_pct)

        except Exception as e:
            print(f"[AVISO] Error cargando arima_lstm: {e} → simulación")
            np.random.seed(_seed(ticker, model_id))
            var_pct = float(np.random.normal(0.004, 0.019))
            p_pred  = p_hoy * (1 + var_pct)

    var_pct = (p_pred / p_hoy - 1) * 100
    tend    = "SUBIDA" if p_pred > p_hoy else "BAJADA"

    return {
        "tipo":          "regresion",
        "tendencia":     tend,
        "precio_hoy":    round(p_hoy, 4),
        "precio_pred":   round(p_pred, 4),
        "variacion_pct": round(var_pct, 2),
        "confianza":     round(min(abs(var_pct / 2) * 10 + 55, 97), 1),
        "mae":           round(mae, 4),
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

def _enviar_orden_ib(ticker: str, accion: str, cantidad: int,
                     tipo: str = "MARKET", precio_lim: float = None) -> dict:
    """
    Envía una orden real a Interactive Brokers via ib_insync.
    Requiere TWS o IB Gateway corriendo en localhost:7497.

    Parámetros:
        ticker     : símbolo (ej: "FSM", "SCCO")
        accion     : "BUY" o "SELL"
        cantidad   : número de acciones
        tipo       : "MARKET" | "LIMIT" | "STOP LIMIT"
        precio_lim : precio límite (solo para LIMIT y STOP LIMIT)
    """
    try:
        from ib_insync import IB, Stock, MarketOrder, LimitOrder, Order

        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=10, timeout=10)

        if not ib.isConnected():
            return {"ok": False, "error": "No se pudo conectar a TWS"}

        # ── Definir el contrato ───────────────────────────────────
        # VOLCABC1 cotiza en Lima, el resto en NYSE/SMART
        if ticker == "VOLCABC1":
            contrato = Stock("VOLCABC1", "LSEX", "PEN")
        else:
            contrato = Stock(ticker, "SMART", "USD")

        # Calificar el contrato (obtener detalles completos de IB)
        ib.qualifyContracts(contrato)

        # ── Definir la orden ──────────────────────────────────────
        if tipo == "MARKET":
            orden = MarketOrder(accion, cantidad)
        elif tipo == "LIMIT":
            orden = LimitOrder(accion, cantidad, precio_lim)
        elif tipo == "STOP LIMIT":
            orden = Order()
            orden.action       = accion
            orden.totalQuantity = cantidad
            orden.orderType    = "STP LMT"
            orden.lmtPrice     = precio_lim
            orden.auxPrice     = precio_lim  # stop price

        # ── Enviar la orden ───────────────────────────────────────
        trade = ib.placeOrder(contrato, orden)
        ib.sleep(1)  # esperar confirmación

        precio_ejecutado = trade.orderStatus.avgFillPrice or precio_lim or 0.0

        ib.disconnect()

        return {
            "ok":                True,
            "order_id":          trade.order.orderId,
            "status":            trade.orderStatus.status,
            "precio_ejecutado":  precio_ejecutado,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}