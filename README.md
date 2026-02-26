# InvestAI v2 · Sistema Predictivo de Acciones Mineras con Operaciones en Perú
### Trabajo Final Semana 7 · Nueva Especificación Reducida

---

## Arquitectura del sistema (4 Componentes)

```
Componente 1          Componente 2              Componente 3       Componente 4
─────────────         ─────────────────         ────────────       ────────────
Yahoo Finance   →     2.1 Clasificación   →     VectorBT      →   Dashboard
API (yfinance)        SVC, RNN, LSTM,           Backtesting        Gráficos Plotly
                      BiLSTM, GRU                                  Recomendaciones
                      2.2 Regresión                                IB Broker API
                      ARIMA, LSTM,
                      ARIMA-LSTM
```

## Estructura de archivos

```
investai_v2/
├── app.py                      ← Entrada + sidebar + enrutador
├── requirements.txt
├── README.md
│
├── pages/
│   ├── dashboard.py            ← Dashboard General: señales de las 6 empresas
│   ├── mercado.py              ← Componente 1: datos Yahoo Finance + indicadores
│   ├── clasificacion.py        ← Componente 2.1: SVC, RNN, LSTM, BiLSTM, GRU
│   ├── regresion.py            ← Componente 2.2: ARIMA, LSTM, ARIMA-LSTM
│   ├── backtesting.py          ← Componente 3: VectorBT · equity curves · métricas
│   └── portafolio.py           ← Componente 4: portafolio + recomendaciones + broker
│
└── utils/
    └── data.py                 ← Empresas, modelos, tema Plotly, capa de datos
```

## Empresas objetivo

| Ticker    | Empresa                         | País              | Sector    |
|-----------|---------------------------------|-------------------|-----------|
| FSM       | Fortuna Silver Mines            | Canadá / Perú     | Plata     |
| VOLCABC1  | Volcan Compañía Minera S.A.A.   | Perú              | Zinc      |
| BVN       | Cía. de Minas Buenaventura      | Perú              | Oro       |
| ABX       | Barrick Gold Corporation        | Canadá / Perú     | Oro       |
| BHP       | BHP Billiton Limited            | Australia / Perú  | Diversif. |
| SCCO      | Southern Copper Corporation     | USA / Perú        | Cobre     |

## Modelos implementados

### Componente 2.1 — Clasificación (tendencia día siguiente)
| Comp. | Modelo     | Tipo             | Salida              |
|-------|------------|------------------|---------------------|
| 2.1.1 | SVC        | Machine Learning | SUBIDA / BAJADA     |
| 2.1.2 | Simple RNN | Deep Learning    | SUBIDA / BAJADA     |
| 2.1.3 | LSTM       | Deep Learning    | SUBIDA / BAJADA     |
| 2.1.4 | BiLSTM     | Deep Learning    | SUBIDA / BAJADA     |
| 2.1.5 | GRU        | Deep Learning    | SUBIDA / BAJADA     |

### Componente 2.2 — Regresión (precio día siguiente)
| Comp. | Modelo     | Tipo             | Salida                          |
|-------|------------|------------------|---------------------------------|
| 2.2.1 | ARIMA      | Series de tiempo | Precio predicho → tendencia     |
| 2.2.2 | LSTM       | Deep Learning    | Precio predicho → tendencia     |
| 2.2.3 | ARIMA-LSTM | Ensemble         | Precio predicho → tendencia     |

## Instalación y ejecución

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Instalar dependencias base (prototipo)
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

## Migración a producción

En `utils/data.py` hay 3 funciones a reemplazar:

### 1. `get_ohlcv()` → Yahoo Finance real
```python
import yfinance as yf
def get_ohlcv(ticker, days=365):
    yf_ticker = "VOLCABC1.LM" if ticker == "VOLCABC1" else ticker
    df = yf.download(yf_ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    return df[["Open","High","Low","Close","Volume"]].dropna()
```

### 2. `predict_clasificacion()` / `predict_regresion()` → modelos reales
```python
import joblib, tensorflow as tf

svc_model  = joblib.load("models/svc_FSM.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_FSM.h5")

def predict_clasificacion(ticker, model_id):
    features = build_features(get_ohlcv(ticker))  # RSI, MACD, etc.
    if model_id == "svc":
        pred = svc_model.predict(features[-1:])
        prob = svc_model.predict_proba(features[-1:])[0].max()
    return {"tendencia": "SUBIDA" if pred[0]==1 else "BAJADA", "confianza": prob*100}
```

### 3. `get_backtest_metrics()` → VectorBT real
```python
import vectorbt as vbt
def get_backtest_metrics(ticker, model_id):
    price    = get_ohlcv(ticker)["Close"]
    signals  = generate_signals(ticker, model_id)   # señales históricas
    entries  = signals == "SUBIDA"
    exits    = signals == "BAJADA"
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10_000)
    return {
        "total_return": pf.total_return(),
        "sharpe":       pf.sharpe_ratio(),
        "max_drawdown": pf.max_drawdown(),
        "win_rate":     pf.trades.win_rate(),
        "total_trades": pf.trades.count(),
    }
```

### 4. Interactive Brokers (`pages/portafolio.py`)
```python
from ib_insync import IB, Stock, MarketOrder
ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)
contrato = Stock(ticker, "SMART", "USD")
orden    = MarketOrder("BUY", cantidad)
trade    = ib.placeOrder(contrato, orden)
```
