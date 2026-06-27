# Terminal de Trading Cuantitativa

Sistema de trading cuantitativo institucional que integra múltiples fuentes de
datos en tiempo real (mercado, macro, sentimiento, flujo de opciones,
microestructura) y genera señales por **confluencia de factores**.

> ⚠️ **Aviso**: Artefacto de investigación para estudio de estrategias
> sistemáticas. **No es asesoramiento financiero.** Nunca uses capital real sin
> backtesting exhaustivo y paper trading previo.

## Arquitectura

```
Market Data Feeds → Ingestion → (Kafka/Redis/TimescaleDB) → Processing
        → Signal Aggregation → Risk/Decision → Alerts/Dashboard
```

| Capa | Módulo | Contenido |
|------|--------|-----------|
| Ingestion | `quant_terminal.ingestion` | Kafka producer, scraper Forex Factory, Twitter/X, noticias RSS, market data (Polygon/Alpaca), orquestador |
| Processing | `quant_terminal.processing` | Indicadores técnicos, modelos estocásticos (OU, GARCH), ML (LSTM+XGBoost, transformers), flujo de opciones, VIX term structure, cross-asset, multi-timeframe, Fed watch, carry trade, order flow, estacionalidad, risk parity, NLP/sentimiento, eventos económicos, reconocimiento de patrones, generación de señales |
| Aggregation | `quant_terminal.aggregation` | Ensemble ponderado dinámico + combinación bayesiana |
| Analysis | `quant_terminal.analysis` | **Motor de ranking en tiempo real**: score 0-100 por categoría, acción STRONG_BUY..STRONG_SELL, razones/catalizadores/riesgos, top oportunidades/riesgos, resumen de mercado |
| Decision | `quant_terminal.decision` | Risk manager (VaR, position sizing), trade executor, sistema integrado |
| Alerts | `quant_terminal.alerts` | Alertas multi-canal (email, SMS, Slack, Discord, Telegram) + **alertas Telegram con formato rico** + dashboard Dash |
| Dashboard | `quant_terminal.dashboard` | Dashboard web dark-mode: overview de mercado, tablas top opportunities/risks, detalle por activo (radar), ranking por clase |
| Orquestador | `main.py` | Integra todo y lo corre en paralelo (ranking + alertas + dashboard + scheduler diario + health) con apagado ordenado |
| Jarvis | `quant_terminal.jarvis` + `jarvis_main.py` | Asistente de trading conversacional: **advisor diario** (briefing con top compras/evitar, riesgos, acciones), **asistente** por texto/voz con detección de intención, e integración con **Claude (Opus 4.8)** para lenguaje natural — con fallback offline por plantilla |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # núcleo (numpy, pandas, scipy, ta)
pip install -e .                          # hace importable el paquete quant_terminal
pip install -r requirements-optional.txt # pesados: tensorflow, transformers, kafka, etc.
```

> Si prefieres no instalar el paquete, ejecuta los scripts con `PYTHONPATH=.`
> (p. ej. `PYTHONPATH=. python examples/run_demo.py`).

El paquete está diseñado para que el **núcleo computacional** (indicadores,
modelos estocásticos, agregación de señales, risk management) funcione solo con
`numpy`/`pandas`/`scipy`/`ta`. Las dependencias pesadas y de red (Kafka,
TensorFlow, transformers, tweepy, dash) se importan de forma perezosa y
opcional: si no están instaladas, esos módulos lanzan un error claro solo al
usarlos.

## Uso rápido

```python
import numpy as np, pandas as pd
from quant_terminal.processing.technical_indicators import TechnicalIndicatorEngine
from quant_terminal.processing.signal_generator import SignalGenerator
from quant_terminal.aggregation.signal_aggregator import SignalAggregator

# 1. Indicadores técnicos sobre OHLCV
engine = TechnicalIndicatorEngine()
df = engine.calculate_all_indicators(ohlcv_df)

# 2. Señal técnica por confluencia
sig = SignalGenerator(engine).generate_signals(df)

# 3. Agregación con otras fuentes
agg = SignalAggregator().aggregate_signals({
    "technical_analysis": {"signal": sig["final_signal"], "confidence": sig["confidence"], "strength": abs(sig["confluence_score"])},
    "sentiment_analysis": {"signal": "BUY", "confidence": 0.6, "strength": 0.6},
})
print(agg["aggregate_signal"], agg["aggregate_score"])
```

Ver `examples/run_demo.py` (decisión por símbolo) y `examples/run_ranking_demo.py`
(ranking de un universo + preview de alerta Telegram) para demostraciones
end-to-end con datos sintéticos.

### Operación en vivo

```bash
cp quant_terminal/config/config.example.yaml config/config.yaml  # y rellena claves
python main.py config/config.yaml
```

`main.py` lanza en paralelo el ranking engine (análisis completo cada 5 min,
top-10 cada 30 s), el dispatcher de alertas Telegram, el dashboard web
(`http://localhost:8050`), el resumen diario programado y el health monitor.

### Jarvis (asistente conversacional)

```bash
cp quant_terminal/config/jarvis_config.example.yaml config/jarvis_config.yaml
export ANTHROPIC_API_KEY=...                 # opcional: activa respuestas con Claude
python jarvis_main.py --once                 # genera un briefing y termina
python jarvis_main.py config/jarvis_config.yaml   # modo interactivo (texto / 'voz')
```

Jarvis funciona **sin LLM ni red** usando un resumen de plantilla y handlers de
intención (ver `examples/run_jarvis_demo.py`); con `ANTHROPIC_API_KEY` y el SDK
`anthropic`, el resumen diario y las preguntas libres pasan a usar Claude
Opus 4.8. La voz (reconocimiento + TTS) requiere `SpeechRecognition`, `gTTS` y
`pygame`.

## Módulos cuantitativos avanzados

Tres familias de 15 clases cada una (45 en total), con núcleos numéricos
funcionales (numpy/scipy/pandas) y dependencias pesadas (TF/torch/SB3/networkx/
hmmlearn) importadas de forma perezosa con fallbacks testeables.

**Módulo 1 — Análisis cuantitativo (`quant_terminal/processing/`)**: superficie
de volatilidad, order book imbalance, microestructura (VPIN/Avellaneda-Stoikov),
pairs trading (cointegración), stat-arb (factores), HF alpha, cross-asset DCC,
detección de régimen (HMM), anomalías, redes (contagio/riesgo sistémico),
burbujas (LPPL), predicción de crashes, carry optimizer, momentum crash,
liquidity shock.

**Módulo 2 — Machine Learning (`quant_terminal/ml/`)**: transformers de series,
RL trader (entorno gym-like), ensemble meta-learning, GAN, redes bayesianas,
graph NN, few-shot/MAML, análisis de atención, neural ODE, normalizing flows,
contrastive learning, meta-optimizer, self-supervised, multi-task, continual
learning (EWC).

**Módulo 3 — Microestructura (`quant_terminal/microstructure/`)**: order flow
toxicity, market impact (Almgren-Chriss), liquidity provider, order book
dynamics, trade classification (Lee-Ready), price discovery, optimal execution
(TWAP/VWAP/IS/POV), smart order router, latency arbitrage, flash crash,
liquidity cycles, order flow predictor, market resilience, toxicity filter,
market quality.

Cobertura: 108 tests sobre los núcleos numéricos (sin requerir las libs pesadas).

## Configuración

Copia `quant_terminal/config/config.example.yaml` a `config.yaml` y rellena las
claves de API. Usa variables de entorno para los secretos en producción.

## Tests

```bash
pytest -q
```

Los tests cubren el núcleo que no requiere red ni modelos pesados (indicadores,
modelos estocásticos, agregación, risk parity, risk manager).
