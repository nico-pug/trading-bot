# Trading Bot Avanzato per Crypto Futures

📌 **Overview**  
Questo progetto è un trading bot avanzato per crypto futures, progettato per operare in modo completamente autonomo su Binance USDT‑M Futures.  
Integra oltre 50 concetti professionali di trading, un motore di alpha multi‑fattore, un sistema ML LightGBM, un sentiment analyzer multi‑fonte, un backtest engine completo, e un launcher multi‑asset.

Il bot è stato progettato per essere:

- **Robusto** → gestione rischio avanzata, drawdown control, Kelly, EV
- **Intelligente** → ML filter, sentiment analysis, scoring multi‑fattore
- **Scalabile** → multi‑asset, multi‑processo, caching, modularità
- **Analitico** → backtest dettagliati, metriche di performance, trade journal
- **Estendibile** → architettura modulare, configurazioni esterne, roadmap chiara

---

🧠 **Architettura del Sistema (Refactor v2.0)**  
L’architettura è ora modulare e organizzata in pacchetti:

```

├── bot\
|
├── core\
| ├── manager.py (Gestore bot)
| ├── execution.py (Esecutore trade)
| ├── data_feed.py (Dati & Indicatori)
| ├── alpha_engine.py (Calcolo segnali)
| ├── risk_manager.py (Calcolo size & controlli)
| ├── performance.py (Metrica & Journaling)
| └── trader.py (Orchestratore centrale)
|
├── ml\
| ├── trainer.py (Training pipeline)
| ├── filter.py (Filtro segnali & Drift detecion)
| └── features.py (Feature Engineering)
|
├── sentiment\
| └── analyzer.py (Orchestratore scraper)
| └── scarapers\
|   ├── base.py (Rate Limiting)
|   └── news.py (News scraper)
|
├── backtest\
| └── simulation.py (Motore simulazione)
|
├── utils\
| ├── config.py (Config & Logging)
| └── logger.py (Gestione log)
|
└── data\
| ├── journals\ (Storico trade live)
| └── logs\ (File di log)
|
├── config.yaml (Configurazione)
└── main.py (Entry point CLI)
```

---

⚙️ **1. Alpha Engine — Motore di Segnali Multi‑Fattore**  
Il cuore del bot è l’AlphaEngine, che combina oltre 50 segnali professionali provenienti da:

- **📊 Microstruttura & Order Flow**

  - Order Blocks
  - Fair Value Gaps
  - Liquidity Grabs
  - Market Structure Shift
  - CVD Divergence
  - DOM Imbalance
  - Breaker Blocks
  - Sweep Liquidity

- **📈 Indicatori Tecnici Avanzati**

  - VWAP / TWAP
  - Bollinger Bands
  - ATR Volatility
  - RSI Extremes
  - Fibonacci Golden Pocket

- **⛓️ On‑Chain Analytics**

  - SOPR
  - NUPL
  - MVRV
  - Exchange Flows
  - Whale Activity

- **📉 Derivati & Futures**
  - Funding Rate
  - Open Interest
  - Basis (Spot‑Futures Premium)
  - Long/Short Ratio

Ogni segnale contribuisce a uno score finale, usato per decidere LONG/SHORT.

---

🤖 **2. Machine Learning — LightGBM Trade Filter**  
Il bot include un modello ML LightGBM che filtra i trade con bassa probabilità di successo.

- **🔍 Pipeline ML**
  - Generazione dataset tramite backtest (simulazione outcome trade)
  - Feature engineering avanzato:
    - RSI normalizzato
    - ATR%
    - BB position
    - VWAP distance
    - CVD trend
    - Volume ratio
    - Momentum
    - Score AlphaEngine
    - Side
  - Training LightGBM con early stopping
  - Feature importance
  - Salvataggio modello + metadata
  - MLFilter integrato nel bot live

**🎯 Obiettivo**  
Ridurre i trade a bassa qualità → aumentare win rate e stabilità.

---

📰 **3. Sentiment Analyzer — Reddit + News + Fear & Greed**  
Il bot integra un sistema di sentiment completo:

- **🔗 Fonti**

  - Reddit (API + fallback no‑API)
  - NewsAPI
  - RSS crypto (CoinTelegraph, CryptoNews, Decrypt)
  - Fear & Greed Index

- **🧮 Aggregazione**
  - Pesi configurabili (reddit/news/fng)
  - Normalizzazione 0‑1
  - Caching intelligente
  - Trading signal sentiment (+1, 0, -1)

**🎯 Uso nel bot**

- Filtrare trade contrari al sentiment
- Aggiungere bonus/malus allo score
- Bloccare trade in condizioni estreme

---

📈 **4. Backtest Engine — Candela‑per‑Candela**  
Il backtest engine include:

- Download storico multi‑chunk
- Indicatori tecnici identici al live
- Simulazione SL/TP
- Position sizing dinamico
- Equity curve
- Max drawdown
- Sharpe ratio
- Report dettagliato
- Trade log completo
- Ottimizzazione parametri (grid search)

**📊 Esempio risultati ottimizzazione (SOL)**

| Codice | Rank | SL  | TP  | Thresh  | P&L  | Win% |
| ------ | ---- | --- | --- | ------- | ---- | ---- |
| 1      | 1.5  | 4.0 | 3.0 | 3121.51 | 35.4 | 8.1  |
| 2      | 2.0  | 4.0 | 3.0 | 1274.12 | 37.9 | 8.7  |
| 3      | 1.5  | 3.0 | 3.0 | 1154.73 | 37.3 | 8.7  |

---

🧮 **5. Risk Manager — Gestione Rischio Professionale**  
Il bot implementa:

- Expected Value
- Kelly Criterion (Half‑Kelly)
- Max Drawdown tracking
- Risk of Ruin
- Sharpe Ratio
- Sortino Ratio
- Position sizing dinamico
- Leverage cap
- Trade Journal CSV

---

🔥 **6. Multi‑Asset Launcher**  
Permette di lanciare più bot in parallelo:

```python
python multi_bot.py BTC ETH SOL
```

Ogni bot:

- Gira in un processo separato
- Ha il suo trade journal
- Opera in autonomia

---

🧭 **Roadmap — Prossimi Miglioramenti**  
Dal file `prossimi_miglioramenti.txt`:

- **🟢 Completati**

  - Backtesting
  - Ottimizzazione parametri
  - Multi‑asset
  - Machine Learning

- **🟡 Da implementare**
  - Alert Telegram/Discord
  - Dashboard Web (Flask/Streamlit)
  - Retrain automatico settimanale
  - Feature ML aggiuntive (funding, OI, Twitter sentiment)
  - Portfolio allocation multi‑asset
  - Report generator PDF/Excel

---

📦 **Installazione**

```bash
pip install -r requirements.txt
```

▶️ **Esecuzione (CLI Unificata)**

Il bot ora utilizza un unico punto di ingresso: `bot.main`

- **Bot Live (Singolo)**

  ```python
  python -m bot.main trade --symbol BTC/USDT
  ```

- **Multi-Bot (Tutti i simboli in config)**

  ```python
  python -m bot.main multi
  ```

- **Backtest**

  ```python
  python -m bot.main backtest --symbol BTC/USDT --days 180
  ```

- **Ottimizzazione Parametri**

  ```python
  python -m bot.main optimize --symbol SOL/USDT
  ```

- **Configurazione**
  Tutti i parametri sono gestiti in `config.yaml`. Non serve più modificare il codice!

---

🧪 **Disclaimer**
Questo progetto è a scopo divulgativo e sperimentale.
Non costituisce consulenza finanziaria.

---

🎯 **Conclusione**
Questo bot rappresenta un sistema di trading AI completo, con:

- Architettura modulare
- Alpha engine multi‑fattore
- ML filter
- Sentiment analyzer
- Risk management professionale
- Backtest engine avanzato
- Multi‑asset launcher

È un progetto da portfolio di altissimo livello, perfetto per:

- Università
- Candidature
- Colloqui tecnici
- Dimostrare competenze reali
