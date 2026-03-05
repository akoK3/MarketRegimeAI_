Good starting point. I'll write a proper full README based on this and everything we've built. Here it is:
markdown# MarketRegimeAI

A machine learning pipeline for identifying and classifying financial market regimes — Bull, Bear, and Sideways — using technical indicators derived from historical price data.

Built on 25 years of SPY data (2000–2025), the pipeline covers the full workflow from data ingestion and regime labeling to feature engineering, model evaluation, and backtesting.

---

## What It Does

Most ML finance projects predict price. This project predicts **market conditions** — the broader environment that drives price behavior. Knowing whether the market is trending, declining, or directionless is more actionable for portfolio decisions than a price target.

---

## Pipeline Overview
```
Data → Regime Labeling → Feature Engineering → Model Training → Evaluation → Backtest
```

1. **Data** — SPY daily OHLCV from Yahoo Finance (2000–2025), adjusted prices
2. **Regime Labeling** — ATR-normalized dynamic thresholds classify each day as Bull, Bear, or Sideways based on forward returns
3. **Feature Engineering** — 25+ technical indicators across trend, momentum, volatility, and volume groups
4. **Model Training** — Logistic Regression, Random Forest, XGBoost, MLP compared using walk-forward validation
5. **Evaluation** — Macro F1, Log Loss, Accuracy, Confusion Matrix per model
6. **Backtest** — Regime-based strategy vs Buy-and-Hold benchmark

---

## Regime Labeling Methodology

Regimes are defined using ATR-normalized forward returns:
```
Bull:     5-day forward return > K × ATR(14) / Close
Bear:     5-day forward return < -K × ATR(14) / Close  
Sideways: everything in between
```

ATR normalization makes thresholds adaptive to current volatility rather than using fixed return cutoffs — a key methodological contribution over standard fixed-threshold approaches.

---

## Features

| Group | Features |
|-------|----------|
| Trend | SMA20/50/200 distances, SMA crossover |
| Momentum | RSI(14), MACD(12/26/9), ROC 5/10/20d |
| Volatility | ATR(14), Rolling std 10/20d |
| Volume | Volume ratio, Volume change |
| Lags | RSI, MACD, ATR, SMA distance lags |

All features are derived solely from OHLCV data. ATR and RSI use Wilder's original EWM specification (com=period-1). MACD uses standard EMA (span).

---

## Models

| Model | Class Imbalance Handling |
|-------|--------------------------|
| Logistic Regression | class_weight='balanced' |
| Random Forest | class_weight='balanced' |
| XGBoost | sample_weight (balanced) |
| MLP | early_stopping=True |

All models evaluated using 5-fold walk-forward validation to preserve temporal ordering and prevent data leakage.

---

## Project Structure
```
MarketRegimeAI/
├── data/
│   ├── raw/               # Raw downloaded data (gitignored)
│   └── processed/         # Labeled and feature-enhanced data (gitignored)
├── src/
│   ├── main.py            # Full pipeline entry point
│   ├── data_loader.py     # Data fetching and loading
│   ├── regime_labeler.py  # ATR-based regime labeling
│   ├── features.py        # Feature engineering functions
│   ├── apply_features.py  # Applies features to labeled data
│   ├── models.py          # Model definitions
│   ├── evaluation.py      # Walk-forward evaluation
│   └── backtest.py        # Regime-based backtesting
├── notebooks/             # Exploratory analysis
├── reports/               # Output reports
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart
```bash
# Clone the repo
git clone https://github.com/ako-K3/MarketRegimeAI.git
cd MarketRegimeAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
cd src
python main.py
```

---

## Research

This project accompanies an ongoing research paper:

**"Which Machine Learning Framework Most Effectively Identifies Financial Market Regimes?"**

The paper evaluates the same pipeline across multiple models and robustness tests including K sensitivity analysis, walk-forward fold sensitivity, scaler comparison, and generalization to QQQ and BTC.

---

## Roadmap

- [ ] VIX and macro feature integration (FRED API)
- [ ] HMM validation against ATR-based labels
- [ ] Optuna hyperparameter tuning
- [ ] Bloomberg Terminal API integration
- [ ] Generalization testing on QQQ and BTC

---

## Author

Built by [Akaki Jvarsheishvili] — connect on [https://www.linkedin.com/in/akaki-jvarsheishvili-891b40359/](#) or [ako_K3](#)