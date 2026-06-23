# Institutional Stochastic Forecasting Engine — Nasdaq (QQQ)

A probabilistic forecasting system for QQQ built **entirely on stochastic processes,
state-space modelling, probability theory and regime modelling**.

> No technical indicators. No RSI. No MACD. No moving-average crossovers.
> Every signal is derived from probability, latent-state inference and Monte Carlo.

## Run it

Open `QQQ_Stochastic_Forecasting_Engine.ipynb` in
[Google Colab](https://colab.research.google.com/) and run all cells.
The first cell installs every dependency. If live data is unavailable the
notebook falls back to a reproducible regime-switching proxy so it still runs
end-to-end.

## Modules

| # | Module | Methods |
|---|--------|---------|
| 0 | Data & stochastic features | log-returns, realized vol, vol clustering, rolling entropy, Hurst (R/S), Higuchi fractal dimension, vol-of-vol |
| 1 | Markov Chain Engine | 1st / 2nd / 3rd order + Variable-Length Markov Chain, transition heatmaps, OOS predictive comparison |
| 2 | Hidden Markov Model | Gaussian HMM, AIC/BIC selection (2–6 states), regime stats, Viterbi overlay |
| 3 | Hidden Semi-Markov Model | explicit-duration HSMM, duration distributions, HMM vs HSMM accuracy |
| 4 | Markov-Switching Volatility | 4-regime switching variance (Low/Med/High/Extreme), persistence & durations |
| 5 | Kalman Filtering | KF (latent trend/drift) + EKF + UKF (latent stochastic volatility) |
| 6 | Particle Filter / SMC | bootstrap particle filter, 1k / 5k / 10k particles |
| 7 | Bayesian Update Engine | daily posterior regime filtering, entropy uncertainty |
| 8 | Monte Carlo Forecasting | 1k/5k/10k regime-switching paths, 1–60 day horizons, drawdown & new-high probabilities |
| 9 | Ensemble Model | accuracy-weighted probability blend of all models |
| 10 | Backtesting | probability-only signals, full metric suite (Sharpe, Sortino, Calmar, Omega, Ulcer, Tail Ratio, Profit Factor, …) |
| 11 | Robustness | walk-forward retraining, calibration, bootstrap, Monte Carlo stress, OOS |
| 12 | Research Report | auto-generated research note + forecast dashboard |

## Notes

- Modular, self-contained code — no placeholders or pseudo-code.
- Probabilities in the backtest are computed causally (forward filtering);
  Module 11 provides true out-of-sample walk-forward validation.
- Research artefact for systematic-strategy study. **Not investment advice.**
