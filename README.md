# NFL Game Predictor Model

A machine learning model that predicts the **win probability** of each team in an NFL game and explains **which factors drove the prediction**.

Built with Python, pandas and scikit-learn on [nflverse](https://github.com/nflverse/nflverse-data) data.

## Approach

- **One row per game**, from the home team's perspective. The target is `home_win`.
- **Features are home-minus-away differences** in pregame team strength: offensive and defensive EPA/play, success rate, pass/rush efficiency, turnover rates, recent form, quarterback performance, Elo rating, rest and home-field advantage.
- **No leakage:** every feature uses only information available before kickoff.
- **Walk-forward validation:** train on seasons up to N, test on season N+1.
- **Models:** logistic regression, random forest and gradient boosting, evaluated on log loss, Brier score and calibration.
- **Benchmark:** Vegas closing lines are used only to measure the model, never as a feature. Favorites win about 66.5% of games from 2006 to 2025.

## Project structure

```
src/
  config.py         paths and constants
  data_loader.py    downloads and caches nflverse data
notebooks/
  01_data_exploration.ipynb
data/               downloaded data (git-ignored)
models/             trained models (git-ignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader        # downloads ~390 MB of data (2006 to present)
```

## Roadmap

- [x] Data loader and exploration
- [ ] Team-game stats from play-by-play
- [ ] Pregame rolling features, QB features, Elo
- [ ] Baselines and logistic regression
- [ ] Random forest and gradient boosting, plus calibration
- [ ] Per-game factor explanations and prediction CLI
