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
  team_stats.py     play-by-play -> one row per team per game (EPA, success rate, turnovers...)
  elo.py            Elo ratings (feature + baseline)
  features.py       pregame rolling features, QB features, home-minus-away game table
  tune_features.py  tunes feature settings on 2012-2018 only
  evaluate.py       walk-forward evaluation, metrics, Elo/Vegas baselines
  models.py         model definitions
  train.py          evaluates models, fits the final model, saves it to models/
notebooks/
  01_data_exploration.ipynb
  02_features.ipynb
  03_logistic_regression.ipynb
tests/
  test_leakage.py   proves features only use information from before kickoff
data/               downloaded data (git-ignored)
models/             tuned feature settings (tracked) and trained models (git-ignored)
reports/            evaluation results
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader        # downloads ~390 MB of data (2006 to present)
python -m src.team_stats         # builds data/processed/team_games.parquet
python -m src.features           # builds data/processed/games_features.parquet
python -m pytest tests           # leakage checks
python -m src.tune_features      # optional: re-tune feature settings (~1 min)
python -m src.train              # walk-forward evaluation + save final model
```

## Features

Every team stat is an exponentially weighted average of the team's earlier games this season,
blended with last season's value (regressed toward the league average). That way week 1 predictions
lean on last year and later weeks on current form.

| Group | Features (home minus away) |
|---|---|
| Offense | EPA/play, neutral-situation EPA, success rate, pass EPA/dropback, rush EPA, explosive rate, sack rate, turnovers, red-zone TD rate, pass rate |
| Defense | the same stats allowed |
| Results | points for/against, point differential, win % |
| Recent form | short-memory point differential and EPA |
| Quarterback | starting QB's EPA/play and CPOE (career history, shrunk toward a replacement-level prior), experience, QB change flag |
| Strength | Elo rating |
| Situation | rest days, home-field advantage (trailing 3-season league home margin, 0 at neutral sites), divisional game |

## Evaluation

**Walk-forward:** to predict season T, the model is trained only on seasons before T. The test seasons are split in two:

- **Tuning (2012–2018):** used to choose the feature settings and the regularization strength.
- **Holdout (2019–2025):** never used for any choice. This is the honest estimate.

Log loss is the main metric (lower is better) because the goal is accurate probabilities, not just picking winners.

| Holdout 2019–2025 (1,954 games) | Accuracy | Log loss | Brier |
|---|---|---|---|
| Vegas moneyline (benchmark) | 0.664 | 0.6083 | 0.2105 |
| **Logistic regression** | **0.647** | **0.6295** | **0.2199** |
| Elo | 0.638 | 0.6367 | 0.2227 |
| Always pick home | ~0.54 | ~0.69 | ~0.25 |

The strongest factors are the starting QB's EPA/play, defensive explosive plays allowed, Elo, offensive success rate, and QB changes.

## Roadmap

- [x] Data loader and exploration
- [x] Team-game stats from play-by-play
- [x] Pregame rolling features, QB features, Elo, leakage tests
- [x] Baselines and logistic regression
- [ ] Random forest and gradient boosting, plus calibration
- [ ] Per-game factor explanations and prediction CLI
