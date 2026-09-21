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
notebooks/
  01_data_exploration.ipynb
  02_features.ipynb
tests/
  test_leakage.py   proves features only use information from before kickoff
data/               downloaded data (git-ignored)
models/             trained models (git-ignored)
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

## Baselines (2007–2025, all games)

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Always pick home | 0.558 | 0.6864 | 0.2466 |
| Elo | 0.647 | 0.6288 | 0.2196 |
| Vegas moneyline | 0.667 | 0.6071 | 0.2100 |

## Roadmap

- [x] Data loader and exploration
- [x] Team-game stats from play-by-play
- [x] Pregame rolling features, QB features, Elo, leakage tests
- [ ] Baselines and logistic regression
- [ ] Random forest and gradient boosting, plus calibration
- [ ] Per-game factor explanations and prediction CLI
