# NFL Game Predictor Model

A machine learning model that predicts the **win probability** of each team in an NFL game and explains **which factors drove the prediction**.

Built with Python, pandas and scikit-learn on [nflverse](https://github.com/nflverse/nflverse-data) data, with weather from [Open-Meteo](https://open-meteo.com) (CC BY 4.0). Includes an interactive Streamlit dashboard.

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
  players.py        skill-player (RB/WR/TE) role sizes and who is missing each game
  venues.py         stadium locations, time zones, altitude, kickoff times (with schedule fixes)
  weather.py        game-time weather from Open-Meteo (archive, forecast, climate fallback)
  situational.py    travel and body clock, fatigue (snaps), cold shock
  features.py       pregame rolling features, QB features, home-minus-away game table
  tune_features.py  tunes feature settings on 2012-2018 only
  evaluate.py       walk-forward evaluation, metrics, Elo/Vegas baselines
  models.py         model definitions
  train.py          evaluates models, fits the final model, saves it to models/
notebooks/
  01_data_exploration.ipynb
  02_features.ipynb
  03_logistic_regression.ipynb
  04_player_availability.ipynb
app/
  dashboard.py      interactive dashboard (streamlit run app/dashboard.py)
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
                                 # (first run also downloads weather, ~15 min)
python -m pytest tests           # leakage checks
python -m src.tune_features      # optional: re-tune feature settings (~1 min)
python -m src.train              # walk-forward evaluation + save final model
streamlit run app/dashboard.py   # open the dashboard at http://localhost:8501
```

## Dashboard

`streamlit run app/dashboard.py` opens an interactive dashboard with six sections:

| Section | What it shows |
|---|---|
| Overview | Headline accuracy vs. Vegas and Elo; the decline of home-field advantage |
| Teams | A team's pregame Elo, offensive/defensive EPA and QB rating over time, plus its game log with model and Vegas win probabilities |
| Games | Every game in a week with out-of-sample model, Vegas and Elo probabilities; model-vs-Vegas scatter for a season |
| Players | Biggest skill-player absences and a team's weekly missing share |
| Situational | Win rates by body-clock kickoff time, time zones traveled, cold shock, wind, and fatigue |
| Model | Calibration, log loss by season, and the model's weights |

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
| Skill players | share of the team's usual RB/WR/TE touches that is unavailable (injured, suspended, resting), and the biggest single absence |
| Strength | Elo rating |
| Situation | rest days, home-field advantage (trailing 3-season league home margin, 0 at neutral sites), divisional game |
| Fatigue | offensive + defensive snaps last game, defensive snaps last game, 3-game snap load, overtime last game, consecutive road games |
| Travel* | distance, time zones crossed east/west, body-clock kickoff time (a 1 PM ET game is 10 AM for a West Coast team), altitude gain |
| Weather* | indoor, temperature, wind, precipitation, cold shock vs. the team's home climate, wind × passing edge |

\*Built and kept for the tree models, but not used by the logistic regression. They did not improve log loss on the tuning seasons (see below).

## Evaluation

**Walk-forward:** to predict season T, the model is trained only on seasons before T. The test seasons are split in two:

- **Tuning (2012–2018):** used to choose the feature settings and the regularization strength.
- **Holdout (2019–2025):** never used for any choice. This is the honest estimate.

Log loss is the main metric (lower is better) because the goal is accurate probabilities, not just picking winners.

| Holdout 2019–2025 (1,954 games) | Accuracy | Log loss | Brier |
|---|---|---|---|
| Vegas moneyline (benchmark) | 0.664 | 0.6083 | 0.2105 |
| **Logistic regression** | **0.656** | **0.6282** | **0.2191** |
| Logistic regression without player features | 0.647 | 0.6295 | 0.2199 |
| Elo | 0.638 | 0.6367 | 0.2227 |
| Always pick home | 0.537 | 0.6929 | 0.2499 |

The strongest factors are the starting QB's EPA/play, defensive explosive plays allowed, Elo, offensive success rate, QB changes, and missing skill players.

### Skill-player availability

Team stats already reflect the players who have been playing. What they miss is **who is out today**. For each game:

1. **Role size:** each RB, WR and TE gets his share of the team's targets plus carries in earlier games.
2. **Participation:** how regularly he has been playing. It fades while he is out, as the team's stats adjust without him.
3. **Missing:** he is still on the roster but doesn't play.
   - For past games, this means he didn't appear in that game's stats. Inactive lists are public 90 minutes before kickoff, so this is pregame information.
   - For upcoming games, it comes from the injury report (Out or Doubtful) and roster status (IR).

A first version valued players by EPA per game and gave no gain; role size is steadier. On the holdout, player features improve accuracy by 0.9 points, with the biggest gains in games with lopsided absences.

### Travel, weather and fatigue

Each group was added separately and kept only if it improved log loss on the tuning seasons.

| Tuning 2012–2018 | Log loss |
|---|---|
| Base + fatigue | **0.6156** |
| Base | 0.6165 |
| Base + weather | 0.6168 |
| Base + travel | 0.6168 |

- **Fatigue** was kept. On the holdout it was neutral (0.6285 vs. 0.6282 without), which is within noise.
- **The West Coast early-kickoff effect** doesn't show up once team strength is controlled for. West Coast teams traveling east won 30% from 2006–2012 but 61% from 2019–2025.
- **Cold shock is real but rare.** Road teams playing 25°F+ colder than their home climate win 38.9%, vs. 45% otherwise, but that's only about 10% of outdoor games.
- **Data fixes:** the nflverse schedule lists the 2025 international games (São Paulo, Dublin, London, Berlin, Madrid) at U.S. stadiums. `venues.py` corrects them.

## Roadmap

- [x] Data loader and exploration
- [x] Team-game stats from play-by-play
- [x] Pregame rolling features, QB features, Elo, leakage tests
- [x] Baselines and logistic regression
- [x] Skill-player availability; travel, weather and fatigue features
- [x] Interactive dashboard
- [ ] Random forest and gradient boosting, plus calibration
- [ ] Per-game factor explanations and prediction CLI
