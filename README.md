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
  matchups.py       QB bad-weather sensitivity, pass protection vs. pass rush, blitz vulnerability
  features.py       pregame rolling features, QB features, home-minus-away game table
  tune_features.py  tunes feature settings on 2012-2018 only
  evaluate.py       walk-forward evaluation, metrics, Elo/Vegas baselines
  models.py         model definitions (logistic regression, random forest, gradient boosting, ensemble)
  tune_models.py    tunes tree-model settings on 2012-2018 only
  train.py          evaluates models, fits the final model, saves it to models/
  explain.py        exact per-game factor breakdown of a prediction
  predict.py        predict a week (or one game) with probabilities and top factors
  track.py          live 2026 test of the blitz feature (records predictions before kickoff)
notebooks/
  01_data_exploration.ipynb
  02_features.ipynb
  03_logistic_regression.ipynb
  04_player_availability.ipynb
  05_tree_models.ipynb
app/
  dashboard.py      interactive dashboard (streamlit run app/dashboard.py)
tests/
  test_leakage.py   proves features only use information from before kickoff
data/               downloaded data (git-ignored)
models/             tuned feature settings (tracked) and trained models (git-ignored)
reports/            evaluation results and weekly predictions (reports/predictions/)
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
python -m src.tune_models        # optional: re-tune tree models (~7 min)
python -m src.train              # walk-forward evaluation + save final model
streamlit run app/dashboard.py   # open the dashboard at http://localhost:8501
```

## Predicting games

```bash
python -m src.predict --refresh                    # pull the latest data, predict the next week
python -m src.predict --week 5                     # a specific week
python -m src.predict --game KC@MIA                # one game (away@home)
python -m src.predict --game KC@MIA --qb KC="Justin Fields"   # what if a backup starts?
python -m src.track                                # live 2026 blitz-test scoreboard
```

Example output (2026 week 3):

```
KC @ MIA  (Sun Sep 27)
  MIA 35.8%  |  KC 64.2%   ->  KC favored at 64.2%   (Vegas: MIA 14.8%)
  Top factors:
     KC + 7.2%  Quarterback            Malik Willis +0.04 vs Patrick Mahomes +0.18 EPA/play
    MIA + 6.2%  Home field             MIA at home
     KC + 5.7%  Defense                MIA +0.036 vs KC -0.017 EPA/play allowed (lower is better)
     KC + 5.3%  Scoring & record       MIA -2.7 vs KC +1.5 point diff/game
```

`--qb KC="Justin Fields"` shows how much the Mahomes-to-backup drop is worth.

**How the explanation works:** the model is a sum, `log-odds = Σ weight × feature`, with no intercept. Every feature is a home-minus-away difference, so evenly matched teams at a neutral site start at exactly 50%, and each factor's push is exact. Related features, such as offensive EPA and success rate, are summed into groups (Quarterback, Offense, Defense, Missing skill players, Home field, Rest, Fatigue…) because the model's split of credit between overlapping features is arbitrary. Each group's number is how far the probability would move if that group were even. `tests/test_explain.py` checks that the factors add up exactly to the model's probability.

**Upcoming games:**
- If the schedule doesn't list a starter yet, each team is assumed to start its most recent QB.
- Player absences come from the latest injury report and roster, and are applied to the next week only.
- Weather comes from the forecast within 16 days, and from the venue's climate average beyond that.

The dashboard's **Games** page shows the same breakdown as a chart for any game.

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
| Quarterback | starting QB's EPA/play and CPOE (career history weighted toward recent seasons, shrunk toward a replacement-level prior), experience, QB change flag |
| QB in bad weather | the starting QB's own drop-off in bad weather beyond the league-wide drop, shrunk heavily; applied only in bad-weather games |
| Skill players | share of the team's usual RB/WR/TE touches that is unavailable (injured, suspended, resting), and the biggest single absence |
| Strength | Elo rating |
| Situation | rest days, home-field advantage (trailing 3-season league home margin, 0 at neutral sites), divisional game |
| Fatigue | defensive snaps last game, defensive snaps per game over the last 3 games, defensive snaps in overtime last game, overtime last game, consecutive road games |
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
| **Logistic regression (final)** | **0.653** | **0.6283** | **0.2191** |
| Random forest | 0.647 | 0.6286 | 0.2194 |
| Gradient boosting | 0.639 | 0.6300 | 0.2202 |
| Elo | 0.638 | 0.6367 | 0.2227 |
| Always pick home | 0.537 | 0.6929 | 0.2499 |

The strongest factors are the starting QB's EPA/play, defensive explosive plays allowed, Elo, offensive success rate, QB changes, and missing skill players.

### Model choice: why logistic regression

Random forest and gradient boosting were tuned on the tuning seasons with all features, including travel and weather.

| Tuning 2012–2018 | Log loss |
|---|---|
| **Logistic regression** | **0.6156** |
| Random forest (tuned) | 0.6214 |
| Gradient boosting (tuned) | 0.6221 |
| 70% logistic + 30% random forest | 0.6162 |

- **The best boosting setup is its simplest one:** 3-leaf trees, few of them, and large leaves. More flexible setups do worse, so the signal is mostly linear.
- **Blending doesn't help.** Tree predictions correlate 0.94–0.97 with the logistic regression's, so they add noise rather than new information.
- **The logistic regression is also the easiest model to explain,** since each factor's contribution can be read directly from its weight.

### Matchup features

| Feature | Data | Tuning 2012–18 | Holdout 2019–25 | Status |
|---|---|---|---|---|
| QB bad-weather sensitivity | play-by-play + weather, 2006+ | 0.6154 → **0.6148** | 0.6278 → 0.6283 | In the model (chosen on tuning; within noise on holdout) |
| Pass protection vs. pass rush | QB hits + sacks, 2006+ | 0.6154 → 0.6159 | – | Not used: sack-rate features already capture it |
| Blitz vulnerability | FTN charting, 2022+ | can't test (no data) | 2023–25 only: 0.6246 → 0.6227 | Held out: its only evidence is from holdout seasons |

- **QB bad-weather sensitivity** is shrunk hard: a QB needs about 2,500 bad-weather dropbacks before his own record counts fully.
  - Most weather-sensitive entering 2026: Brock Purdy and Baker Mayfield.
  - Least sensitive: Drake Maye.
  - Jalen Hurts is about average. His completion % drops in bad weather, but his EPA doesn't drop more than other QBs'.
- **Pressure:** the Chargers were the 4th-most pressured offense in 2025, allowing a hit or sack on 20% of dropbacks. But they are among the *best* offenses when blitzed. Their protection problem is losing one-on-one, not the blitz.
- **Blitz vulnerability** shows the largest gain of anything tested so far. Because that evidence comes from holdout seasons, it is being tested live on the 2026 season (below).

### Live 2026 test: blitz vulnerability

The rule was set before any 2026 results: **after the 2026 regular season, if the model with blitz vulnerability (`logistic_blitz`) has lower log loss on 2026 games than the main model, blitz vulnerability joins the model.**

- `python -m src.train` saves both models.
- Every `python -m src.predict` run records both models' probabilities for games that haven't kicked off, in `reports/live_tracking.csv`. A game's row stops updating at kickoff.
- Weeks played before tracking started were backfilled as if live, with models trained only on earlier games: `python -m src.track --backfill`.
- `python -m src.track` prints the scoreboard, which is also on the dashboard's Model page.

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

- **Fatigue** was kept. The first version (total snaps) was neutral on the holdout.
- **Defense-focused fatigue** replaced it and beat it on the tuning seasons (0.6154 vs. 0.6156). Defensive snaps are the opponent's plays, and linebackers and defensive backs play 88–90% of snaps. Snap counts also show the defensive line is the most-rotated group (60%) while the offensive line plays 97%, so the case for defense-focused fatigue is that it's time spent on the field, not a lack of substitution. What mattered:
  - the 3-game defensive load, at about −1.5 points per 10 extra snaps per game
  - defensive snaps in overtime, at about −0.4 points per snap
  - an overtime flag on top of those
- **QB ratings** weight each earlier season at 80% of the one after it (`qb_season_decay = 0.8`). This was a judgment call: tuning slightly preferred equal weight for a QB's whole career (by 0.0002, within noise), but that rated Deshaun Watson on his Houston peak.
- **On the holdout, each change improved log loss by 0.0003** (0.6284 → 0.6278 combined).
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
- [x] Random forest and gradient boosting, plus calibration and ensembles (logistic regression kept)
- [x] Per-game factor explanations and prediction CLI (with QB what-ifs)
- [x] Matchup features: QB weather sensitivity (in the model), pass protection vs. pass rush (not helpful), blitz vulnerability (live 2026 test running)
