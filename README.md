# MLB First-Inning Pitch-Count Model (a negative result)

Can a pitcher's **first-inning pitch count** in their *next* start be predicted
from pre-game information (their recent first-inning counts and Statcast
velocity/spin/command summaries)?

Short answer: **no, not meaningfully.** A gradient-boosted model does not beat
simply predicting the training-set average for every game. This repository is
published as an honest negative result, with every number reproducible from the
committed data.

## The question

For each pitcher-game, predict that pitcher's first-inning pitch count in their
**next** start, using only information available beforehand.

## Data

- `data/features/pitcher_games_enhanced.csv` — 15,977 pitcher-games, one row per
  pitcher per game. Built from MLB Statcast (2023–2025) by `src/data/scrape.py`:
  first-inning pitches aggregated per pitcher-game into the real first-inning
  pitch count (`pitches_inning1_true`) plus per-game Statcast summaries
  (velocity mean/std, spin, plate-appearances faced, first-inning strikeouts,
  command spread). The data includes regular-season, spring-training (~1,000
  rows) and postseason games; no game type is filtered out.
- `data/game_dates.csv` — each game's official calendar date, fetched from MLB's
  public statsapi schedule by `scripts/fetch_game_dates.py`. This is needed
  because **`game_pk` is not monotonic with date** (e.g. game_pk 716531 is
  2023-09-18 but 716621 is 2023-09-12), so games must be ordered by their real
  date, not by `game_pk`.
- The raw Statcast download is **not** committed; the two CSVs above are, and the
  evaluation runs from them directly.

## Features and target

- **Target:** the pitcher's next-game `pitches_inning1_true`, where "next" is by
  official date (shift of -1 within pitcher, date-ordered). The current game
  precedes the predicted game, so there is no look-ahead.
- **Features (known at or before the current game):** 5- and 10-game rolling
  averages of the pitcher's real first-inning pitch count, plus that game's
  Statcast summaries (`velo_avg`, `velo_std`, `spin_avg`, `pas_faced`,
  `ks_inning1`, `command_std`).

## What was tried

An XGBoost regressor (500 trees, depth 6, learning rate 0.05; configured in
`src/evaluate.py`, with fixed hyperparameters — no tuning) trained on a **global
chronological split**: all games sorted by official date, split at a date
boundary into the first ~80% (training) and last ~20% (test). It is compared
against two naive baselines:

1. **train-mean** — predict the single constant `mean(y_train)` for every test
   game;
2. **per-pitcher prior-games average** — predict the pitcher's own average
   first-inning count over their past games only.

Because a training row's label is its pitcher's *next* game, training rows whose
target game falls in the test period are dropped (filtered on the target's
date), so no label leaks across the split. The evaluation reports how many rows
this removes.

## Results

Reproduce with:

```bash
pip install -r requirements.txt
python src/evaluate.py
```

Printed output (deterministic, `random_state=42`, pinned dependencies):

| Predictor | Test MAE (pitches) | R² |
|---|---|---|
| Model (XGBoost) | **5.315** | **-0.045** |
| Baseline: train mean (16.70) | 5.183 | — |
| Baseline: per-pitcher prior-games average | 5.296 | — |

n_train = 11,869, n_test = 3,073 (283 training rows dropped because their target
game falls in the test period). Split is chronological by official date: train
2023-03-15 to 2025-05-19, test 2025-05-25 to 2025-10-28. The model's MAE is
**worse** than both baselines and its R² is negative — it is beaten by predicting
a single constant.

## Why it likely fails

A pitcher's first-inning pitch count is close to noise given pre-game
information. It is driven mostly by how the half-inning happens to unfold —
baserunners, foul balls, umpire calls, how many batters reach — rather than by
anything stable and knowable beforehand. The outcome variable has a narrow,
roughly symmetric spread, so a constant near the mean is hard to beat, and the
pre-game features carry little signal about the specific next game. A negative
R² is the expected result when features are near-uninformative and the model
adds variance without reducing bias.

## Bugs found and fixed in the original (lessons)

This project began with inflated, unsupported accuracy claims. Two defects made
its headline number (an MAE of ~3.94 presented as a success) meaningless:

1. **Fabricated target.** The pipeline trained on `pitches_inning1` — which was
   actually the *maximum pitch number* reached in the first inning, an
   undercount averaging about 6 — multiplied by an arbitrary `3.2`
   (`pitches_inning1 * 3.2`). That scaled quantity had a smaller spread (std
   ~5.0 vs ~6.6 for the real count), which flattered the error. Fixed: the model
   now uses the real `pitches_inning1_true`, and the fabricated column is gone.
2. **Mislabeled / mis-ordered split.** The code sorted by `(pitcher, game_pk)`
   and took the last 20% of rows as a "time-based" test set. That was wrong
   twice over: the sort was pitcher-first (so the test set was the last 20% of
   *pitchers by id*), and `game_pk` is not even chronological. Fixed: games are
   ordered by their official calendar date everywhere — for rolling features,
   for the next-game target, and for a genuine global date split.

The real lesson: an impressive-looking error means nothing until you (a) predict
the real quantity, (b) order and hold out by actual time, and (c) compare
against a trivial baseline. Doing all three here turns a "success" into an
honest negative result.

## What would come next

- Stronger pre-game context: opponent lineup quality, rest days, park, weather,
  catcher — features more plausibly tied to how a first inning unfolds.
- Model the full first-inning event sequence (batters faced, outcomes) rather
  than the pitch count directly.
- Treat the problem as distributional (predict a range) rather than a point
  estimate, and judge it against the same naive baselines.

## Repository layout

```
data/features/pitcher_games_enhanced.csv   committed source data
data/game_dates.csv                        official date per game_pk (committed)
scripts/fetch_game_dates.py                fetch dates from MLB statsapi (stdlib)
src/data/scrape.py                         Statcast provenance (not run by eval)
src/data/etl.py                            feature engineering + date split
src/evaluate.py                            the evaluation (prints the table above)
```

## License

MIT — see [LICENSE](LICENSE).
