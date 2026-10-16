"""
Feature engineering for the first-inning pitch-count study.

Builds the model dataset from two committed files so the evaluation
reproduces without a database or a network scrape:

* ``data/features/pitcher_games_enhanced.csv`` -- one row per pitcher-game
  with the real first-inning pitch count and per-game Statcast summaries;
* ``data/game_dates.csv`` -- each game's official calendar date (from MLB
  statsapi; see scripts/fetch_game_dates.py).

Why the date file is needed
---------------------------
``game_pk`` is NOT monotonic with date (e.g. 716531 is 2023-09-18 but 716621
is 2023-09-12). Ordering by game_pk therefore mis-sorts games in time. Every
ordering below uses the official date instead.

Target framing
--------------
For each pitcher-game we predict that pitcher's *next* first-inning pitch
count (``pitches_inning1_true`` shifted by -1 within the pitcher, in date
order). The current game precedes the predicted game, so there is no
look-ahead.

Chronological split
-------------------
All games are sorted by official date and split at a date boundary: games
before the ~80th-percentile date are training, that date and later are test.
No single date straddles the boundary.

The data spans 2023-2025 and includes regular-season, spring-training and
postseason games; no game type is filtered out.
"""

import os

import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
ENHANCED_CSV = os.path.join(_DATA, "features", "pitcher_games_enhanced.csv")
GAME_DATES_CSV = os.path.join(_DATA, "game_dates.csv")

# Columns known at or before the current game, used as model inputs.
FEATURE_COLS = [
    "pitches_rolling_5",
    "pitches_rolling_10",
    "velo_avg",
    "velo_std",
    "spin_avg",
    "pas_faced",
    "ks_inning1",
    "command_std",
]

TARGET_COL = "target_next_pitches"
COUNT_COL = "pitches_inning1_true"  # the real, un-scaled first-inning pitch count


def load_enhanced(path: str = ENHANCED_CSV, dates_path: str = GAME_DATES_CSV) -> pd.DataFrame:
    """Load the committed feature table and attach each game's official date."""
    df = pd.read_csv(path)
    dates = pd.read_csv(dates_path, parse_dates=["date"])[["game_pk", "date"]]
    df = df.merge(dates, on="game_pk", how="left")
    missing = int(df["date"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} rows have no date in {os.path.basename(dates_path)}; "
            "rerun scripts/fetch_game_dates.py"
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-pitcher rolling features and the next-game target, in date order.

    Uses the real ``pitches_inning1_true`` count throughout. (The original
    pipeline trained on ``pitches_inning1`` -- the max pitch number in the
    inning, an undercount -- multiplied by an arbitrary 3.2; that fabrication
    has been removed.)
    """
    # Date order within each pitcher; game_pk only breaks same-date ties.
    df = df.sort_values(["pitcher", "date", "game_pk"]).reset_index(drop=True)

    grp = df.groupby("pitcher")[COUNT_COL]
    df["pitches_rolling_5"] = (
        grp.rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["pitches_rolling_10"] = (
        grp.rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    )

    # Target: the pitcher's next first-inning pitch count (next game in time),
    # and the date of that next game (used to prevent split leakage).
    df[TARGET_COL] = df.groupby("pitcher")[COUNT_COL].shift(-1)
    df["target_date"] = df.groupby("pitcher")["date"].shift(-1)

    # Prior-games average known before the predicted game: running mean of the
    # pitcher's first-inning counts up to and including the current game (all
    # before the next game). Always defined (>=1 game), so no fallback needed.
    df["prior_games_avg"] = grp.expanding().mean().reset_index(0, drop=True)

    # Drop the last game per pitcher (no next game to predict).
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Fill any missing feature values forward from the same pitcher's earlier
    # games only (never later games, never other pitchers), then default
    # remaining gaps to 0.
    for col in FEATURE_COLS:
        df[col] = df.groupby("pitcher")[col].ffill()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Global split by official date; no single date straddles the boundary.

    A training row's target is its pitcher's *next* game. If that next game
    falls on or after the test start date, the training label is drawn from
    the test period and leaks across the boundary, so those training rows are
    dropped (filtered on the target's date). Returns (train, test, n_dropped).
    """
    df = df.sort_values(["date", "game_pk", "pitcher"]).reset_index(drop=True)
    k = int(train_frac * len(df))
    cutoff = df.iloc[k]["date"]
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    leaks = train["target_date"] >= cutoff
    n_dropped = int(leaks.sum())
    train = train[~leaks].copy()
    return train, test, n_dropped


def build_dataset(path: str = ENHANCED_CSV, train_frac: float = 0.8):
    """One call: committed files -> (train_df, test_df, n_dropped)."""
    df = build_features(load_enhanced(path))
    return chronological_split(df, train_frac=train_frac)


if __name__ == "__main__":
    train_df, test_df, n_dropped = build_dataset()
    print(f"train rows: {len(train_df):,} | test rows: {len(test_df):,} | dropped (leak): {n_dropped}")
    print(f"features: {FEATURE_COLS}")
    print(f"target: {TARGET_COL} (next-game {COUNT_COL})")
