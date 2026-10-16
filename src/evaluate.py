"""
Evaluation for the first-inning pitch-count model.

Reproduces every number in the README from the committed data
(``data/features/pitcher_games_enhanced.csv`` + ``data/game_dates.csv``). Run:

    python src/evaluate.py

It trains the model on a global chronological (date-ordered) split and
compares it against two naive baselines:

* **train-mean** -- predict the single constant ``mean(y_train)`` for every
  test game;
* **per-pitcher prior-games average** -- predict the pitcher's own average
  first-inning pitch count over their past games only.

The headline finding is negative: the model does not beat the train-mean
constant, so its R-squared on the test set is negative.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from data.etl import build_dataset, FEATURE_COLS, TARGET_COL, COUNT_COL


def fit_model(X_train, y_train):
    """XGBoost regressor with the same configuration as the original project."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    train_df, test_df, n_dropped = build_dataset()

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    # Model
    model = fit_model(X_train, y_train)
    model_pred = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, model_pred)
    model_r2 = r2_score(y_test, model_pred)

    # Baseline 1: constant train mean
    train_mean = float(y_train.mean())
    mean_mae = mean_absolute_error(y_test, np.full_like(y_test, train_mean, dtype=float))

    # Baseline 2: per-pitcher prior-games average (always defined; see etl)
    prior_pred = test_df["prior_games_avg"].to_numpy(dtype=float)
    prior_mae = mean_absolute_error(y_test, prior_pred)

    tr_lo, tr_hi = train_df["date"].min().date(), train_df["date"].max().date()
    te_lo, te_hi = test_df["date"].min().date(), test_df["date"].max().date()

    print("=" * 64)
    print("First-inning pitch-count model -- honest evaluation")
    print("=" * 64)
    print(f"Target               : next-game {COUNT_COL}")
    print(f"Features             : {', '.join(FEATURE_COLS)}")
    print(f"n_train              : {len(train_df):,}")
    print(f"n_test               : {len(test_df):,}")
    print(f"train rows dropped   : {n_dropped} (target game falls in the test period)")
    print("Split                : global chronological by official date (first ~80% / last ~20%)")
    print(f"  train date range   : {tr_lo} .. {tr_hi}")
    print(f"  test  date range   : {te_lo} .. {te_hi}")
    print("-" * 64)
    print(f"{'Predictor':<34}{'Test MAE':>12}{'R^2':>12}")
    print("-" * 64)
    print(f"{'Model (XGBoost)':<34}{model_mae:>12.3f}{model_r2:>12.3f}")
    print(f"{'Baseline: train mean (%.2f)' % train_mean:<34}{mean_mae:>12.3f}{'':>12}")
    print(f"{'Baseline: prior-games average':<34}{prior_mae:>12.3f}{'':>12}")
    print("-" * 64)
    verdict = (
        "Model does NOT beat the train-mean constant."
        if model_mae >= mean_mae
        else "Model beats the train-mean constant."
    )
    print(verdict)
    print("=" * 64)


if __name__ == "__main__":
    main()
