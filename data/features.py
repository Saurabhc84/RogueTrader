"""
Stage 2 — Feature Engineering
==============================
Transforms raw daily trade records into behavioural features:
  - rolling means/stds (short and long windows)
  - per-trader z-scores (deviation from a trader's OWN baseline)
  - temporal lag / velocity features
  - off-hours and limit-breach indicators

The per-trader z-score is central: rogue detection is about a trader
deviating from THEIR baseline, not from the desk average.
"""

import pandas as pd
import numpy as np

RAW_COLS = ["position_notional_m", "n_trades", "trade_hour",
            "daily_pnl_m", "cancel_rate", "limit_util"]


def engineer(trades: pd.DataFrame, short=5, long=20) -> pd.DataFrame:
    trades = trades.sort_values(["trader_id", "day"]).copy()
    out = []

    for tid, g in trades.groupby("trader_id"):
        g = g.sort_values("day").copy()

        # Off-hours flag (before 7am or after 18:00).
        g["off_hours"] = ((g["trade_hour"] < 7) | (g["trade_hour"] > 18)).astype(int)

        # Cumulative concealed P&L (running loss buildup).
        g["cum_pnl_m"] = g["daily_pnl_m"].cumsum()

        for col in RAW_COLS:
            # Rolling baselines.
            g[f"{col}_roll_mean"] = g[col].rolling(long, min_periods=3).mean()
            g[f"{col}_roll_std"] = g[col].rolling(long, min_periods=3).std().fillna(1e-6)
            # Per-trader z-score vs own rolling baseline.
            g[f"{col}_z"] = (g[col] - g[f"{col}_roll_mean"]) / (g[f"{col}_roll_std"] + 1e-6)
            # Short-term velocity (rate of change).
            g[f"{col}_vel"] = g[col].diff(short).fillna(0)

        out.append(g)

    feat = pd.concat(out, ignore_index=True)

    # Replace inf/NaN created by early-window rolling ops.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(0)
    return feat


# Columns the unsupervised / sequence models consume.
def feature_columns():
    cols = []
    for c in RAW_COLS:
        cols += [f"{c}_z", f"{c}_vel"]
    cols += ["off_hours", "cum_pnl_m"]
    return cols


if __name__ == "__main__":
    import os
    here = os.path.dirname(__file__)
    trades = pd.read_csv(os.path.join(here, "trades.csv"))
    feat = engineer(trades)
    feat.to_csv(os.path.join(here, "features.csv"), index=False)
    print("Feature columns:", feature_columns())
    print("Engineered shape:", feat.shape)
    print(feat[feature_columns()].describe().T[["mean", "std", "min", "max"]].round(3))
