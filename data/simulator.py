"""
Stage 1 — Synthetic Trading-Floor Simulator
=============================================
Generates a synthetic dataset of traders over a trading period.
Most traders behave normally; a small number are injected with
rogue behaviour (London-Whale-style slow concealed position buildup,
off-hours trading, escalating hidden losses).

Because misconduct is *injected*, we have perfect ground-truth labels,
which is what lets the POC measure false-positive rate and detection
lead time honestly.

Outputs:
    - trades.csv         : daily per-trader trade/behaviour records
    - communications.csv : daily per-trader communication snippets
    - labels.csv         : ground-truth rogue flag + rogue-onset day
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

# ---- Communication templates -------------------------------------------------
NORMAL_COMMS = [
    "Closed the book early today, all positions within limits.",
    "Good session, hedged the EUR exposure as planned.",
    "Risk looks fine, nothing unusual to report.",
    "Reconciled with middle office, numbers tie out.",
    "Quiet day on the desk, standard flow.",
    "Booked the client trade, confirmation sent to ops.",
    "Limits checked, all green on the risk dashboard.",
    "Took some profit on the rates position, staying conservative.",
]

# Language associated with concealment / stress / rule circumvention.
ROGUE_COMMS = [
    "Don't loop in risk on this one yet, I'll manage it myself.",
    "Need to keep this position off the main book for now.",
    "If anyone asks, the marks are fine, just don't dig into it.",
    "I can recover this if they give me a bit more time, don't escalate.",
    "Restructure the trade so it doesn't trip the limit alert.",
    "Keep this between us, compliance doesn't need the detail.",
    "I'll fix the P&L before month-end, no need to flag it.",
    "Move it to the other account so it stays under the radar.",
]


def _simulate_trader(trader_id, n_days, is_rogue):
    """Generate one trader's daily records."""
    # Each trader has a personal behavioural baseline.
    base_position = RNG.uniform(5, 20)          # $M typical notional
    base_trades = RNG.integers(8, 25)           # trades/day
    base_hour = RNG.uniform(10, 15)             # typical trading hour (24h)
    base_pnl_vol = RNG.uniform(0.2, 0.8)        # P&L volatility $M

    rows = []
    rogue_onset = None
    if is_rogue:
        # Misconduct begins partway through the period and escalates.
        rogue_onset = int(RNG.integers(int(n_days * 0.45), int(n_days * 0.65)))

    for day in range(n_days):
        rogue_active = is_rogue and day >= rogue_onset
        # Escalation factor grows the longer the rogue behaviour runs.
        if rogue_active:
            esc = 1 + (day - rogue_onset) / max(1, (n_days - rogue_onset)) * 4.0
        else:
            esc = 1.0

        position = abs(RNG.normal(base_position, base_position * 0.15)) * (esc if rogue_active else 1.0)
        n_trades = max(1, int(RNG.normal(base_trades, 3) * (1 + 0.4 * (esc - 1))))
        # Off-hours trading creeps in for rogue traders.
        if rogue_active:
            trade_hour = RNG.choice([base_hour, RNG.uniform(19, 23)], p=[0.5, 0.5])
        else:
            trade_hour = RNG.normal(base_hour, 1.0)

        # P&L: normal traders mean-revert near zero; rogue traders accumulate
        # a growing, concealed loss.
        if rogue_active:
            daily_pnl = RNG.normal(-0.3 * esc, base_pnl_vol * esc)
        else:
            daily_pnl = RNG.normal(0.0, base_pnl_vol)

        # Cancellation / amendment rate spikes when hiding trades.
        cancel_rate = RNG.beta(2, 20)
        if rogue_active:
            cancel_rate = min(0.9, cancel_rate + 0.15 * (esc - 1))

        # Limit utilisation (fraction of allowed limit consumed).
        limit_util = min(1.3, position / (base_position * 2.5))

        # Communication snippet for the day.
        if rogue_active and RNG.random() < 0.5:
            comm = RNG.choice(ROGUE_COMMS)
        else:
            comm = RNG.choice(NORMAL_COMMS)

        rows.append({
            "trader_id": trader_id,
            "day": day,
            "position_notional_m": round(position, 3),
            "n_trades": n_trades,
            "trade_hour": round(float(trade_hour), 2),
            "daily_pnl_m": round(float(daily_pnl), 3),
            "cancel_rate": round(float(cancel_rate), 3),
            "limit_util": round(float(limit_util), 3),
            "communication": comm,
            "rogue_active": int(rogue_active),  # day-level ground truth
        })
    return rows, rogue_onset


def generate(n_traders=50, n_rogue=3, n_days=250, out_dir=None):
    out_dir = out_dir or os.path.dirname(__file__)
    rogue_ids = set(RNG.choice(n_traders, size=n_rogue, replace=False).tolist())

    all_rows, label_rows = [], []
    for tid in range(n_traders):
        is_rogue = tid in rogue_ids
        rows, onset = _simulate_trader(f"T{tid:03d}", n_days, is_rogue)
        all_rows.extend(rows)
        label_rows.append({
            "trader_id": f"T{tid:03d}",
            "is_rogue": int(is_rogue),
            "rogue_onset_day": onset if onset is not None else -1,
        })

    df = pd.DataFrame(all_rows)
    labels = pd.DataFrame(label_rows)

    trades = df.drop(columns=["communication"])
    comms = df[["trader_id", "day", "communication", "rogue_active"]]

    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    comms.to_csv(os.path.join(out_dir, "communications.csv"), index=False)
    labels.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    print(f"Generated {n_traders} traders ({n_rogue} rogue) x {n_days} days "
          f"= {len(df)} records")
    print(f"Rogue trader IDs: {sorted(labels[labels.is_rogue==1].trader_id)}")
    return trades, comms, labels


if __name__ == "__main__":
    generate()
