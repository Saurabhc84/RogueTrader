"""
Stage 4 & 5 — Ensemble Fusion + SHAP, Rule-Based Baseline, Benchmarking
=======================================================================
- EnsembleFusion: weighted late-fusion meta-learner (LogisticRegression)
  over the three base scores, with SHAP attribution for explainability (H3).
- RuleBaseline: the threshold-based system the proposal aims to beat.
- benchmark(): computes AUROC, precision/recall, false-positive rate, and
  DETECTION LEAD TIME (days between first alert and rogue-loss event) for
  both systems -> this is the core evidence for H1 and H2.

Ground truth is the injected day-level `rogue_active` flag.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

try:
    import shap
    SHAP = True
except ImportError:
    SHAP = False

BASE_SCORES = ["score_iforest", "score_lstm", "score_nlp"]


class EnsembleFusion:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, df, y):
        self.clf.fit(df[BASE_SCORES], y)
        return self

    def score(self, df):
        return self.clf.predict_proba(df[BASE_SCORES])[:, 1]

    def weights(self):
        return dict(zip(BASE_SCORES, self.clf.coef_[0].round(3)))

    def shap_values(self, df):
        """Per-row SHAP attributions explaining each ensemble score."""
        if SHAP:
            explainer = shap.LinearExplainer(self.clf, df[BASE_SCORES])
            return explainer.shap_values(df[BASE_SCORES])
        # Fallback: contribution = coef * (value - mean).
        coef = self.clf.coef_[0]
        means = df[BASE_SCORES].mean().values
        return (df[BASE_SCORES].values - means) * coef


class RuleBaseline:
    """
    Threshold-based legacy system: fires when a hard limit is breached.
    Deliberately simple and reactive, like real rule engines.
    """
    def __init__(self, limit_util_thr=0.9, pnl_thr=-2.0, cancel_thr=0.5):
        self.limit_util_thr = limit_util_thr
        self.pnl_thr = pnl_thr
        self.cancel_thr = cancel_thr

    def score(self, trades):
        alert = ((trades["limit_util"] > self.limit_util_thr) |
                 (trades["daily_pnl_m"] < self.pnl_thr) |
                 (trades["cancel_rate"] > self.cancel_thr))
        return alert.astype(float).values


def _first_alert_day(alerts, days):
    hit = np.where(alerts >= 1)[0]
    return days[hit[0]] if len(hit) else None


def benchmark(scored, ensemble_score, rule_alert, labels, threshold=0.5):
    """Compute comparative metrics for ensemble vs rule baseline."""
    df = scored.copy()
    df["ensemble"] = ensemble_score
    df["rule"] = rule_alert
    y = df["rogue_active"].values

    ens_pred = (df["ensemble"].values >= threshold).astype(int)
    rule_pred = (df["rule"].values >= 1).astype(int)

    def metrics(pred, score=None):
        m = {
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            # FP rate = FP / (FP + TN)
            "false_positive_rate": (
                ((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
            ),
        }
        if score is not None:
            m["auroc"] = roc_auc_score(y, score)
        return m

    ens_m = metrics(ens_pred, df["ensemble"].values)
    rule_m = metrics(rule_pred)

    # ---- Detection lead time per rogue trader -------------------------------
    lead = []
    for _, row in labels[labels.is_rogue == 1].iterrows():
        tid, onset = row["trader_id"], row["rogue_onset_day"]
        g = df[df.trader_id == tid].sort_values("day")
        days = g["day"].values
        ens_day = _first_alert_day(g["ensemble"].values >= threshold, days)
        rule_day = _first_alert_day(g["rule"].values, days)
        lead.append({
            "trader_id": tid,
            "rogue_onset_day": onset,
            "ensemble_first_alert": ens_day,
            "rule_first_alert": rule_day,
            "ensemble_lead_days": (onset - ens_day) if ens_day is not None else None,
            "rule_lead_days": (onset - rule_day) if rule_day is not None else None,
        })
    lead_df = pd.DataFrame(lead)

    return ens_m, rule_m, lead_df


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dp = os.path.join(here, "data")
    os.makedirs(os.path.join(here, "outputs"), exist_ok=True)
    scored = pd.read_csv(os.path.join(dp, "scored.csv"))
    trades = pd.read_csv(os.path.join(dp, "trades.csv"))
    labels = pd.read_csv(os.path.join(dp, "labels.csv"))

    y = scored["rogue_active"].values
    ens = EnsembleFusion().fit(scored, y)
    scored["ensemble"] = ens.score(scored)
    print("Ensemble weights:", ens.weights())

    rule = RuleBaseline().score(trades)
    ens_m, rule_m, lead_df = benchmark(scored, scored["ensemble"], rule, labels)

    print("\n=== ENSEMBLE ===")
    for k, v in ens_m.items(): print(f"  {k:22s}: {v:.3f}")
    print("=== RULE BASELINE ===")
    for k, v in rule_m.items(): print(f"  {k:22s}: {v:.3f}")
    print("\n=== DETECTION LEAD TIME (days before rogue onset... ")
    print("     positive = caught BEFORE onset; negative = caught after) ===")
    print(lead_df.to_string(index=False))

    scored.to_csv(os.path.join(dp, "final_scored.csv"), index=False)
    lead_df.to_csv(os.path.join(here, "outputs", "lead_time.csv"), index=False)
