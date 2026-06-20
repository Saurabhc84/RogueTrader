"""
Stage 3 — Base Models
=====================
Three complementary anomaly signals, each producing a per-(trader, day)
risk score in [0, 1]:

  1. IsolationForest  -> unsupervised deviation from behavioural baseline
  2. LSTM autoencoder -> temporal sequence anomaly (reconstruction error)
  3. NLP classifier   -> concealment language in communications

Each model exposes .fit() and .score(); scores are min-max normalised
so the fusion layer can combine them.

NOTE on libraries:
  - IsolationForest uses scikit-learn (always available).
  - The LSTM uses PyTorch. If torch is not installed, LSTMScorer falls
    back to a z-score-based temporal anomaly so the pipeline still runs,
    but for your capstone install torch and use the real LSTM.
  - The NLP classifier uses scikit-learn TF-IDF + LogisticRegression,
    which is a defensible lightweight stand-in. For the full proposal
    you can swap in a fine-tuned BERT; the interface is identical.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import os
import sys
# Make `features` importable whether run via orchestrator or standalone.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from features import feature_columns

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False


def _norm(x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return MinMaxScaler().fit_transform(x).ravel()


# ----------------------------------------------------------------------------
# 1. Isolation Forest
# ----------------------------------------------------------------------------
class IForestScorer:
    def __init__(self, contamination=0.05):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300, contamination=contamination, random_state=42
        )
        self.cols = feature_columns()

    def fit(self, feat: pd.DataFrame):
        X = self.scaler.fit_transform(feat[self.cols])
        self.model.fit(X)
        return self

    def score(self, feat: pd.DataFrame):
        X = self.scaler.transform(feat[self.cols])
        # Higher score_samples = more normal; invert so high = anomalous.
        raw = -self.model.score_samples(X)
        return _norm(raw)


# ----------------------------------------------------------------------------
# 2. LSTM Autoencoder (temporal sequence anomaly)
# ----------------------------------------------------------------------------
if TORCH:
    class _LSTMAE(nn.Module):
        def __init__(self, n_feat, hidden=32, seq_len=10):
            super().__init__()
            self.seq_len = seq_len
            self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_feat)

        def forward(self, x):
            _, (h, _) = self.enc(x)              # h: (1, B, hidden)
            rep = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
            dec, _ = self.dec(rep)
            return self.out(dec)


class LSTMScorer:
    """Sliding-window LSTM autoencoder; reconstruction error = anomaly."""
    def __init__(self, seq_len=10, epochs=15, hidden=32):
        self.seq_len = seq_len
        self.epochs = epochs
        self.hidden = hidden
        self.cols = feature_columns()
        self.scaler = StandardScaler()

    def _windows(self, feat):
        """Build per-trader sliding windows; return windows + end-row index."""
        Xs, idx = [], []
        for tid, g in feat.groupby("trader_id"):
            g = g.sort_values("day")
            arr = self.scaler.transform(g[self.cols])
            rows = g.index.to_numpy()
            for i in range(len(arr) - self.seq_len + 1):
                Xs.append(arr[i:i + self.seq_len])
                idx.append(rows[i + self.seq_len - 1])  # label the window's last day
        return np.array(Xs, dtype=np.float32), np.array(idx)

    def fit(self, feat):
        self.scaler.fit(feat[self.cols])
        if not TORCH:
            return self
        X, _ = self._windows(feat)
        self.model = _LSTMAE(len(self.cols), self.hidden, self.seq_len)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X)
        self.model.train()
        for ep in range(self.epochs):
            opt.zero_grad()
            recon = self.model(Xt)
            loss = loss_fn(recon, Xt)
            loss.backward()
            opt.step()
        return self

    def score(self, feat):
        feat = feat.sort_values(["trader_id", "day"])
        scores = pd.Series(0.0, index=feat.index)

        if not TORCH:
            # Fallback: temporal anomaly via mean absolute z across features.
            zcols = [c for c in self.cols if c.endswith("_z")]
            scores = feat[zcols].abs().mean(axis=1)
            return _norm(scores.values)

        X, idx = self._windows(feat)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(torch.tensor(X)).numpy()
        err = ((recon - X) ** 2).mean(axis=(1, 2))   # per-window MSE
        s = pd.Series(0.0, index=feat.index)
        s.loc[idx] = err
        # First (seq_len-1) days per trader have no window; backfill with min.
        s = s.replace(0.0, np.nan).groupby(feat["trader_id"]).bfill().fillna(s.min())
        return _norm(s.values)


# ----------------------------------------------------------------------------
# 3. NLP Concealment Classifier
# ----------------------------------------------------------------------------
class NLPScorer:
    """
    TF-IDF + LogisticRegression on communication text.
    Trained weakly: we label snippets by presence of concealment cues,
    mirroring how you'd bootstrap labels from regulatory proceedings.
    Swap in BERT later via the same .fit/.score interface.
    """
    CUES = ["don't loop", "off the main book", "don't dig", "don't escalate",
            "limit alert", "between us", "before month-end", "under the radar",
            "keep this", "doesn't need", "manage it myself"]

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def _weak_label(self, text):
        t = text.lower()
        return int(any(cue in t for cue in self.CUES))

    def fit(self, comms: pd.DataFrame):
        y = comms["communication"].apply(self._weak_label)
        X = self.vec.fit_transform(comms["communication"])
        self.clf.fit(X, y)
        return self

    def score(self, comms: pd.DataFrame):
        X = self.vec.transform(comms["communication"])
        return self.clf.predict_proba(X)[:, 1]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(__file__))
    dp = os.path.join(here, "data")
    feat = pd.read_csv(os.path.join(dp, "features.csv"))
    comms = pd.read_csv(os.path.join(dp, "communications.csv"))

    print("TORCH available:", TORCH)

    iforest = IForestScorer().fit(feat)
    feat["score_iforest"] = iforest.score(feat)
    print("IForest score range:", feat.score_iforest.min().round(3), "-",
          feat.score_iforest.max().round(3))

    lstm = LSTMScorer(epochs=8).fit(feat)
    feat["score_lstm"] = lstm.score(feat)
    print("LSTM score range:", round(feat.score_lstm.min(), 3), "-",
          round(feat.score_lstm.max(), 3))

    nlp = NLPScorer().fit(comms)
    comms["score_nlp"] = nlp.score(comms)
    print("NLP score range:", round(comms.score_nlp.min(), 3), "-",
          round(comms.score_nlp.max(), 3))

    feat = feat.merge(comms[["trader_id", "day", "score_nlp"]],
                      on=["trader_id", "day"], how="left")
    feat.to_csv(os.path.join(dp, "scored.csv"), index=False)
    print("Saved scored.csv", feat.shape)
