# AI-Powered Rogue Trader Early Warning System — POC

Capstone prototype: an ensemble behavioural-surveillance system that detects
rogue-trading patterns earlier and with fewer false positives than a
threshold-based rule engine. Built on a synthetic trading-floor simulator so
ground truth is known and H1/H2 can be measured directly.

## Architecture

```
Synthetic Simulator → Feature Engineering → 3 Base Models → Ensemble Fusion → Dashboard
   (data/)               (data/features.py)   ┌─ Isolation Forest (behavioural)
                                              ├─ LSTM autoencoder (temporal)   → Weighted fusion
                                              └─ NLP classifier (comms)          + SHAP → Streamlit
                                                                    vs. Rule-based baseline
```

## Setup
```bash
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

## Run
```bash
python run_pipeline.py            # simulate → train → benchmark → write outputs/
streamlit run app/dashboard.py    # launch the compliance dashboard
```

## Modules
| File | Stage | Purpose |
|------|-------|---------|
| `data/simulator.py`   | 1 | Synthetic traders + injected rogue behaviour + labels |
| `data/features.py`    | 2 | Rolling stats, per-trader z-scores, velocity, off-hours |
| `models/base_models.py` | 3 | Isolation Forest, LSTM autoencoder (PyTorch), NLP classifier |
| `models/ensemble.py`  | 4–5 | Weighted fusion + SHAP, rule baseline, benchmarking |
| `run_pipeline.py`     | — | End-to-end orchestration |
| `app/dashboard.py`    | 6 | Streamlit: leaderboard, drill-down, comparison, efficiency |

## Notes for the capstone
- **Synthetic data is a deliberate choice** (proposal §6.2): it proves the
  *architecture* and *decision economics*, not real-world accuracy. Frame the
  POC as justification for a supervised pilot.
- **NLP** uses TF-IDF + LogisticRegression as a lightweight stand-in; swap in
  fine-tuned BERT via the same `.fit()/.score()` interface for the full build.
- **LSTM** uses a PyTorch autoencoder (reconstruction error = anomaly). Without
  torch installed it falls back to a z-score signal so the pipeline still runs.
- Tune `n_rogue`, `n_days`, escalation, and the rule thresholds to explore the
  precision/recall/lead-time trade-off for your results chapter.
