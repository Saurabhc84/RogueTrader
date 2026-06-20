"""
Stage 6 — Streamlit Compliance Dashboard
=========================================
Run AFTER run_pipeline.py:  streamlit run app/dashboard.py

Four views, mirroring how a compliance officer would actually work:
  1. Risk Leaderboard   — traders ranked by ensemble risk
  2. Trader Drill-Down  — anomaly timeline + SHAP "why flagged" + comms
  3. System Comparison  — Ensemble vs Rule baseline (FP collapse, lead time)
  4. Efficiency Model   — translate metrics into analyst hours & cost saved
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the pipeline artifacts (CSVs) live.
# Resolution order:
#   1. ROGUE_OUTPUTS env var, if set  (most portable)
#   2. the hard-coded local folder below, if it exists
#   3. the project's own ./outputs as a fallback
LOCAL_OUTPUTS = r"E:\Saurabh\College\IIM_Sirmaur\MaterialShared by Prof\Term3\Capstone\Claude_rogue\rogue_outputs"

if os.environ.get("ROGUE_OUTPUTS"):
    OUT = os.environ["ROGUE_OUTPUTS"]
elif os.path.isdir(LOCAL_OUTPUTS):
    OUT = LOCAL_OUTPUTS
else:
    OUT = os.path.join(HERE, "outputs")

st.set_page_config(page_title="Rogue Trader Early Warning System",
                   layout="wide", page_icon="🛡️")


@st.cache_data
def load():
    required = ["final_scored.csv", "communications_scored.csv",
                "labels.csv", "lead_time.csv", "benchmark.csv"]
    missing = [f for f in required if not os.path.isfile(os.path.join(OUT, f))]
    if missing:
        st.error(
            f"Could not find these files in:\n\n`{OUT}`\n\n"
            f"Missing: {', '.join(missing)}\n\n"
            "Copy the CSVs produced by the Colab run (rogue_outputs.zip) into that "
            "folder, or set the ROGUE_OUTPUTS environment variable to point at them."
        )
        st.stop()
    scored = pd.read_csv(os.path.join(OUT, "final_scored.csv"))
    comms = pd.read_csv(os.path.join(OUT, "communications_scored.csv"))
    labels = pd.read_csv(os.path.join(OUT, "labels.csv"))
    lead = pd.read_csv(os.path.join(OUT, "lead_time.csv"))
    bench = pd.read_csv(os.path.join(OUT, "benchmark.csv"))
    return scored, comms, labels, lead, bench


scored, comms, labels, lead, bench = load()
BASE = ["score_iforest", "score_lstm", "score_nlp"]
NAMES = {"score_iforest": "Behavioural (IForest)",
         "score_lstm": "Temporal (LSTM)",
         "score_nlp": "Communications (NLP)"}

st.title("🛡️ AI-Powered Rogue Trader Early Warning System")
st.caption("Ensemble behavioural surveillance POC · synthetic data · "
           "JPMorgan Chase case context")

view = st.sidebar.radio("View", ["Risk Leaderboard", "Trader Drill-Down",
                                  "System Comparison", "Efficiency Model"])
threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.caption(f"📂 Data folder:\n`{OUT}`")

# ----------------------------------------------------------------------------
if view == "Risk Leaderboard":
    st.subheader("Trader Risk Leaderboard")
    agg = (scored.groupby("trader_id")
           .agg(peak_risk=("ensemble", "max"),
                mean_risk=("ensemble", "mean"),
                alert_days=("ensemble", lambda s: int((s >= threshold).sum())))
           .reset_index()
           .merge(labels[["trader_id", "is_rogue"]], on="trader_id")
           .sort_values("peak_risk", ascending=False))
    agg["status"] = np.where(agg.peak_risk >= threshold, "⚠️ FLAGGED", "OK")

    c1, c2, c3 = st.columns(3)
    c1.metric("Traders monitored", len(agg))
    c2.metric("Currently flagged", int((agg.peak_risk >= threshold).sum()))
    c3.metric("True rogue traders (ground truth)", int(agg.is_rogue.sum()))

    fig = px.bar(agg.head(20), x="trader_id", y="peak_risk", color="is_rogue",
                 color_continuous_scale=["#4C78A8", "#E45756"],
                 title="Top 20 traders by peak ensemble risk")
    fig.add_hline(y=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg.round(3), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
elif view == "Trader Drill-Down":
    st.subheader("Trader Drill-Down")
    tid = st.selectbox("Select trader", sorted(scored.trader_id.unique()),
                       index=int(np.argmax(
                           scored.groupby("trader_id").ensemble.max().values)))
    g = scored[scored.trader_id == tid].sort_values("day")
    lab = labels[labels.trader_id == tid].iloc[0]

    if lab.is_rogue:
        st.error(f"Ground truth: ROGUE — misconduct onset day {lab.rogue_onset_day}")
    else:
        st.success("Ground truth: clean trader")

    # Risk timeline with component breakdown.
    fig = go.Figure()
    for c in BASE:
        fig.add_trace(go.Scatter(x=g.day, y=g[c], name=NAMES[c],
                                 line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=g.day, y=g.ensemble, name="ENSEMBLE",
                             line=dict(width=3, color="#E45756")))
    fig.add_hline(y=threshold, line_dash="dash", line_color="grey")
    if lab.is_rogue:
        fig.add_vline(x=lab.rogue_onset_day, line_dash="dot", line_color="black",
                      annotation_text="rogue onset")
    fig.update_layout(title=f"Risk timeline — {tid}", height=420,
                      yaxis_title="risk score")
    st.plotly_chart(fig, use_container_width=True)

    # SHAP-style "why flagged" at peak-risk day.
    peak = g.loc[g.ensemble.idxmax()]
    st.markdown(f"**Why flagged — drivers at peak risk (day {int(peak.day)}):**")
    contrib = pd.DataFrame({
        "signal": [NAMES[c] for c in BASE],
        "score": [peak[c] for c in BASE],
    }).sort_values("score", ascending=True)
    st.plotly_chart(
        px.bar(contrib, x="score", y="signal", orientation="h",
               title="Component risk contribution", height=260),
        use_container_width=True)

    # Flagged communications.
    cg = comms[(comms.trader_id == tid) & (comms.score_nlp > 0.5)]
    if len(cg):
        st.markdown("**Communications flagged by NLP:**")
        st.dataframe(cg[["day", "communication", "score_nlp"]].round(3),
                     use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
elif view == "System Comparison":
    st.subheader("Ensemble AI vs Rule-Based Baseline")
    m = bench.set_index("system")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Ensemble AI")
        st.metric("False positive rate", f"{m.loc['Ensemble AI','false_positive_rate']:.1%}")
        st.metric("Recall (rogue days caught)", f"{m.loc['Ensemble AI','recall']:.1%}")
        st.metric("AUROC", f"{m.loc['Ensemble AI','auroc']:.3f}")
    with c2:
        st.markdown("### Rule Baseline")
        st.metric("False positive rate", f"{m.loc['Rule Baseline','false_positive_rate']:.1%}")
        st.metric("Recall (rogue days caught)", f"{m.loc['Rule Baseline','recall']:.1%}")
        st.metric("AUROC", "n/a (binary)")

    st.markdown("### Detection lead time (days before rogue onset)")
    st.caption("Positive = caught before losses build; the core H2 evidence.")
    ld = lead.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ld.trader_id, y=ld.ensemble_lead_days, name="Ensemble"))
    fig.add_trace(go.Bar(x=ld.trader_id, y=ld.rule_lead_days, name="Rule"))
    fig.update_layout(barmode="group", yaxis_title="lead days", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ld, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
elif view == "Efficiency Model":
    st.subheader("Compliance Efficiency Model")
    st.caption("Translate detection metrics into analyst hours and cost. "
               "Adjust assumptions to fit your desk.")
    c1, c2, c3 = st.columns(3)
    daily_alerts = c1.number_input("Daily alerts (legacy system)", 100, 5000, 1000, 50)
    mins_per_alert = c2.number_input("Minutes to triage one alert", 1, 60, 8)
    analyst_cost = c3.number_input("Analyst cost ($/hour)", 20, 300, 75)

    fp_rule = bench.set_index("system").loc["Rule Baseline", "false_positive_rate"]
    fp_ens = bench.set_index("system").loc["Ensemble AI", "false_positive_rate"]
    # Illustrative: legacy industry FP ~95%; ensemble reduces volume of FPs.
    legacy_fp_share = 0.95
    ens_fp_share = max(0.05, legacy_fp_share * (fp_ens / max(fp_rule, 1e-6)) * 0.4)
    ens_fp_share = min(ens_fp_share, 0.40)

    def annual_cost(fp_share):
        wasted = daily_alerts * fp_share
        hrs = wasted * mins_per_alert / 60 * 252
        return hrs, hrs * analyst_cost

    h0, c0 = annual_cost(legacy_fp_share)
    h1, c1v = annual_cost(ens_fp_share)
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Legacy FP rate (assumed)", f"{legacy_fp_share:.0%}")
    cc2.metric("Ensemble FP rate (target)", f"{ens_fp_share:.0%}")
    cc3.metric("Annual analyst-hours saved", f"{h0-h1:,.0f}")
    st.metric("Estimated annual cost saving", f"${c0-c1v:,.0f}")
    st.plotly_chart(
        px.bar(pd.DataFrame({"system": ["Legacy", "Ensemble AI"],
                             "annual_cost": [c0, c1v]}),
               x="system", y="annual_cost", title="Annual triage cost"),
        use_container_width=True)
    st.info("Synthetic illustration. For the capstone, calibrate against "
            "published RegTech benchmarks and stakeholder-interview estimates.")
