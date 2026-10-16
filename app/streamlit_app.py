from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from plotly.subplots import make_subplots


APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR.parent

st.set_page_config(page_title="Stress Risk Monitor", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(BASE / "src" / "model.joblib")
    return bundle["model"], bundle["features"]


@st.cache_data
def load_data():
    return pd.read_csv(APP_DIR / "demo_features.csv")


model, FEATS = load_model()
df = load_data()
@st.cache_resource
def heldout_model(subject):
    """Train on every subject EXCEPT this one, using baseline-calibrated features."""
    train = df[df["subject"] != subject]
    ref = train[train["label"] == 1]
    mu = ref.groupby("subject")[FEATS].mean()
    sd = ref.groupby("subject")[FEATS].std().replace(0, np.nan)
    z = (train[FEATS].values - mu.loc[train["subject"]].values) / sd.loc[train["subject"]].values
    z = np.clip(z, -10, 10)
    m = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", random_state=42, n_jobs=-1),
    )
    return m.fit(z, train["stress"].values)

LABEL_NAMES = {1: "Baseline", 2: "Stress task", 3: "Amusement"}
LABEL_COLORS = {1: "#9e9e9e", 2: "#e53935", 3: "#43a047"}


def calibrate_and_score(sub, n_cal, smooth, mdl):
    """Calibrate on the subject's first n_cal baseline windows, then score the rest."""
    sub = sub.sort_values("start_s").reset_index(drop=True)
    cal_idx = sub.index[sub["label"] == 1][:n_cal]
    ref = sub.loc[cal_idx, FEATS]
    mu = ref.mean()
    sd = ref.std().replace(0, np.nan)
    z = ((sub[FEATS] - mu) / sd).clip(-10, 10)
    sub["risk_raw"] = mdl.predict_proba(z.values)[:, 1]
    scored = sub.drop(index=cal_idx).reset_index(drop=True)
    scored["risk"] = scored["risk_raw"].rolling(smooth, min_periods=1).mean()
    scored["minute"] = scored["start_s"] / 60
    return scored

def break_at_gaps(v):
    out, prev = [], None
    for _, r in v.iterrows():
        if prev is not None and r["start_s"] - prev > 45:
            out.append({"minute": (prev + r["start_s"]) / 120, "risk": np.nan})
        out.append({"minute": r["minute"], "risk": r["risk"]})
        prev = r["start_s"]
    return pd.DataFrame(out)

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
subjects = sorted(df["subject"].unique(), key=lambda s: int(s[1:]))
subject = st.sidebar.selectbox("Demo patient", subjects)
n_cal = st.sidebar.slider("Calibration windows (about 30 s each)", 5, 30, 20)
smooth = st.sidebar.slider("Smoothing (windows)", 1, 5, 3)
threshold = st.sidebar.slider("Alert threshold", 0.10, 0.90, 0.50, 0.05)
show_truth = st.sidebar.checkbox("Show ground truth (demo only)", value=True)

st.title("Stress Risk Monitor")
st.caption(
    "Wearable-based self-management demo. The model compares your wrist signals "
    "to your own calm baseline and estimates how likely you are to be in a physiological stress state."
)

scored = calibrate_and_score(df[df["subject"] == subject], n_cal, smooth, heldout_model(subject))
st.caption(f"Scoring {subject} with a model trained only on the other 14 subjects.")

tab1, tab2 = st.tabs(["Session replay", "What-if explorer"])

# ---------------- Tab 1: replay ----------------
with tab1:
    pos = st.slider("Replay position (drag to move through the session)",
                    1, len(scored), len(scored))
    view = scored.iloc[:pos]
    cur = view.iloc[-1]
    recent = view["risk"].tail(2)
    alert = len(recent) == 2 and bool((recent >= threshold).all())

    if alert:
        st.warning(
            "Your signals have been elevated compared with your usual calm baseline. "
            "You might try a few minutes of slow breathing or a short break. "
            "If you feel unwell, follow your care plan or contact your care team."
        )
    else:
        st.success("Your signals are within your usual range.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Current risk score", f"{cur['risk']:.0%}")
    m2.metric("Windows flagged so far", int((view["risk"] >= threshold).sum()))
    m3.metric("Session time", f"{cur['minute']:.0f} min")

    rows = 2 if show_truth else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25] if show_truth else [1.0],
        vertical_spacing=0.06,
    )

    line_df = break_at_gaps(view)
    fig.add_trace(
        go.Scatter(x=line_df["minute"], y=line_df["risk"], mode="lines+markers",
                   name="Risk score", line=dict(color="#1e88e5", width=3)),
        row=1, col=1,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#fb8c00",
                  annotation_text="Alert threshold", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="Risk score", row=1, col=1)
    if show_truth:
        fig.add_trace(
            go.Bar(x=view["minute"], y=[1] * len(view), width=0.45,
                   marker_color=[LABEL_COLORS[l] for l in view["label"]],
                   hovertext=[LABEL_NAMES[l] for l in view["label"]],
                   showlegend=False),
            row=2, col=1,
        )
        fig.update_yaxes(visible=False, row=2, col=1)
        fig.add_annotation(
            text="Ground truth: gray = baseline, red = stress task, green = amusement",
            xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
            font=dict(size=12),
        )

    fig.update_xaxes(range=[0, scored["minute"].max() + 1])
    fig.update_xaxes(title_text="Session time (minutes)", row=rows, col=1)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=50), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Session summary")
    flagged = scored["risk"] >= threshold
    s1, s2 = st.columns(2)
    s1.metric("Share of session flagged", f"{flagged.mean():.0%}")
    s1.metric("Peak risk score", f"{scored['risk'].max():.0%}")
    if show_truth:
        truth = scored["stress"] == 1
        if truth.any():
            s2.metric("Stress windows caught (sensitivity)", f"{flagged[truth].mean():.0%}")
        if (~truth).any():
            s2.metric("Calm windows left alone (specificity)", f"{(~flagged[~truth]).mean():.0%}")

# ---------------- Tab 2: what-if ----------------
with tab2:
    st.write(
        "Drag the sliders to see how the model reacts. Values are in standard "
        "deviations above your personal calm baseline (0 means typical for you)."
    )
    c1, c2, c3 = st.columns(3)
    eda = c1.slider("Skin conductance level (EDA)", -2.0, 8.0, 0.0, 0.1)
    peaks = c2.slider("EDA response peaks", -2.0, 8.0, 0.0, 0.1)
    hr = c3.slider("Heart rate", -3.0, 5.0, 0.0, 0.1)

    x = pd.DataFrame([{f: 0.0 for f in FEATS}])
    x["eda_mean"], x["eda_peaks"], x["hr_mean"] = eda, peaks, hr
    p = float(model.predict_proba(x[FEATS].values)[0, 1])

    st.metric("Estimated stress risk", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption("All other signals are held at your baseline. In real data these features move together.")

st.divider()
st.caption(
    "Demo only. Trained on lab data from 15 healthy adults (WESAD). "
    "Not a medical device and not for diagnosis."
)