import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

df = pd.read_csv("data/features_all.csv")
ALL_FEATS = [c for c in df.columns if c not in ("subject", "start_s", "label", "stress")]


def subject_normalize(d, cols):
    d = d.copy()
    g = d.groupby("subject")[cols]
    d[cols] = (d[cols] - g.transform("mean")) / g.transform("std")
    return d

def baseline_normalize(d, cols):
    """Normalize each subject using ONLY their baseline (label==1) windows,
    like a real app calibrating on a calm starting period."""
    d = d.copy()
    ref = d[d["label"] == 1]
    mu = ref.groupby("subject")[cols].mean()
    sd = ref.groupby("subject")[cols].std().replace(0, np.nan)
    d[cols] = (d[cols].values - mu.loc[d["subject"]].values) / sd.loc[d["subject"]].values
    d[cols] = d[cols].clip(-10, 10)
    return d

def loso(d, feats, model):
    X = d[feats].values
    y = d["stress"].values
    groups = d["subject"].values
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append({
            "subject": groups[te][0],
            "auc": roc_auc_score(y[te], p),
            "sens": tp / (tp + fn),
            "spec": tn / (tn + fp),
        })
    return pd.DataFrame(rows)


models = {
    "LogReg": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)),
    "RandForest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", random_state=42, n_jobs=-1)),
}

no_temp = [f for f in ALL_FEATS if f != "temp_mean"]
setups = {
    "raw": (df, ALL_FEATS),
    "raw, no temp_mean": (df, no_temp),
    "per-subject z-score": (subject_normalize(df, ALL_FEATS), ALL_FEATS),
    "baseline-calibrated": (baseline_normalize(df, ALL_FEATS), ALL_FEATS),
}

summary = []
detail = {}
for setup_name, (d, feats) in setups.items():
    for model_name, model in models.items():
        res = loso(d, feats, model)
        detail[(setup_name, model_name)] = res
        summary.append({
            "features": setup_name, "model": model_name,
            "AUC": f"{res.auc.mean():.3f} ± {res.auc.std():.3f}",
            "sensitivity": f"{res.sens.mean():.3f} ± {res.sens.std():.3f}",
            "specificity": f"{res.spec.mean():.3f} ± {res.spec.std():.3f}",
        })
        print(f"done: {setup_name} / {model_name}", flush=True)

print("\nLeave-one-subject-out results (mean ± std across 15 held-out subjects):\n")
print(pd.DataFrame(summary).to_string(index=False))

best = detail[("baseline-calibrated", "RandForest")].round(3)
print("\nPer-subject results, RandForest + per-subject z-score:\n")
print(best.to_string(index=False))

pd.DataFrame(summary).to_csv("data/loso_results.csv", index=False)