import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/features_all.csv")
FEATS = [c for c in df.columns if c not in ("subject", "start_s", "label", "stress")]


def baseline_normalize(d, cols):
    d = d.copy()
    ref = d[d["label"] == 1]
    mu = ref.groupby("subject")[cols].mean()
    sd = ref.groupby("subject")[cols].std().replace(0, np.nan)
    d[cols] = (d[cols].values - mu.loc[d["subject"]].values) / sd.loc[d["subject"]].values
    d[cols] = d[cols].clip(-10, 10)
    return d


norm = baseline_normalize(df, FEATS)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                           class_weight="balanced", random_state=42, n_jobs=-1),
)
model.fit(norm[FEATS].values, norm["stress"].values)

imp = pd.Series(model[-1].feature_importances_, index=FEATS).sort_values(ascending=False)
print("Feature importances:\n")
print(imp.round(3).to_string())

# .joblib, not .pkl, because *.pkl is in .gitignore and Streamlit Cloud needs this file
joblib.dump({"model": model, "features": FEATS}, "src/model.joblib")

# Small raw-feature file for the demo app (1,032 rows, safe to commit)
df[["subject", "start_s", "label", "stress"] + FEATS].to_csv("app/demo_features.csv", index=False)
print("\nSaved src/model.joblib and app/demo_features.csv")