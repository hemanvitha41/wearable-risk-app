import pandas as pd
from features import build_features

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

frames = []
for sid in SUBJECTS:
    print(f"Processing {sid}...", flush=True)
    try:
        df = build_features(sid)
        frames.append(df)
        print(f"  {sid}: {len(df)} windows, {int(df['stress'].sum())} stress")
    except Exception as e:
        print(f"  {sid} FAILED: {e}")

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv("data/features_all.csv", index=False)

print("\nTotal windows:", all_df.shape)
print("\nWindows per subject (stress / total):")
print(all_df.groupby("subject")["stress"].agg(["sum", "count"]))
print("\nNaN counts:")
print(all_df.isna().sum()[lambda s: s > 0])
print("\nSaved data/features_all.csv")