import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks

FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32, "LABEL": 700}
WINDOW_S = 60
STEP_S = 30


def load_subject(subject_id):
    path = f"data/WESAD/{subject_id}/{subject_id}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    return {
        "BVP": wrist["BVP"].flatten(),
        "EDA": wrist["EDA"].flatten(),
        "TEMP": wrist["TEMP"].flatten(),
        "ACC": wrist["ACC"],              # shape (N, 3)
        "LABEL": data["label"].flatten(),
    }


def slope(x, fs):
    t = np.arange(len(x)) / fs
    return np.polyfit(t, x, 1)[0]


def hrv_features(bvp, fs):
    try:
        cleaned = nk.ppg_clean(bvp, sampling_rate=fs)
        _, info = nk.ppg_peaks(cleaned, sampling_rate=fs)
        peaks = info["PPG_Peaks"]
        rr = np.diff(peaks) / fs * 1000              # inter-beat intervals, ms
        if len(rr) < 20:
            raise ValueError("too few beats")

        med = np.median(rr)
        good = (rr > 0.8 * med) & (rr < 1.2 * med)    # tighter beat filter
        rr_good = rr[good]

        both = good[:-1] & good[1:]                   # neighbouring valid pairs only
        d = np.diff(rr)[both]
        d = d[np.abs(d) < 0.2 * med]                  # drop implausible jumps
        if len(rr_good) < 20 or len(d) < 10:
            raise ValueError("too few clean beats")

        return {
            "hr_mean": 60000 / rr_good.mean(),
            "sdnn": rr_good.std(ddof=1),
            "rmssd": np.sqrt(np.mean(d ** 2)),
        }
    except Exception:
        return {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan}

def build_features(subject_id):
    s = load_subject(subject_id)
    duration = len(s["EDA"]) / FS["EDA"]
    rows = []
    start = 0
    while start + WINDOW_S <= duration:
        end = start + WINDOW_S

        # Window label = majority label, only if the window is >= 90% pure
        lab = s["LABEL"][int(start * FS["LABEL"]): int(end * FS["LABEL"])]
        vals, counts = np.unique(lab, return_counts=True)
        major = vals[counts.argmax()]
        purity = counts.max() / len(lab)

        if major in (1, 2, 3) and purity >= 0.9:
            sl = lambda x, fs: x[int(start * fs): int(end * fs)]
            eda = sl(s["EDA"], FS["EDA"])
            temp = sl(s["TEMP"], FS["TEMP"])
            bvp = sl(s["BVP"], FS["BVP"])
            acc_mag = np.linalg.norm(sl(s["ACC"], FS["ACC"]), axis=1)

            peaks, props = find_peaks(eda, prominence=0.02)
            row = {
                "subject": subject_id,
                "start_s": start,
                "label": int(major),
                "stress": int(major == 2),
                "eda_mean": eda.mean(),
                "eda_std": eda.std(),
                "eda_slope": slope(eda, FS["EDA"]),
                "eda_peaks": len(peaks),
                "eda_peak_amp": props["prominences"].mean() if len(peaks) else 0.0,
                "temp_mean": temp.mean(),
                "temp_slope": slope(temp, FS["TEMP"]),
                "acc_mean": acc_mag.mean(),
                "acc_std": acc_mag.std(),
            }
            row.update(hrv_features(bvp, FS["BVP"]))
            rows.append(row)
        start += STEP_S

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_features("S2")
    print("Windows:", df.shape)
    print("\nStress vs non-stress windows:")
    print(df["stress"].value_counts())

    feats = [c for c in df.columns if c not in ("subject", "start_s", "label", "stress")]
    print("\nFeature means by class (0 = non-stress, 1 = stress):")
    print(df.groupby("stress")[feats].mean().round(3).T)

    print("\nNaN counts:")
    print(df[feats].isna().sum())

    df.to_csv("data/features_S2.csv", index=False)
    print("\nSaved data/features_S2.csv")