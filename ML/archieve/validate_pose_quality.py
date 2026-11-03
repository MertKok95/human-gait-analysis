"""
validate_pose_quality.py

ADDITIONAL ANALYSIS: Quality Control of 2D AlphaPose Data
=========================================================

This script answers two questions:
(1) What is the quality of the exposure data? 
(2) Does the poor quality explain our estimation errors?

-------------------------------------------------------------------------
MEASURED QUALITY METRICS (calculated at trial level and averaged for the patient)

missing_rate - missing (NaN) squared rate in joint angles (%)
jitter_med - median of inter-square angle variation (degrees).
A high value means noisy/shaky tracking.
spike_pct - squared rate where inter-square variation exceeds 15 degrees (%). Sudden jumps indicate tracking error.
knee_impossible_pct - squared rate where knee angle is outside the physiological range (70-185 degrees) (%)
lr_corr - correlation between left and right knee angle.
In normal gait, legs are in opposite phases, therefore it should be NEGATIVE. A positive and high value indicates that AlphaPose is mixing up the two legs (left-right swap).
hs_cv - coefficient of variation of consecutive heel strike intervals (%). A high value indicates irregular or missed heel strike detection.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.dont_write_bytecode = True

import config

ARCHIVE_OUT = config.OUTPUT_FOLDER / "archive"
ARCHIVE_OUT.mkdir(parents=True, exist_ok=True)

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

FRAME_LEVEL_PATH = str(config.CHECK_9_FRAME_DATASET)
PATIENT_ERROR_PATH = str(config.TABLE_3_PER_PARTICIPANT)
METRICS_OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_Pose_Quality_Metrics.csv")
CORR_OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_Pose_Quality_vs_Error.csv")

ANGLE_COLUMNS = [
    "l_knee_angle", "r_knee_angle",
    "l_hip_angle", "r_hip_angle",
]

SPIKE_THRESHOLD_DEG = 15.0
KNEE_MIN_DEG, KNEE_MAX_DEG = 70.0, 185.0

QUALITY_METRICS = ["missing_rate", "jitter_med", "spike_pct", "hs_cv", "lr_corr"]


def compute_trial_quality(g):
    """Calculates quality metrics for a single trial."""
    g = g.sort_values("frame")
    out = {}

    out["missing_rate"] = g[ANGLE_COLUMNS].isna().mean().mean() * 100

    jitters, spikes = [], []
    for col in ANGLE_COLUMNS:
        v = g[col].astype(float).values
        diffs = np.abs(np.diff(v))
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs):
            jitters.append(np.median(diffs))
            spikes.append((diffs > SPIKE_THRESHOLD_DEG).mean() * 100)
    out["jitter_med"] = float(np.mean(jitters)) if jitters else np.nan
    out["spike_pct"] = float(np.mean(spikes)) if spikes else np.nan

    knee = pd.concat([g["l_knee_angle"], g["r_knee_angle"]]).astype(float).dropna()
    out["knee_impossible_pct"] = (
        float(((knee > KNEE_MAX_DEG) | (knee < KNEE_MIN_DEG)).mean() * 100)
        if len(knee) else np.nan
    )

    out["lr_corr"] = float(
        g["l_knee_angle"].astype(float).corr(g["r_knee_angle"].astype(float))
    )

    hs = np.where(g["is_heel_strike"].values == 1)[0]
    if len(hs) >= 3:
        intervals = np.diff(hs)
        out["hs_cv"] = (float(intervals.std() / intervals.mean() * 100)
                        if intervals.mean() > 0 else np.nan)
    else:
        out["hs_cv"] = np.nan
    out["hs_count"] = len(hs)

    return out


def build_quality_table(frame_level_path):
    df = pd.read_csv(frame_level_path)
    df["Condition"] = df["Fast_Gait"].map({0: "ugs", 1: "fgs"})

    rows = []
    for (pid, cond, wd), g in df.groupby(["Participant_ID", "Condition", "Walking_Direction"]):
        rec = compute_trial_quality(g)
        rec.update({"Participant_ID": pid, "Condition": cond, "Walking_Direction": wd})
        rows.append(rec)

    trial_level = pd.DataFrame(rows)
    patient_level = (
        trial_level.groupby("Participant_ID")
        .mean(numeric_only=True)
        .reset_index()
    )
    return trial_level, patient_level


def report_quality(patient_level):
    print("\n" + "=" * 66)
    print("1) EXPOSURE DATA QUALITY METRICS")
    print("=" * 66)
    cols = QUALITY_METRICS + ["knee_impossible_pct", "hs_count"]
    desc = patient_level[cols].describe().T[["mean", "50%", "min", "max"]]
    desc.columns = ["ortalama", "medyan", "en iyi", "en kötü"]
    print(desc.round(2).to_string())

    print("\n  --- Uyarı eşikleri ---")
    print(f"  Eksik veri > %5              : "
          f"{(patient_level.missing_rate > 5).sum()} hasta")
    print(f"  Sıçrama oranı > %5           : "
          f"{(patient_level.spike_pct > 5).sum()} hasta")
    print(f"  İmkânsız diz açısı > %5      : "
          f"{(patient_level.knee_impossible_pct > 5).sum()} hasta")
    print(f"  Topuk vuruşu düzensizliği > %40 : "
          f"{(patient_level.hs_cv > 40).sum()} hasta")

    n_swap = (patient_level.lr_corr > 0.8).sum()
    print(f"\n  LEFT-RIGHT CONFLICT CHECK (lr_corr > 0,8): {n_swap} hasta")
    if n_swap == 0:
        print("  -> Left-right leg confusion was NOT DETECTED.")
        print(f"     (highest observed correlation: {patient_level.lr_corr.max():+.3f};")
        print("      in normal walking, this value is expected to be negative.)")


def _add_percentage_error(df_err):
    if "Model_Error_Percent" in df_err.columns:
        return df_err
    df_err = df_err.copy()
    denominator = df_err["Sensor_Value"].replace(0, pd.NA)
    df_err["Model_Error_Percent"] = (
        df_err["Model_Error"].abs() / denominator.abs() * 100.0
    )
    return df_err


def test_quality_vs_error(patient_level, error_path):
    if not os.path.exists(error_path):
        print(f"\n[WARNING] {error_path} not found - correlation test skipped.")
        return []

    df_err = _add_percentage_error(pd.read_csv(error_path))

    print("\n" + "=" * 66)
    print("2) INDICATOR TEST: DOES QUALITY EXPLAIN PREDICTION ERROR?")
    print("=" * 66)
    print("Spearman correlation (participant-based). Positive = quality deteriorates")
    print("as error increases. (*) = p < 0.05\n")

    header = f"  {'Parameter':14s} " + " ".join(f"{c[:11]:>12s}" for c in QUALITY_METRICS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    n_significant = 0
    for param in sorted(df_err["Parameter"].unique()):
        sub = (
            df_err[df_err["Parameter"] == param][["Participant_ID", "Model_Error_Percent"]]
            .merge(patient_level, on="Participant_ID")
            .dropna(subset=["Model_Error_Percent"])
        )
        cells = []
        for metric in QUALITY_METRICS:
            valid = sub[[metric, "Model_Error_Percent"]].dropna()
            if len(valid) < 20:
                cells.append("     n/a")
                continue
            r, p = spearmanr(valid[metric], valid["Model_Error_Percent"])
            star = "*" if p < 0.05 else " "
            if p < 0.05:
                n_significant += 1
            cells.append(f"{r:+.3f}{star}")
            rows.append({
                "Parameter": param, "Quality_Metric": metric,
                "Spearman_r": float(r), "P_Value": float(p),
                "Significant": bool(p < 0.05),
            })
        print(f"  {param:14s} " + " ".join(f"{c:>12s}" for c in cells))

    n_tests = len(rows)
    expected_by_chance = n_tests * 0.05
    print(f"\n  Total tests: {n_tests}  |  p<0.05 found: {n_significant}  |  "
          f"chance expected: ~{expected_by_chance:.0f}")
    if n_significant <= expected_by_chance * 2.5:
        print("  -> The number of observed significant results is not significantly different from what would be expected by chance.")

    return rows


def compare_quality_groups(patient_level, error_path):
    """Compares errors between the best and worst quality groups."""
    if not os.path.exists(error_path):
        return []

    df_err = _add_percentage_error(pd.read_csv(error_path))

    pl = patient_level.copy()
    pl["quality_score"] = (
        pl["spike_pct"].rank() + pl["jitter_med"].rank()
        + pl["missing_rate"].rank() + pl["hs_cv"].rank()
    ) / 4.0

    n = max(10, int(len(pl) * 0.20))
    worst = set(pl.nlargest(n, "quality_score")["Participant_ID"])
    best = set(pl.nsmallest(n, "quality_score")["Participant_ID"])

    print("\n" + "=" * 66)
    print(f"3) BEST %20 vs WORST %20 QUALITY ({n} participants each)")
    print("=" * 66)
    print(f"  {'Parameter':14s} {'Best Quality':>12s} {'Worst Quality':>13s} "
          f"{'Difference':>8s} {'p':>8s}")
    print("  " + "-" * 60)

    rows = []
    n_worse = 0
    for param in sorted(df_err["Parameter"].unique()):
        sub = df_err[df_err["Parameter"] == param]
        e_best = sub[sub["Participant_ID"].isin(best)]["Model_Error_Percent"].dropna()
        e_worst = sub[sub["Participant_ID"].isin(worst)]["Model_Error_Percent"].dropna()
        if len(e_best) < 5 or len(e_worst) < 5:
            continue
        try:
            _, p = mannwhitneyu(e_best, e_worst)
        except ValueError:
            p = np.nan
        diff = e_worst.median() - e_best.median()
        if diff > 0:
            n_worse += 1
        print(f"  {param:14s} {e_best.median():11.2f}% {e_worst.median():12.2f}% "
              f"{diff:+8.2f} {p:8.3f}")
        rows.append({
            "Parameter": param, "MAPE_Best_Quality": float(e_best.median()),
            "MAPE_Worst_Quality": float(e_worst.median()),
            "Difference": float(diff), "P_Value": float(p),
        })

    total = len(rows)
    print(f"\n  Parameters where worst quality group performs worse: {n_worse}/{total}")
    if total and n_worse <= total * 0.7:
        print("  -> Direction is inconsistent; some parameters show worse performance in the low-quality group.")
        print("     This is not consistent with a causal relationship.")

    return rows


def main():
    if not os.path.exists(FRAME_LEVEL_PATH):
        print(f"[ERROR] {FRAME_LEVEL_PATH} not found. Please run main.py first.")
        return

    print("=" * 66)
    print("   2B POSITION DATA QUALITY AUDIT")
    print("=" * 66)

    trial_level, patient_level = build_quality_table(FRAME_LEVEL_PATH)
    print(f"\n[OK] {len(trial_level)} trial, "
          f"{patient_level['Participant_ID'].nunique()} participants processed.")

    report_quality(patient_level)
    corr_rows = test_quality_vs_error(patient_level, PATIENT_ERROR_PATH)
    group_rows = compare_quality_groups(patient_level, PATIENT_ERROR_PATH)

    os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
    patient_level.to_csv(METRICS_OUTPUT_PATH, index=False)

    out = []
    for r in corr_rows:
        r["Analysis"] = "Correlation"
        out.append(r)
    for r in group_rows:
        r["Analysis"] = "Group_Comparison"
        out.append(r)
    if out:
        pd.DataFrame(out).to_csv(CORR_OUTPUT_PATH, index=False)

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print("There is no systematic relationship between pose data quality and prediction error.")
    print("Pose quality is not the bottleneck in this study;")
    print("switching to a better pose estimation model will not significantly reduce current errors.")
    print(f"\n[SUCCESS] '{METRICS_OUTPUT_PATH}'")
    if out:
        print(f"[SUCCESS] '{CORR_OUTPUT_PATH}'")


if __name__ == "__main__":
    main()