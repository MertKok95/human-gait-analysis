"""
extract_upper_body_features.py

ADDITIONAL STEP: Extraction of Upper Body and Arm Swing Characteristics

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.dont_write_bytecode = True

import config

# Archive script outputs should be in a separate folder to prevent them 
# from mixing with the main tables.
ARCHIVE_OUT = config.OUTPUT_FOLDER / "archive"
ARCHIVE_OUT.mkdir(parents=True, exist_ok=True)


import os

import numpy as np
import pandas as pd

SUBJECT_FEATURES_PATH = str(config.CHECK_4_TRIAL_MEASURES)
OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_UpperBody_Features.csv")

POSE_TYPES = ["ugs", "fgs"]
WOJ_TRIALS = ["WoJ_1", "WoJ_2"]
FPS = 30.0


def get_surviving_patient_ids(path):
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. main.py must be run first.")
        return []
    return sorted(pd.read_csv(path)["Participant_ID"].unique())


def _angle(a, b, c):
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    denom[denom == 0] = np.nan
    cos = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _amplitude(signal):
    """
    Oscillation amplitude: 5%-95% percentile range (outlier tolerant to outliers)."""
    s = pd.Series(signal).dropna()
    if len(s) < 10:
        return np.nan
    return float(np.percentile(s, 95) - np.percentile(s, 5))


def extract_from_trial(csv_path):
    """Extract upper body features from a single cleaned pose CSV."""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    need = ["l_shoulder_x", "r_shoulder_x", "l_wrist_x", "r_wrist_x",
            "l_hip_x", "r_hip_x", "l_ankle_y", "r_ankle_y"]
    if not all(c in df.columns for c in need):
        return None
    if len(df) < 20:
        return None

    out = {}

    # --- Reference segment lengths (for individual normalization) ---
    mid_shoulder_y = (df["l_shoulder_y"] + df["r_shoulder_y"]) / 2.0
    mid_hip_y = (df["l_hip_y"] + df["r_hip_y"]) / 2.0
    mid_ankle_y = (df["l_ankle_y"] + df["r_ankle_y"]) / 2.0

    trunk_len = float(np.nanmedian(np.abs(mid_hip_y - mid_shoulder_y)))
    leg_len = float(np.nanmedian(np.abs(mid_ankle_y - mid_hip_y)))

    if not np.isfinite(trunk_len) or trunk_len <= 1:
        return None
    if not np.isfinite(leg_len) or leg_len <= 1:
        return None

    out["TrunkSegmentRatio"] = trunk_len / leg_len

    # --- Arm swing: Wrist-Hip horizontal distance ---
    swing_l = df["l_wrist_x"] - df["l_hip_x"]
    swing_r = df["r_wrist_x"] - df["r_hip_x"]

    amp_l = _amplitude(swing_l)
    amp_r = _amplitude(swing_r)

    # normalized to stem length (dimensionless)
    out["ArmSwingAmp_L"] = amp_l / trunk_len if np.isfinite(amp_l) else np.nan
    out["ArmSwingAmp_R"] = amp_r / trunk_len if np.isfinite(amp_r) else np.nan

    if np.isfinite(amp_l) and np.isfinite(amp_r) and (amp_l + amp_r) > 0:
        out["ArmSwingAsym"] = abs(amp_l - amp_r) / ((amp_l + amp_r) / 2.0) * 100.0
        out["ArmSwingAmp_Mean"] = (out["ArmSwingAmp_L"] + out["ArmSwingAmp_R"]) / 2.0
    else:
        out["ArmSwingAsym"] = np.nan
        out["ArmSwingAmp_Mean"] = np.nan

    # --- arm/leg phase relationship ---
    leg_r = df["r_ankle_x"] - df["r_hip_x"] if "r_ankle_x" in df.columns else None
    if leg_r is not None:
        pair = pd.DataFrame({"arm": swing_l, "leg": leg_r}).dropna()
        if len(pair) > 20 and pair["arm"].std() > 0 and pair["leg"].std() > 0:
            out["ArmLegPhaseCorr"] = float(pair["arm"].corr(pair["leg"]))
        else:
            out["ArmLegPhaseCorr"] = np.nan
    else:
        out["ArmLegPhaseCorr"] = np.nan

    # elbow angle ---
    for side in ["l", "r"]:
        cols = [f"{side}_shoulder_x", f"{side}_shoulder_y",
                f"{side}_elbow_x", f"{side}_elbow_y",
                f"{side}_wrist_x", f"{side}_wrist_y"]
        if not all(c in df.columns for c in cols):
            out[f"ElbowAngle_Mean_{side.upper()}"] = np.nan
            out[f"ElbowROM_{side.upper()}"] = np.nan
            continue

        sh = df[[cols[0], cols[1]]].to_numpy(dtype=float)
        el = df[[cols[2], cols[3]]].to_numpy(dtype=float)
        wr = df[[cols[4], cols[5]]].to_numpy(dtype=float)
        ang = _angle(sh, el, wr)
        valid = ang[np.isfinite(ang)]

        if len(valid) >= 10:
            out[f"ElbowAngle_Mean_{side.upper()}"] = float(np.mean(valid))
            out[f"ElbowROM_{side.upper()}"] = float(
                np.percentile(valid, 95) - np.percentile(valid, 5)
            )
        else:
            out[f"ElbowAngle_Mean_{side.upper()}"] = np.nan
            out[f"ElbowROM_{side.upper()}"] = np.nan

    # --- head forward tilt ---
    if "nose_x" in df.columns:
        mid_shoulder_x = (df["l_shoulder_x"] + df["r_shoulder_x"]) / 2.0
        head_off = (df["nose_x"] - mid_shoulder_x) / trunk_len
        out["HeadForwardOffset"] = float(np.nanmedian(np.abs(head_off)))
        out["HeadOffsetSD"] = float(np.nanstd(head_off))
    else:
        out["HeadForwardOffset"] = np.nan
        out["HeadOffsetSD"] = np.nan

    # --- shoulder-hip vertical ratio ---
    out["ShoulderHipVert"] = trunk_len / leg_len

    # --- upper body oscillation amplitude (shoulder center horizontal oscillation) ---
    mid_shoulder_x = (df["l_shoulder_x"] + df["r_shoulder_x"]) / 2.0
    mid_hip_x = (df["l_hip_x"] + df["r_hip_x"]) / 2.0
    rel = (mid_shoulder_x - mid_hip_x) / trunk_len
    out["ShoulderLeadAmp"] = _amplitude(rel)
    out["ShoulderLeadMean"] = float(np.nanmedian(rel))

    return out


def main():
    print("=" * 63)
    print("   ARTICLE 7 - ADDITIONAL STEP: UPPER BODY / ARM SWING CHARACTERISTICS")
    print("=" * 63)

    patient_ids = get_surviving_patient_ids(SUBJECT_FEATURES_PATH)
    if not patient_ids:
        return

    print(f"Processing {len(patient_ids)} patients...\n")

    records = []
    failed = 0

    for i, patient_id in enumerate(patient_ids, 1):
        for pose_type in POSE_TYPES:
            for trial in WOJ_TRIALS:
                csv_path = str(config.clean_pose_file(patient_id, pose_type, trial))
                feats = extract_from_trial(csv_path)
                if feats is None:
                    failed += 1
                    continue
                feats.update({"Participant_ID": patient_id, "Condition": pose_type})
                records.append(feats)

        if i % 25 == 0:
            print(f"  ... {i}/{len(patient_ids)} patients processed "
                  f"({len(records)} trials processed)")

    if not records:
        print("\n[WARNING] No trials processed. Check the cleaned pose CSV paths: "
              "data/outputs/clean_pose/{ugs,fgs}/*_clean_pose.csv")
        return

    df_long = pd.DataFrame(records)

    # downsample to patient x condition level (mean of WoJ_1 and WoJ_2)
    feature_cols = [c for c in df_long.columns
                    if c not in ("Participant_ID", "Condition")]
    cond_level = (
        df_long.groupby(["Participant_ID", "Condition"])[feature_cols]
        .mean()
        .reset_index()
    )

    # geniş formata çevir (Cadence_ugs / Cadence_fgs ile aynı düzen)
    wide = cond_level.pivot(index="Participant_ID", columns="Condition")
    wide.columns = [f"{col}_{cond}" for col, cond in wide.columns]
    wide = wide.reset_index()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    wide.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[SUCCESS] {len(df_long)} trials processed, "
          f"{failed} trials skipped (file not found or insufficient data).")
    print(f"  Number of patients : {wide['Participant_ID'].nunique()}")
    print(f"  Number of features: {wide.shape[1] - 1}")
    print(f"  Saved to   : '{OUTPUT_PATH}'")
    print("\nNext step: train_classification_madde7.py "
          "(this file exists, it will be used automatically)")


if __name__ == "__main__":
    main()