"""
STEP 6 - RELIABILITY AND CEILING ANALYSES
======================================================================

A) METHOD COMPARISON (Table 4)
Compares whether the new measurement methods we developed are truly 
better or worse than the old ones.

Cadence: heel strike based vs frequency analysis based
Velocity: step x cadence vs body displacement

B) MEASUREMENT RELIABILITY AND THEORETICAL CEILING (Table 5)
Each participant walks TWICE under the same conditions. How well the two measurements match each other gives the repeatability of the measurement itself and is COMPLETELY INDEPENDENT OF THE SENSOR.

g1 = reliability of a single trial (correlation between trial 1 and trial 2)
g2 = reliability of the mean of two trials (Spearman-Brown formula)
CEILING = square root(g2) mathematical limit of the observable correlation
RATIO = observed r / CEILING percentage of the ceiling reached

C) INFORMATION CEILING (Table 7)
How successful would the model be if, in addition to the camera, 
the sensor's OWN other measurements were also provided ? 
This is the highest achievable score for that target.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import config
from helper import *

warnings.filterwarnings("ignore")

# (our measurement, corresponding sensor measurement, method label)
OLD_VS_NEW_METHODS = [
    ("Cadence", "Cadence", "PREVIOUS - from heel-strike intervals"),
    ("SpectralCadence", "Cadence", "NEW - from spectral analysis"),
    ("MeanSpeed", "Velocity", "PREVIOUS - step length x cadence"),
    ("DisplacementSpeed", "Velocity", "NEW - from trunk displacement"),
]

# Measures for which to calculate reliability
MEASURES_TO_CHECK = ["DisplacementSpeed", "SpectralCadence", "MeanSpeed",
                        "Cadence", "StepLength_Mean", "StrideLength_Mean",
                        "StrideDuration_Mean", "KneeROM_Mean"]

# Sensor measurements that can be used as clues in the ceiling analysis
SENSOR_CLUES = ["Step", "Stride", "Velocity", "Cadence", "MonoSP", "BiSP"]
TWINS = {"Step": "Stride", "Stride": "Step"}  


# =====================================================================
# A) METHOD COMPARISON
# =====================================================================

def compare_methods(dataset):
    """Correlation of previous and new measurement methods with the sensornu karşılaştırır."""
    print("\n--- A) COMPARISON OF PREVIOUS AND NEW METHODS ---\n")
    rows = []

    for measure, sensor_root, label in OLD_VS_NEW_METHODS:
        for condition_upper, condition_lower in [("UGS", "ugs"), ("FGS", "fgs")]:
            our_column = f"{measure}_{condition_lower}"
            sensor_column = f"{sensor_root}_{condition_upper}"
            if our_column not in dataset.columns or sensor_column not in dataset.columns:
                continue

            correlation_r, p_value, patient_count = correlation(
                dataset[our_column], dataset[sensor_column])
            rows.append({
                "Measurement": config.MEASUREMENT_LABELS.get(measure, measure),
                "Method": "NEW" if "NEW" in label else "PREVIOUS",
                "Description": label,
                "Condition": config.CONDITION_LABELS[condition_upper],
                "Correlation_With_Sensor_r": round(correlation_r, 3),
                "p_value": f"{p_value:.1e}",
                "Participants": patient_count,
            })

    table = pd.DataFrame(rows)

    print(f"  {'Measurement':30s} {'Condition':16s} {'prev r':>8s} {'new r':>8s} {'gain':>9s}")
    print("  " + "-" * 76)
    for old, new, name in [("Cadence", "SpectralCadence", "Kadans"),
                           ("MeanSpeed", "DisplacementSpeed", "Yürüme hızı")]:
        for condition in ["normal yürüyüş", "hızlı yürüyüş"]:
            old_row = table[(table["Description"].str.contains("PREVIOUS"))
                            & (table["Condition"] == condition)
                            & (table["Measurement"] == config.MEASUREMENT_LABELS.get(old, old))]
            new_row = table[(table["Description"].str.contains("NEW"))
                            & (table["Condition"] == condition)
                            & (table["Measurement"] == config.MEASUREMENT_LABELS.get(new, new))]
            if old_row.empty or new_row.empty:
                continue
            r_old = old_row["Correlation_With_Sensor_r"].iloc[0]
            r_new = new_row["Correlation_With_Sensor_r"].iloc[0]
            print(f"  {name:30s} {condition:16s} {r_old:8.3f} {r_new:8.3f} "
                  f"{r_new - r_old:+9.3f}")

    return table


# =====================================================================
# B) MEASUREMENT RELIABILITY AND THEORETICAL CEILING
# =====================================================================

def two_trial_reliability(single_trial_reliability, trial_count=2):
    """
    From the reliability of a single trial, it gives the reliability of 
    the Mean of n trials.   
    """
    single = single_trial_reliability
    if not np.isfinite(single) or single <= -1:
        return np.nan
    return trial_count * single / (1.0 + (trial_count - 1) * single)


def measure_reliability(trials, dataset):
    """
    Calculates the reliability and theoretical ceiling for each measurement.
    """
    print("\n--- B) MEASUREMENT RELIABILITY AND THEORETICAL CEILING ---\n")

    valid_patients = set(dataset["Subject_ID"])
    df = trials[trials["Subject_ID"].isin(valid_patients)].copy()

    # ONLY mandatory (jacketless) trials are used.
    if "Jacket_Trial" in df.columns:
        df = df[df["Jacket_Trial"] == 0]

    df["Condition"] = df["Video_ID"].str.split("_").str[0]
    df["Trial"] = df["Video_ID"].str.split("_", n=1).str[1]
    df = df[df["Trial"].isin(config.REQUIRED_TRIALS)]

    # Which measurement has a direct counterpart on the sensor?
    sensor_match = {"DisplacementSpeed": "Velocity", "MeanSpeed": "Velocity",
                    "SpectralCadence": "Cadence", "Cadence": "Cadence",
                    "StepLength_Mean": "Step", "StrideLength_Mean": "Stride"}

    rows = []
    for measure in MEASURES_TO_CHECK:
        if measure not in df.columns:
            continue

        for condition_lower, condition_upper in [("ugs", "UGS"), ("fgs", "FGS")]:
            paired = (df[df["Condition"] == condition_lower]
                      .pivot_table(index="Subject_ID", columns="Trial", values=measure)
                      .reindex(columns=config.REQUIRED_TRIALS)
                      .dropna())
            if paired.shape[1] < 2 or len(paired) < 3:
                continue

            single_reliability, _, patient_count = correlation(paired.iloc[:, 0], paired.iloc[:, 1])
            mean_reliability = two_trial_reliability(single_reliability)
            ceiling = float(np.sqrt(mean_reliability)) if np.isfinite(mean_reliability) and mean_reliability > 0 else np.nan

            observed_r, percent_of_ceiling = np.nan, np.nan
            sensor_root = sensor_match.get(measure)
            if sensor_root:
                our_column = f"{measure}_{condition_lower}"
                sensor_column = f"{sensor_root}_{condition_upper}"
                if {our_column, sensor_column} <= set(dataset.columns):
                    observed_r, _, _ = correlation(dataset[our_column],
                                                         dataset[sensor_column])
                    if np.isfinite(ceiling) and ceiling > 0:
                        percent_of_ceiling = observed_r / ceiling * 100

            rows.append({
                "Measurement": config.MEASUREMENT_LABELS.get(measure, measure),
                "Condition": config.CONDITION_LABELS[condition_upper],
                "Participants": patient_count,
                "Single_Trial_Reliability": round(single_reliability, 3),
                "Two_Trial_Mean_Reliability": round(mean_reliability, 3),
                "Theoretical_Ceiling": round(ceiling, 3) if np.isfinite(ceiling) else np.nan,
                "Observed_r_With_Sensor": round(observed_r, 3)
                if np.isfinite(observed_r) else np.nan,
                "Percent_Of_Ceiling": round(percent_of_ceiling, 1)
                if np.isfinite(percent_of_ceiling) else np.nan,
            })

    table = pd.DataFrame(rows)

    print(f"  {'Measurement':28s} {'Condition':16s} {'Reliab.':>7s} {'CEILING':>7s} "
          f"{'observed r':>11s} {'RATIO':>8s}")
    print("  " + "-" * 82)
    for _, row in table.iterrows():
        ratio = row["Percent_Of_Ceiling"]
        print(f"  {row['Measurement'][:27]:28s} {row['Condition']:16s} "
              f"{row['Two_Trial_Mean_Reliability']:7.3f} "
              f"{row['Theoretical_Ceiling'] if pd.notna(row['Theoretical_Ceiling']) else float('nan'):7.3f} "
              f"{row['Observed_r_With_Sensor'] if pd.notna(row['Observed_r_With_Sensor']) else float('nan'):11.3f} "
              f"{('%' + format(ratio, '.1f')) if pd.notna(ratio) else '-':>8s}")

    return table


# =====================================================================
# C) INFORMATION CEILING
# =====================================================================

def information_ceiling(dataset):
    """
    It sets the upper limit for predictability of the target.
    """
    print("\n--- C) INFORMATION CEILING (what if the camera were given hints?) ---\n")

    camera_features = [c for c in config.model_feature_columns() if c in dataset.columns]
    rows = []

    print(f"  {'Target':38s} {'Camera R2':>10s} {'Ceiling R2':>10s} {'Ratio':>8s}")
    print("  " + "-" * 70)

    for target in config.ALL_TARGETS:
        if target not in dataset.columns:
            continue

        y = dataset[target].values
        camera_r2, _ = fit_and_score(dataset[camera_features].values, y, seed_count=config.STABILITY_SPLIT_COUNT)[:2]

        with_clues = dataset[camera_features].copy()
        root = target.split("_")[0]
        condition = target.split("_")[1]

        for clue in SENSOR_CLUES:
            clue_column = f"{clue}_{condition}"
            if clue_column == target or TWINS.get(root) == clue:
                continue
            if clue_column in dataset.columns:
                with_clues[f"IPUCU_{clue}"] = dataset[clue_column]

        ceiling_r2, _ = fit_and_score(with_clues.values, y, seed_count=config.STABILITY_SPLIT_COUNT)[:2]
        
        # Oran, tabloda GORUNEN yuvarlanmis degerlerden hesaplanir; boylece
        # okuyucu kalemle kontrol ettiginde ayni sonucu bulur.
        camera_r2 = round(camera_r2, 3)
        ceiling_r2 = round(ceiling_r2, 3)
        ratio = camera_r2 / ceiling_r2 * 100 if ceiling_r2 > 0 else np.nan

        name = config.readable_name(target)
        print(f"  {name[:37]:38s} {camera_r2:10.3f} {ceiling_r2:10.3f} "
              f"{('%' + format(ratio, '.1f')) if np.isfinite(ratio) else '-':>8s}")

        rows.append({
            "Target": name,
            "Camera_Only_R2": camera_r2,
            "R2_With_Sensor_Hints": ceiling_r2,
            "Camera_Percent_Of_Ceiling": round(ratio, 1)
            if np.isfinite(ratio) else np.nan,
        })

    return pd.DataFrame(rows)


# =====================================================================
# MAIN FLOW
# =====================================================================

def main():
    config.create_folders()
    if not files_ready([config.CHECK_5_MODEL_INPUT, config.CHECK_4_TRIAL_MEASURES]):
        return

    title("STEP 8 - RELIABILITY AND CEILING ANALYSES")

    dataset = read_table(config.CHECK_5_MODEL_INPUT)
    trials = read_table(config.CHECK_4_TRIAL_MEASURES)

    write_table(compare_methods(dataset), config.TABLE_4_METHOD_COMPARISON)
    write_table(measure_reliability(trials, dataset), config.TABLE_5_MEASUREMENT_RELIABILITY)
    write_table(information_ceiling(dataset), config.TABLE_7_INFORMATION_CEILING)

    print("\n  REMINDER: 'Theoretical_Ceiling' and 'Percent_Of_Ceiling' are DIFFERENT quantities.")
    print("  The ceiling is a correlation; the percentage is a ratio. Do not confuse them.")
    print("\n[DONE] Next step: python ML/explain_model.py")


if __name__ == "__main__":
    main()
