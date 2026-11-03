"""
STEP 3 - BUILDING THE MODEL DATASET
====================================================================
It translates the experimental measurements into a single table that the model can use.

FLOW

1. Trials of the same patient under the same conditions are AVERAGED
(including trials with jackets -> measurement noise is reduced)
2. Table is flipped: ONE ROW = ONE PATIENT
This makes data leakage STRUCTURALLY impossible in the training/test split
3. Derived step length is calculated
4. Sensor measurements (correct answer) and demographics are added
5. Sensor support phase reading is marked with physiological limits

outputs: data/outputs/dogrulama/Kontrol5_Model_Veri_Seti.csv  
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import pandas as pd

import config
from helper import *


def add_derived_step_length(patient_table):
    """
    DERIVED STEP LENGTH
        step length = speed / step frequency
        DerivedStep = DisplacementSpeed / (SpectralCadence / 60) x 100   [cm]
    """
    for condition in config.WALKING_CONDITIONS:
        speed = patient_table[f"DisplacementSpeed_{condition}"]
        steps_per_minute = patient_table[f"SpectralCadence_{condition}"]
        patient_table[f"DerivedStep_{condition}"] = (
            speed / (steps_per_minute / 60.0) * 100.0)
    return patient_table


def add_sensor_targets_and_demographics(patient_table):
    """
    It includes sensor measurements and height/gender information (shared patients).
    """
    sensor_table = read_table(config.CHECK_2_CLEAN_SENSOR)
    patient_info = pd.read_csv(config.PARTICIPANT_INFO)

    dropped = sorted(set(patient_table["Subject_ID"]) - set(sensor_table["ID"]))
    if dropped:
        print(f"  [INFO] {len(dropped)} participants removed because they did not pass "
              f"sensor quality control.")

    merged = patient_table.merge(
        sensor_table[["ID"] + config.ALL_TARGETS],
        left_on="Subject_ID", right_on="ID").drop(columns=["ID"])
    merged = merged.merge(
        patient_info[["ID"] + config.DEMOGRAPHIC_FEATURES],
        left_on="Subject_ID", right_on="ID").drop(columns=["ID"])
    return merged


def flag_physiological_range(patient_table):
    """
    The sensor checks whether the support phase reading is 
    within physiological limits.
    """
    
    for condition in ["UGS", "FGS"]:
        mono_ok = patient_table[f"MonoSP_{condition}"].between(
            *config.MONO_SUPPORT_RANGE_SECONDS)
        bilateral_ok = patient_table[f"BiSP_{condition}"].between(
            *config.BILATERAL_SUPPORT_RANGE_SECONDS)
        column = f"Within_Physiological_Limits_{condition}"
        patient_table[column] = (mono_ok & bilateral_ok).astype(int)

        valid_count = int(patient_table[column].sum())
        print(f"  [CHECK] {condition}: {valid_count}/{len(patient_table)} participants "
              f"within physiological limits "
              f"({len(patient_table) - valid_count} flagged, not removed).")
    return patient_table


def report_dataset(patient_table, feature_columns):
    print("\n--- DATASET CHECKS ---")
    print(f"  Participants          : {len(patient_table)}")
    print(f"  Model features        : {len(feature_columns)}")
    print(f"  Targets               : {len(config.MAIN_TARGETS)} main + "
          f"{len(config.SUPPORT_PHASE_TARGETS)} support phase")

    for label, columns in [("model features", feature_columns),
                           ("targets", config.ALL_TARGETS)]:
        missing = patient_table[columns].isna().sum()
        if missing.sum():
            print(f"  [WARNING] Missing values found in {label}:")
            print(missing[missing > 0].to_string())
        else:
            print(f"  [OK] No missing values in {label}.")


def main():
    config.create_folders()
    if not files_ready([config.CHECK_4_TRIAL_MEASURES,
                        config.CHECK_2_CLEAN_SENSOR, config.PARTICIPANT_INFO]):
        print("        Run main.py first.")
        return

    title("STEP 3 - BUILDING THE MODEL DATASET")

    trials = read_table(config.CHECK_4_TRIAL_MEASURES)

    # The trials with jackets are separated: only the specified measurements are included in the mean.
    jacket_trials = None
    if "Jacket_Trial" in trials.columns:
        jacket_rows = trials[trials["Jacket_Trial"] == 1].copy()
        trials = trials[trials["Jacket_Trial"] == 0]
        if not jacket_rows.empty:
            jacket_rows["Condition"] = jacket_rows["Video_ID"].str.split("_").str[0]
            keep = ["Subject_ID", "Condition"] + config.MEASUREMENTS_AVERAGED_WITH_JACKET_TRIALS
            jacket_trials = jacket_rows[[c for c in keep if c in jacket_rows.columns]]
            print(f"  [INFO] {len(jacket_trials)} jacket trials included in the average "
                  f"({jacket_trials['Subject_ID'].nunique()} participants).")
    if jacket_trials is None:
        print("  [INFO] No jacket trials found - using 2 trials only.")

    measurements = config.VIDEO_MEASUREMENTS + config.SUPPORT_PHASE_MEASUREMENTS
    condition_level = average_trials(trials, measurements, jacket_trials)
    patient_table = one_row_per_patient(condition_level)

    patient_table = add_derived_step_length(patient_table)
    patient_table = add_sensor_targets_and_demographics(patient_table)
    patient_table = flag_physiological_range(patient_table)

    feature_columns = config.model_feature_columns()
    quality_flags = ["Within_Physiological_Limits_UGS",
                     "Within_Physiological_Limits_FGS"]
    final_columns = (["Subject_ID"] + feature_columns
                     + config.support_phase_columns()
                     + config.ALL_TARGETS + quality_flags)

    dataset = patient_table[final_columns].copy()
    report_dataset(dataset, feature_columns)
    write_table(dataset, config.CHECK_5_MODEL_INPUT)

    print("\n[DONE] Next step: python ML/train_models.py")


if __name__ == "__main__":
    main()
