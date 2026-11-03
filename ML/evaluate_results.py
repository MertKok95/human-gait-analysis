"""
STEP 5 - RESULT EVALUATION
===============================================================
- It generates the MAIN TABLES in Chapter 4 of the thesis.
- Comparison and statistical testing (Wilcoxon) are performed.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import config
from helper import *

MAIN_MODEL_NAME = "Ridge"
REFERENCE_MODEL_NAME = "Median_Reference"
PRIMARY_TRACK = "All_Participants"


# =====================================================================
# TABLO 1 - ÖRNEK ZİNCİRİ
# =====================================================================

def build_sample_chain():
    """
    It shows how the Participants In decrease at each stage.
    """
    start_count = len(pd.read_csv(config.PARTICIPANT_INFO))
    after_sensor = len(read_table(config.CHECK_2_CLEAN_SENSOR))
    final_count = len(read_table(config.CHECK_5_MODEL_INPUT))

    return pd.DataFrame([
        {"Stage": "1. Raw dataset", "Participants": start_count, "Excluded": 0,
         "Description": "All participants in the Health & Gait dataset"},
        {"Stage": "2. Sensor quality control", "Participants": after_sensor,
         "Excluded": start_count - after_sensor,
         "Description": "Implausible speed, contradictory reading, duplicate record, missing value"},
        {"Stage": "3. Pose and kinematic quality control", "Participants": final_count,
         "Excluded": after_sensor - final_count,
         "Description": "Corrupted pose data, no valid gait cycle detected"},
    ])


# =====================================================================
# TABLE 2 - MAIN RESULTS
# =====================================================================

def get_reference_values(predictions, baseline_table, target_name, patient_index):
    """
    It returns the comparison reference for the given target.
    If the baseline produces this parameter, it uses it; otherwise, it uses the median predictor.
    """
    if target_name in baseline_table.columns:
        return ("Baseline camera method",
                baseline_table.reindex(patient_index)[target_name])

    reference_rows = predictions[(predictions["Target"] == target_name)
                                 & (predictions["Model"] == REFERENCE_MODEL_NAME)
                                 & (predictions["Track"] == PRIMARY_TRACK)]
    values = reference_rows.set_index("Subject_ID")["Model_Prediction"]
    return "Median predictor", values.reindex(patient_index)


def describe_result(p_value, improvement_percent):
    """It gives the statistical result."""
    if pd.isna(p_value):
        return "Test not possible"
    if p_value >= 0.05:
        return "Statistically indistinguishable"
    return ("Model significantly better" if improvement_percent > 0
            else "Reference significantly better")


def build_main_results(predictions, baseline_table, stability_table):
    """
    One row for each parameter:
    how much error the reference makes, how much the model makes, is the difference significant?
    """
    baseline_table = baseline_table.set_index("ID")
    rows = []

    for target_name in config.ALL_TARGETS:
        model_rows = predictions[
            (predictions["Target"] == target_name)
            & (predictions["Model"] == MAIN_MODEL_NAME)
            & (predictions["Track"] == PRIMARY_TRACK)].set_index("Subject_ID")
        if model_rows.empty:
            continue

        sensor_values = model_rows["Sensor_Value"]
        model_values = model_rows["Model_Prediction"]
        model_scores = errors(sensor_values, model_values)
        model_errors = (sensor_values - model_values).abs()

        reference_name, reference_values = get_reference_values(
            predictions, baseline_table, target_name, model_rows.index)

        shared_patients = reference_values.dropna().index
        reference_scores = errors(sensor_values[shared_patients],
                                  reference_values[shared_patients])
        reference_errors = (sensor_values[shared_patients]
                            - reference_values[shared_patients]).abs()

        p_value = paired_test(reference_errors.values,
                              model_errors[shared_patients].values)

        reference_mae = round(reference_scores["MAE"], 4)
        model_mae = round(model_scores["MAE"], 4)
        improvement = (reference_mae - model_mae) / reference_mae * 100

        stability = stability_table[
            (stability_table["Target_Code"] == target_name)
            & (stability_table["Track"] == PRIMARY_TRACK)]

        rows.append({
            "Parameter": config.readable_name(target_name),
            "Unit": config.unit_of(target_name),
            "Participants": len(shared_patients),
            "Reference_Method": reference_name,
            "Reference_MAE": reference_mae,
            "Model_MAE": model_mae,
            "Improvement_Percent": round(improvement, 1),
            "Reference_MedAPE_Percent": round(reference_scores["MedAPE"], 1),
            "Model_MedAPE_Percent": round(model_scores["MedAPE"], 1),
            "Model_R2": stability["R2_Mean"].iloc[0] if not stability.empty else np.nan,
            "Model_R2_SD": stability["R2_Deviation"].iloc[0] if not stability.empty else np.nan,
            "p_value": round(p_value, 4) if pd.notna(p_value) else np.nan,
            "Conclusion": describe_result(p_value, improvement),
            "Percent_Participants_Model_Better": round(
                (model_errors[shared_patients] < reference_errors).mean() * 100, 1),
        })

    return pd.DataFrame(rows)


# =====================================================================
# TABLE 3 - PATIENT-BASED COMPARISON
# =====================================================================

def build_per_patient_table(predictions, baseline_table):
    """
    A single line for each participant x parameter: 
    sensor, baseline, model, and errors.
    """
    baseline_table = baseline_table.set_index("ID")
    rows = []

    for target_name in config.ALL_TARGETS:
        model_rows = predictions[
            (predictions["Target"] == target_name)
            & (predictions["Model"] == MAIN_MODEL_NAME)
            & (predictions["Track"] == PRIMARY_TRACK)]
        if model_rows.empty:
            continue

        baseline_available = target_name in baseline_table.columns

        for _, row in model_rows.iterrows():
            patient = row["Subject_ID"]
            sensor_value = row["Sensor_Value"]
            model_value = row["Model_Prediction"]
            baseline_value = (baseline_table.loc[patient, target_name]
                              if baseline_available and patient in baseline_table.index
                              else np.nan)

            model_error = abs(sensor_value - model_value)
            baseline_error = abs(sensor_value - baseline_value)

            rows.append({
                "Subject_ID": patient,
                "Parameter": config.readable_name(target_name),
                "Unit": config.unit_of(target_name),
                "Sensor_Value": round(sensor_value, 4),
                "Baseline_Estimate": (round(baseline_value, 4)
                                            if pd.notna(baseline_value) else np.nan),
                "Model_Prediction": round(model_value, 4),
                "Baseline_Error": (round(baseline_error, 4)
                                    if pd.notna(baseline_error) else np.nan),
                "Model_Error": round(model_error, 4),
                "Model_Is_Better": (("Yes" if model_error < baseline_error else "No")
                                   if pd.notna(baseline_error) else "No baseline"),
            })

    return pd.DataFrame(rows)


def main():
    config.create_folders()
    if not files_ready([config.CHECK_6_PREDICTIONS,
                        config.CHECK_7_STABILITY,
                        config.BASELINE_ESTIMATES, config.CHECK_5_MODEL_INPUT]):
        print("        Run ML/train_models.py first.")
        return

    title("STEP 5 - RESULT EVALUATION")

    predictions = read_table(config.CHECK_6_PREDICTIONS)
    stability_table = read_table(config.CHECK_7_STABILITY)
    baseline_table = pd.read_csv(config.BASELINE_ESTIMATES)

    write_table(build_sample_chain(), config.TABLE_1_SAMPLE_CHAIN)

    main_results = build_main_results(predictions, baseline_table, stability_table)
    write_table(main_results, config.TABLE_2_MAIN_RESULTS)
    write_table(build_per_patient_table(predictions, baseline_table),
                config.TABLE_3_PER_PARTICIPANT)

    print("\n" + "-" * 104)
    print("MAIN RESULTS TABLE")
    print("-" * 104)
    shown = ["Parameter", "Reference_MAE", "Model_MAE",
             "Improvement_Percent", "Model_R2", "p_value", "Conclusion"]
    print(main_results[shown].to_string(index=False))

    print("\n  REMINDER: rows reading 'Statistically indistinguishable' must NOT be "
          "written as 'the model is worse'. p > 0.05 means the difference "
          "could not be demonstrated.")
    print("\n[DONE] Next step: python ML/reliability_analysis.py")


if __name__ == "__main__":
    main()
