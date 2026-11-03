"""
file_guide.py - GUIDE TO OUTPUT FILES
"""

import pandas as pd

import config

OUTPUT_GUIDE = [
    # Thesis outputs: directly included in the thesis
    (config.TABLE_1_SAMPLE_CHAIN, "—",
     "How the sample was reduced from 398 to 183", "Yes - Section 4.1"),
    (config.TABLE_2_MAIN_RESULTS, "Ozet_Karsilastirma_Tablosu.csv",
     "Baseline vs model error with significance test", "Yes - Section 4.6"),
    (config.TABLE_3_PER_PARTICIPANT, "Hasta_Bazli_Karsilastirma_183.csv",
     "Sensor, baseline and model values per participant and parameter", "Yes - Appendix"),
    (config.TABLE_4_METHOD_COMPARISON, "Method_Correlation_Results.csv",
     "Correlation of previous and new measurement methods with the sensor", "Yes - Section 4.3"),
    (config.TABLE_5_MEASUREMENT_RELIABILITY,
     "Measurement_Reliability.csv + Measurement_Ceiling_Analysis.csv",
     "Test-retest reliability and theoretical ceiling", "Yes - Section 4.2"),
    (config.TABLE_6_FEATURE_IMPORTANCE, "SHAP_Feature_Importance.csv",
     "How much the model relies on each measurement", "Yes - Section 4.8"),
    (config.TABLE_7_INFORMATION_CEILING, "Oracle_Ceiling_Results.csv",
     "Predictability ceiling of each target", "Yes - Section 4.7"),

    # --- VALIDATION OUTPUTS: for control purposes ---
    (config.CHECK_1_EXCLUDED, "Corrupted_Patients_Log.csv",
     "Which participants were excluded and why", "No - validation"),
    (config.CHECK_2_CLEAN_SENSOR, "filtered_gait_sensor_data.csv",
     "Sensor data that passed quality control (ground truth)", "No - validation"),
    (config.CHECK_3_GAIT_CYCLES, "ML_Master_Gait_Cycles.csv",
     "One row per gait cycle: duration, phases, step length, QC result", "No - validation"),
    (config.CHECK_4_TRIAL_MEASURES, "ML_Master_Subject_Features.csv",
     "One row per participant x condition x trial: all video measurements", "No - validation"),
    (config.CHECK_5_MODEL_INPUT, "ML_Regression_Dataset.csv",
     "One row per participant: model inputs and targets", "No - validation"),
    (config.CHECK_6_PREDICTIONS, "Model_Predictions_CV.csv",
     "Observed value and cross-validated prediction per participant", "No - validation"),
    (config.CHECK_7_STABILITY, "Model_R2_MultiSeed.csv",
     "Mean and standard deviation of R2 across 20 splits", "No - source of R2"),
    (config.CHECK_8_OVERFITTING, "Chapter4_T45_Model_Selection.csv",
     "Train/test R2 gap - evidence for selecting Ridge", "No - validation"),
    (config.CHECK_9_FRAME_DATASET, "DeepLearning_Input_Dataset.csv",
     "Frame-level measurements (for deep-learning experiments)", "No - optional"),
    (config.CHECK_10_CLEANING_SUMMARY, "cleaning_summary.csv",
     "Row counts before/after cleaning and the rules applied", "No - validation"),
]


def write_guide():
    """Writes the guide to disk. Non-existent files are marked as 'not produced yet'."""
    rows = []
    for path, old_name, description, in_thesis in OUTPUT_GUIDE:
        rows.append({
            "File": path.name,
            "Folder": path.parent.name,
            "Previous_Name": old_name,
            "Contents": description,
            "In_Thesis": in_thesis,
            "Status": "produced" if path.exists() else "not produced yet",
        })

    guide = pd.DataFrame(rows)
    config.FILE_GUIDE.parent.mkdir(parents=True, exist_ok=True)
    guide.to_csv(config.FILE_GUIDE, index=False, encoding="utf-8-sig")
    return guide
