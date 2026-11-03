"""
STEP 4 - MODEL TRAINING
============================================
It trains a regression model for 12 objectives and produces an honest measure of success.

3 DIFFERENT MEASUREMENT METHODS

1. MEMORIZATION CONTROL (80% training / 20% testing)

2. CROSS-VALIDATION (5-fold)

3. STABILITY (20 different splits)
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict,
                                     train_test_split)

import config
from helper import *

warnings.filterwarnings("ignore")

MAIN_MODEL_NAME = "Ridge"
REFERENCE_MODEL_NAME = "Median_Reference"


def build_random_forest():
    """
    Comparative model.
    """
    forest = TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=400,
                                        random_state=config.RANDOM_SEED),
        func=np.log, inverse_func=np.exp)
    return GridSearchCV(
        forest,
        {"regressor__max_depth": [3, 5],
         "regressor__min_samples_leaf": [3, 6],
         "regressor__max_features": ["sqrt", 0.6]},
        cv=KFold(config.CROSS_VALIDATION_FOLDS, shuffle=True,
                 random_state=config.RANDOM_SEED),
        scoring="neg_mean_absolute_error", n_jobs=-1)


def check_overfitting(features, target, target_name):
    """
    80% training / 20% testing. Measures the difference 
    in memorization between Ridge and Random Forest.
    """
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=config.HOLDOUT_TEST_RATIO,
        random_state=config.RANDOM_SEED)

    rows = []
    candidates = {MAIN_MODEL_NAME: build_model(features.shape[1]),
                  "RandomForest": build_random_forest()}

    for model_name, search in candidates.items():
        search.fit(train_features, train_target)
        best_model = search.best_estimator_

        train_score = r2_score(train_target, best_model.predict(train_features))
        test_score = r2_score(test_target, best_model.predict(test_features))

        rows.append({
            "Target": config.readable_name(target_name), "Model": model_name,
            "Train_R2": round(train_score, 3), "Test_R2": round(test_score, 3),
            "Overfitting_Gap": round(train_score - test_score, 3)})

    return rows


def cross_validate(features, target, patient_ids, target_name,
                   track="All_Participants", add_reference=False):
    """5-fold cross-validation + stability measurement in 20 splits."""
    mean_score, score_deviation, all_scores, model = fit_and_score(features, target)

    predictions = cross_val_predict(
        model, features, target,
        cv=KFold(config.CROSS_VALIDATION_FOLDS, shuffle=True,
                 random_state=config.RANDOM_SEED),
        n_jobs=-1)

    rows = [{"Subject_ID": patient, "Target": target_name, "Model": MAIN_MODEL_NAME,
             "Track": track, "Sensor_Value": actual, "Model_Prediction": predicted}
            for patient, actual, predicted in zip(patient_ids, target, predictions)]

    if add_reference:
        median_predictions = np.empty_like(target, dtype=float)
        splitter = KFold(config.CROSS_VALIDATION_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
        for train_index, test_index in splitter.split(features):
            median_predictions[test_index] = np.median(target[train_index])
        rows += [{"Subject_ID": patient, "Target": target_name,
                  "Model": REFERENCE_MODEL_NAME, "Track": track,
                  "Sensor_Value": actual, "Model_Prediction": predicted}
                 for patient, actual, predicted
                 in zip(patient_ids, target, median_predictions)]

    stability = {
        "Target": config.readable_name(target_name), "Target_Code": target_name,
        "Track": track, "Split_Count": config.STABILITY_SPLIT_COUNT,
        "R2_Mean": round(mean_score, 3),
        "R2_Deviation": round(score_deviation, 3),
        "R2_Lowest": round(min(all_scores), 3),
        "R2_Highest": round(max(all_scores), 3)}

    print(f"    {config.readable_name(target_name)[:38]:40s} {track:18s} "
          f"R2 = {mean_score:+.3f} ± {score_deviation:.3f}")

    return rows, stability


def main():
    config.create_folders()
    if not files_ready([config.CHECK_5_MODEL_INPUT]):
        print("        Run ML/prepare_ml_dataset.py first.")
        return

    title("STEP 4 - MODEL TRAINING")

    dataset = read_table(config.CHECK_5_MODEL_INPUT)
    feature_columns = config.model_feature_columns()
    features = dataset[feature_columns].values
    patient_ids = dataset["Subject_ID"].values

    print(f"Participants: {len(dataset)} | Features: {len(feature_columns)}\n")

    # For cross-validation, at least a few participants should remain on each floor.
    minimum_patients = config.CROSS_VALIDATION_FOLDS * 3
    if len(dataset) < minimum_patients:
        print(f"[ERROR] {config.CROSS_VALIDATION_FOLDS}-fold cross-validation requires "
              f"at least {minimum_patients} participants, but only {len(dataset)} are available.")
        print("       Too many participants may have been excluded earlier; "
              "see Check1_Excluded_Participants.csv.")
        return

    predictions, overfitting_rows, stability_rows = [], [], []

    print("--- 8 MAIN TARGETS (compared against the baseline camera method) ---")
    for target_name in config.MAIN_TARGETS:
        target = dataset[target_name].values
        overfitting_rows += check_overfitting(features, target, target_name)
        rows, stability = cross_validate(features, target, patient_ids, target_name)
        predictions += rows
        stability_rows.append(stability)

        search = build_model(len(feature_columns))
        search.fit(features, target)
        joblib.dump(search.best_estimator_,
                    config.MODEL_FOLDER / f"{target_name}.joblib")

    print("\n--- 4 SUPPORT-PHASE TARGETS (not produced by the baseline) ---")
    for target_name in config.SUPPORT_PHASE_TARGETS:
        condition = target_name.split("_")[1]
        physiological_column = f"Within_Physiological_Limits_{condition}"

        tracks = [("All_Participants", dataset),
                  ("Physiologically_Valid", dataset[dataset[physiological_column] == 1])]

        for track_name, subset in tracks:
            rows, stability = cross_validate(
                subset[feature_columns].values, subset[target_name].values,
                subset["Subject_ID"].values, target_name, track_name,
                add_reference=True)
            predictions += rows
            stability_rows.append(stability)

        search = build_model(len(feature_columns))
        search.fit(features, dataset[target_name].values)
        joblib.dump(search.best_estimator_,
                    config.MODEL_FOLDER / f"{target_name}.joblib")

    write_table(pd.DataFrame(predictions), config.CHECK_6_PREDICTIONS)
    write_table(pd.DataFrame(stability_rows), config.CHECK_7_STABILITY)
    write_table(pd.DataFrame(overfitting_rows), config.CHECK_8_OVERFITTING)

    print("\n  IMPORTANT: R2 values in the thesis must always come from Check7_Model_Stability.csv")
    print("  using the 'R2_Mean +/- R2_SD' columns.")
    print("\n[DONE] Next step: python ML/evaluate_results.py")


if __name__ == "__main__":
    main()
