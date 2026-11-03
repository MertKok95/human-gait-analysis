"""
validate_classification_confounders.py

ADDITIONAL ANALYSIS: Mixer Decomposition in Gender/Age Classification
=================================================================================
This script decomposes the source of the high AUC obtained after adding upper body features (extract_upper_body_features.py): it checks whether it is a real movement
(gait dynamics) signal or body SIZE/RATIO
information leaked from the video.

-------------------------------------------------------------------------
FOUR TESTS

TEST 1 - Final Model Performance
UB (upper body) + GAIT (gait summary) features, SelectKBest (k=15) +
LogisticRegression with 10 different cross-validation seeds.

TEST 2 - Permutation Test
Labels are randomly shuffled 200 times and the same procedure is repeated.
The p-value is calculated based on the observed AUC's distribution by chance.

TEST 3 - HEIGHT LEAKAGE TEST (determinative test)
For each feature, the linear relationship between that feature and height is extracted using regression.
How much the AUC decreases indicates how much of the signal is height-related.

TEST 4 - Feature Significance and Height Correlation
The discrimination score (F-statistic) of each selected feature and its raw correlation with height and gender are reported side-by-side.

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

DATASET_PATH = str(config.CHECK_5_MODEL_INPUT)
PATIENT_MEASURES_PATH = str(config.PARTICIPANT_INFO)
UPPER_BODY_PATH = str(ARCHIVE_OUT / "Archive_UpperBody_Features.csv")
OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_Confounder_Analysis.csv")

SEEDS = list(range(10))
N_PERMUTATIONS = 200
K_FEATURES = 15
AGE_THRESHOLD = 45


def are_files_exist(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[HATA] Eksik dosya(lar): {missing}")
        return False
    return True


def build_gait_feature_list(df):
    base = [
        "Cadence_ugs", "Cadence_fgs", "MeanSpeed_ugs", "MeanSpeed_fgs",
        "DisplacementSpeed_ugs", "DisplacementSpeed_fgs",
        "SpectralCadence_ugs", "SpectralCadence_fgs",
        "StepLength_Mean_ugs", "StepLength_Mean_fgs",
        "StrideLength_Mean_ugs", "StrideLength_Mean_fgs",
        "StrideDuration_Mean_ugs", "StrideDuration_Mean_fgs",
        "KneeROM_Mean_ugs", "KneeROM_Mean_fgs",
        "StanceTime_Mean_ugs", "StanceTime_Mean_fgs",
        "SwingTime_Mean_ugs", "SwingTime_Mean_fgs",
        "DoubleSupportTime_Mean_ugs", "DoubleSupportTime_Mean_fgs",
        "SingleSupportTime_Mean_ugs", "SingleSupportTime_Mean_fgs",
    ]
    phase = [c for c in df.columns if "Percent_Mean" in c]
    return [c for c in base + phase if c in df.columns]


def build_pipeline(k):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", LogisticRegression(max_iter=5000)),
    ])


# ---------------------------------------------------------
# 2. TESTS
# ---------------------------------------------------------

def test_final_performance(X, y, k=K_FEATURES, seeds=SEEDS):
    print("\n--- TEST 1: Final model performance ---")
    aucs, baccs = [], []
    for seed in seeds:
        pipe = build_pipeline(min(k, X.shape[1]))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, proba))
        baccs.append(balanced_accuracy_score(y, (proba >= 0.5).astype(int)))
    auc_mean, auc_sd = float(np.mean(aucs)), float(np.std(aucs))
    bacc_mean = float(np.mean(baccs))
    print(f"  AUC = {auc_mean:.3f} ± {auc_sd:.3f}  |  "
          f"Balanced Accuracy = {bacc_mean:.3f}  ({len(seeds)} seeds)")
    return auc_mean, auc_sd, bacc_mean


def test_permutation(X, y, observed_auc, k=K_FEATURES, n_perm=N_PERMUTATIONS):
    print(f"\n--- TEST 2: Permutation Test ({n_perm} iterations) ---")
    rng = np.random.default_rng(0)
    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        pipe = build_pipeline(min(k, X.shape[1]))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        proba = cross_val_predict(pipe, X, y_perm, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        null_aucs.append(roc_auc_score(y_perm, proba))
    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_perm + 1)
    print(f"  Chance (null) distribution: {null_aucs.mean():.3f} ± {null_aucs.std():.3f}")
    print(f"  Observed AUC = {observed_auc:.3f}  ->  p = {p_value:.4f}")
    return p_value, float(null_aucs.mean()), float(null_aucs.std())


def test_height_leakage(feature_sets, y, height, k=K_FEATURES, seeds=range(5)):
    """
    DETERMINISTIC TEST: after removing the height effect with regression
    (within each fold, no leakage) it examines how much the AUC decreases.
    """
    print("\n--- TEST 3: Height Leakage Test (Deterministic) ---")
    print(f"  {'Feature Set':22s} {'Raw AUC':>9s} {'Height-Adjusted AUC':>22s} "
          f"{'BalAcc(arınd.)':>15s}")

    rows = []
    ref_auc = roc_auc_score(y, height)
    print(f"\n  Reference AUC: just height alone = {ref_auc:.3f}\n")

    for name, X_df in feature_sets.items():
        X = X_df.values
        H = height.reshape(-1, 1)

        raw_aucs = []
        for seed in seeds:
            pipe = build_pipeline(min(k, X.shape[1]))
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            raw_aucs.append(roc_auc_score(y, proba))
        raw_auc = float(np.mean(raw_aucs))

        resid_aucs, resid_baccs = [], []
        for seed in seeds:
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            proba = np.zeros(len(y))
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
                col_median = np.nanmedian(X_train, axis=0)
                X_train = np.where(np.isnan(X_train), col_median, X_train)
                X_test = np.where(np.isnan(X_test), col_median, X_test)

                height_model = LinearRegression().fit(H[train_idx], X_train)
                X_train_resid = X_train - height_model.predict(H[train_idx])
                X_test_resid = X_test - height_model.predict(H[test_idx])

                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(f_classif, k=min(k, X.shape[1]))),
                    ("model", LogisticRegression(max_iter=5000)),
                ])
                clf.fit(X_train_resid, y[train_idx])
                proba[test_idx] = clf.predict_proba(X_test_resid)[:, 1]

            resid_aucs.append(roc_auc_score(y, proba))
            resid_baccs.append(balanced_accuracy_score(y, (proba >= 0.5).astype(int)))

        resid_auc = float(np.mean(resid_aucs))
        resid_bacc = float(np.mean(resid_baccs))

        print(f"  {name:22s} {raw_auc:9.3f} {resid_auc:22.3f} {resid_bacc:15.3f}")

        rows.append({
            "Feature_Set": name, "Raw_AUC": raw_auc,
            "Height_Adjusted_AUC": resid_auc,
            "Height_Adjusted_BalAcc": resid_bacc,
            "AUC_Drop": raw_auc - resid_auc,
            "Reference_Height_Only_AUC": ref_auc,
        })
        
    return rows


def test_feature_correlations(feature_sets, y, height, k=K_FEATURES):
    print("\n--- TEST 4: Gender/height correlation of selected traits ---")
    rows = []
    for name, X_df in feature_sets.items():
        selector = SelectKBest(f_classif, k=min(k, X_df.shape[1]))
        X_imputed = X_df.fillna(X_df.median())
        selector.fit(X_imputed, y)
        mask = selector.get_support()
        scores = selector.scores_

        print(f"\n  {name} — selected {mask.sum()} features:")
        selected = [(c, scores[i]) for i, c in enumerate(X_df.columns) if mask[i]]
        for col, score in sorted(selected, key=lambda t: -t[1]):
            r_sex = float(pd.Series(X_imputed[col]).corr(pd.Series(y)))
            r_height = float(pd.Series(X_imputed[col]).corr(pd.Series(height)))
            print(f"    {col:32s} F={score:7.1f}  r(gender)={r_sex:+.3f}  "
                  f"r(height)={r_height:+.3f}")
            rows.append({
                "Feature_Set": name, "Feature": col, "F_Score": float(score),
                "Corr_With_Sex": r_sex, "Corr_With_Height": r_height,
            })
    return rows


# ---------------------------------------------------------
# 3. MAIN STREAM
# ---------------------------------------------------------

def main():
    if not are_files_exist([DATASET_PATH, PATIENT_MEASURES_PATH, UPPER_BODY_PATH]):
        print("[WARNING] UpperBody_Features.csv is required. First, "
                    "extract_upper_body_features.py must be run.")
        return

    print("=" * 70)
    print("   POINT 7 - MIXER (SIZE) SEPARATION ANALYSIS")
    print("=" * 70)

    df_gait = pd.read_csv(DATASET_PATH).drop(columns=["Height"], errors="ignore")
    df_measures = pd.read_csv(PATIENT_MEASURES_PATH).rename(columns={"Sex": "Sex_target"})
    df_upper = pd.read_csv(UPPER_BODY_PATH)

    df_upper = df_upper[[c for c in df_upper.columns if "ShoulderHipVert" not in c]]

    df = (
        df_gait
        .merge(df_measures[["ID", "Age", "Sex_target", "Height"]],
               left_on="Participant_ID", right_on="ID")
        .drop(columns=["ID"])
        .merge(df_upper, on="Participant_ID")
    )

    gait_features = build_gait_feature_list(df)
    upper_features = [c for c in df_upper.columns if c != "Participant_ID"]

    print(f"\n[OK] {len(df)} patients | {len(gait_features)} gait features | "
          f"{len(upper_features)} upper body features")

    y = df["Sex_target"].values
    height = df["Height"].values

    feature_sets = {
        "UB_Plus_Gait": df[upper_features + gait_features],
        "UB_Only": df[upper_features],
        "Gait_Only": df[gait_features],
    }

    all_rows = []

    # Ana model UB_Plus_Gait üzerinden
    auc, auc_sd, bacc = test_final_performance(df[upper_features + gait_features], y)
    all_rows.append({
        "Analysis": "Final_Performance", "Feature_Set": "UB_Plus_Gait",
        "AUC": auc, "AUC_SD": auc_sd, "Balanced_Accuracy": bacc,
    })

    p_value, null_mean, null_sd = test_permutation(
        df[upper_features + gait_features].values, y, auc
    )
    all_rows.append({
        "Analysis": "Permutation_Test", "Feature_Set": "UB_Plus_Gait",
        "Observed_AUC": auc, "Null_AUC_Mean": null_mean,
        "Null_AUC_SD": null_sd, "P_Value": p_value,
    })

    leakage_rows = test_height_leakage(feature_sets, y, height)
    for r in leakage_rows:
        r["Analysis"] = "Height_Leakage_Test"
    all_rows.extend(leakage_rows)

    corr_rows = test_feature_correlations(feature_sets, y, height)
    for r in corr_rows:
        r["Analysis"] = "Feature_Correlation"
    all_rows.extend(corr_rows)

    print("\n" + "=" * 70)
    print("   SAME ANALYSIS — AGE GROUP (>=45) FOR SUMMARY")
    print("=" * 70)
    y_age = (df["Age"] >= AGE_THRESHOLD).astype(int).values
    auc_age, auc_age_sd, bacc_age = test_final_performance(
        df[upper_features + gait_features], y_age, seeds=range(5)
    )
    all_rows.append({
        "Analysis": "Final_Performance_AgeGroup", "Feature_Set": "UB_Plus_Gait",
        "AUC": auc_age, "AUC_SD": auc_age_sd, "Balanced_Accuracy": bacc_age,
    })
    print(f"\n  (Comparison: gender AUC was {auc:.3f}; upper body "
          f"features do not contribute as much to the age group.)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"  Raw AUC (UB+GAIT)          : {auc:.3f}")
    print(f"  Permutation p-value        : {p_value:.4f}")
    ub_row = next(r for r in leakage_rows if r["Feature_Set"] == "UB_Plus_Gait")
    print(f"  Height-Adjusted AUC        : {ub_row['Height_Adjusted_AUC']:.3f}")
    print(f"  Reference (height alone)  : {ub_row['Reference_Height_Only_AUC']:.3f}")
    print(f"\n[SUCCESS] '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()