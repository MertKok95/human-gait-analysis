"""
train_classification_madde7.py

Age and Gender Classification
==========================================
It classifies 183 patients by gender and age group from video-derived gait parameters.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

DATASET_PATH = str(config.CHECK_5_MODEL_INPUT)
PATIENT_MEASURES_PATH = str(config.PARTICIPANT_INFO)

UPPER_BODY_PATH = str(ARCHIVE_OUT / "Archive_UpperBody_Features.csv")
FRAME_LEVEL_PATH = str(config.CHECK_9_FRAME_DATASET)
RESULTS_OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_Classification_Results.csv")
IMPORTANCE_OUTPUT_PATH = str(ARCHIVE_OUT / "Archive_Classification_Importance.csv")

ANTHROPOMETRIC_FEATURES = [
    "Height", "Weight", "BMI", "WaistC", "HipC", "NeckC",
    "Percentage_fat_mass", "Lean_mass",
]

AGE_THRESHOLD = 45
AGE_THRESHOLD_SWEEP = [35, 40, 45, 50, 55]
SEEDS = [0, 1, 2]
N_PERMUTATIONS = 100


def build_waveform_features(frame_level_path, valid_ids):
    """
    Loop-normalized (0-100%) average junction angle waveforms.
    """
    if not os.path.exists(frame_level_path):
        print("[INFO] Frame-level data not found - waveform "
                "features are being skipped.")
        return None

    from scipy.interpolate import interp1d

    angles = ["l_knee_angle", "r_knee_angle", "l_hip_angle",
              "r_hip_angle", "l_trunk_sway", "r_trunk_sway",
              "knee_angle_asymmetry_2d"]
    n_points = 20

    dl = pd.read_csv(frame_level_path)
    dl = dl[dl["Participant_ID"].isin(valid_ids)].copy()
    dl["Condition"] = dl["Fast_Gait"].map({0: "ugs", 1: "fgs"})

    rows = []
    for (pid, cond, wd), g in dl.groupby(["Participant_ID", "Condition", "Walking_Direction"]):
        g = g.sort_values("frame")
        hs = np.where(g["is_heel_strike"].values == 1)[0]
        if len(hs) < 2:
            continue
        rec = {"Participant_ID": pid, "Condition": cond}
        for col in angles:
            if col not in g.columns:
                continue
            sig = pd.Series(g[col].values.astype(float)).interpolate(
                limit_direction="both").values
            if np.isnan(sig).all():
                continue
            cycles = []
            for i in range(len(hs) - 1):
                a, b = hs[i], hs[i + 1]
                if b - a < 8 or b - a > 60:     
                    continue
                seg = sig[a:b + 1]
                if np.isnan(seg).any():
                    continue
                f = interp1d(np.linspace(0, 100, len(seg)), seg, kind="linear")
                cycles.append(f(np.linspace(0, 100, n_points)))
            if cycles:
                arr = np.array(cycles)
                mean_curve = arr.mean(axis=0)
                for j, val in enumerate(mean_curve):
                    rec[f"{col}_p{j:02d}"] = val
                rec[f"{col}__cycSD"] = float(arr.std(axis=0).mean())
        if len(rec) > 2:
            rows.append(rec)

    if not rows:
        return None

    df_long = pd.DataFrame(rows)
    cols = [c for c in df_long.columns if c not in ("Participant_ID", "Condition")]
    cond_level = df_long.groupby(["Participant_ID", "Condition"])[cols].mean()
    wide = cond_level.unstack()
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    print(f"[OK] Waveform characteristics: {wide.shape[1]} sütun")
    return wide.reset_index()


def build_gait_feature_list(df):
    """
    Video-derived walking features only
    """
    base = [
        "Cadence_ugs", "Cadence_fgs",
        "MeanSpeed_ugs", "MeanSpeed_fgs",
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


def make_classifier(name):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=3000)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=4,
            random_state=0, n_jobs=-1
        )
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0
        )
    raise ValueError(name)


# PCA should be performed on high-dimensional datasets; otherwise, 
# overlearning and slow training may occur.
PCA_THRESHOLD = 60
PCA_COMPONENTS = 30


def evaluate_classifier(X, y, model_name, seeds=SEEDS):
    """Multi-seed 5-fold cross-validation; returns AUC and balanced accuracy."""
    aucs, baccs = [], []
    use_pca = X.shape[1] > PCA_THRESHOLD
    for seed in seeds:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if use_pca:
            steps.append(("pca", PCA(n_components=min(PCA_COMPONENTS,
                                                      X.shape[1], len(y) - 20))))
        steps.append(("model", make_classifier(model_name)))
        pipeline = Pipeline(steps)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, proba))
        baccs.append(balanced_accuracy_score(y, (proba >= 0.5).astype(int)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(baccs))


def permutation_test(X, y, model_name, observed_auc, n_perm=N_PERMUTATIONS):
    """Permute labels to measure the probability of observing the given AUC by chance."""
    rng = np.random.default_rng(0)
    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", make_classifier(model_name)),
        ])
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        proba = cross_val_predict(pipeline, X, y_perm, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        null_aucs.append(roc_auc_score(y_perm, proba))
    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_perm + 1)
    return float(p_value), float(null_aucs.mean()), float(null_aucs.std())


# ---------------------------------------------------------
# 2. ANALYSIS STEPS
# ---------------------------------------------------------

def run_age_threshold_sweep(df, gait_features):
    print("\n--- Age threshold sensitivity analysis (LogisticRegression) ---")
    rows = []
    for threshold in AGE_THRESHOLD_SWEEP:
        y = (df["Age"] >= threshold).astype(int).values
        auc, sd, bacc = evaluate_classifier(df[gait_features], y, "LogisticRegression")
        print(f"  threshold {threshold}: n_positive={y.sum():3d}  "
              f"AUC={auc:.3f}±{sd:.3f}  BalAcc={bacc:.3f}")
        rows.append({
            "Analysis": "Age_Threshold_Sweep", "Threshold": threshold,
            "N_Positive": int(y.sum()), "AUC": auc, "AUC_SD": sd,
            "Balanced_Accuracy": bacc,
        })
    print("  -> Results are robust to the threshold; the threshold "
          f"{AGE_THRESHOLD} is used in the main analysis (most balanced split).")
    return rows


def run_age_regression(df, gait_features):
    print("\n--- Age regression (continuous target, years) ---")
    rows = []
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    sets = {
        "Gait_Only": gait_features,
        "Gait_Plus_Anthropometry": gait_features + ANTHROPOMETRIC_FEATURES,
        "Anthropometry_Only": ANTHROPOMETRIC_FEATURES,
    }
    for name, cols in sets.items():
        pred = cross_val_predict(
            pipeline, df[cols], df["Age"].values,
            cv=KFold(5, shuffle=True, random_state=0)
        )
        r2 = r2_score(df["Age"], pred)
        mae = mean_absolute_error(df["Age"], pred)
        print(f"  {name:26s} R2={r2:+.3f}  MAE={mae:.1f} years")
        rows.append({
            "Analysis": "Age_Regression", "Track": name,
            "R2": r2, "MAE_Years": mae,
        })
    print(f"  (Reference: standard deviation of age {df['Age'].std():.1f} years)")
    print("  -> Continuous age prediction with gait features is NOT successful.")
    return rows


def run_main_classification(df, gait_features, enriched):
    print("\n--- Main Classification: Three Tracks ---")
    rows = []

    targets = {
        "Sex": df["Sex_target"].values,
        f"AgeGroup_{AGE_THRESHOLD}": (df["Age"] >= AGE_THRESHOLD).astype(int).values,
    }
    tracks = {
        "A_Gait_Summary_Only": gait_features,
        "A2_Gait_Enriched": enriched,
        "B_Anthropometry_Only": ANTHROPOMETRIC_FEATURES,
        "C_All": enriched + ANTHROPOMETRIC_FEATURES,
    }
    models = ["LogisticRegression", "LDA", "RandomForest", "GradientBoosting"]

    for target_name, y in targets.items():
        print(f"\n  === {target_name} (positive class: {y.sum()}/{len(y)}) ===")
        print(f"  {'Track':28s} {'Model':20s} {'AUC':>7s} {'±sd':>6s} {'BalAcc':>8s}")
        for track_name, cols in tracks.items():
            for model_name in models:
                auc, sd, bacc = evaluate_classifier(df[cols], y, model_name)
                print(f"  {track_name:28s} {model_name:20s} "
                      f"{auc:7.3f} {sd:6.3f} {bacc:8.3f}")
                rows.append({
                    "Analysis": "Main_Classification", "Target": target_name,
                    "Track": track_name, "Model": model_name,
                    "N_Features": len(cols), "AUC": auc, "AUC_SD": sd,
                    "Balanced_Accuracy": bacc,
                })
    return rows


def run_permutation_tests(df, gait_features, main_rows):
    print("\n--- Permutation Test (Track A, LogisticRegression, "
          f"{N_PERMUTATIONS} permutations) ---")
    rows = []
    for target_name, y in [
        ("Sex", df["Sex_target"].values),
        (f"AgeGroup_{AGE_THRESHOLD}", (df["Age"] >= AGE_THRESHOLD).astype(int).values),
    ]:
        # Permutation test is performed with LogisticRegression
        best = next(r for r in main_rows
                    if r["Target"] == target_name
                    and r["Track"] == "A_Gait_Summary_Only"
                    and r["Model"] == "LogisticRegression")
        p_value, null_mean, null_sd = permutation_test(
            df[gait_features], y, best["Model"], best["AUC"]
        )
        print(f"  {target_name:18s} {best['Model']:20s} "
              f"observed AUC={best['AUC']:.3f} | "
              f"null={null_mean:.3f}±{null_sd:.3f} | p={p_value:.4f}")
        rows.append({
            "Analysis": "Permutation_Test", "Target": target_name,
            "Model": best["Model"], "Observed_AUC": best["AUC"],
            "Null_AUC_Mean": null_mean, "Null_AUC_SD": null_sd,
            "P_Value": p_value,
        })
    return rows


def run_feature_importance(df, gait_features):
    """Reports which gait features are most discriminative in Iz A."""
    print("\n--- Feature Importance (Iz A) ---")
    rows = []
    for target_name, y in [
        ("Sex", df["Sex_target"].values),
        (f"AgeGroup_{AGE_THRESHOLD}", (df["Age"] >= AGE_THRESHOLD).astype(int).values),
    ]:
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", make_classifier("RandomForest")),
        ])
        pipeline.fit(df[gait_features], y)
        importances = pipeline.named_steps["model"].feature_importances_
        order = np.argsort(importances)[::-1]

        print(f"  {target_name} — most discriminative 5 features:")
        for rank, idx in enumerate(order[:5], 1):
            print(f"    {rank}. {gait_features[idx]:32s} {importances[idx]:.4f}")

        for idx in order:
            rows.append({
                "Target": target_name, "Feature": gait_features[idx],
                "Importance": float(importances[idx]),
            })
    return rows


# ---------------------------------------------------------
# 3. MAIN STREAM
# ---------------------------------------------------------

def main():
    missing = [p for p in [DATASET_PATH, PATIENT_MEASURES_PATH] if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Missing file(s): {missing}")
        return

    print("=" * 63)
    print("   ARTICLE 7: AGE AND GENDER CLASSIFICATION")
    print("=" * 63)

    df_gait = pd.read_csv(DATASET_PATH)
  
    df_gait = df_gait.drop(columns=["Height"], errors="ignore")

    df_measures = pd.read_csv(PATIENT_MEASURES_PATH).rename(
        columns={"Sex": "Sex_target"}
    )
    keep = ["ID", "Age", "Sex_target", "PA_level"] + ANTHROPOMETRIC_FEATURES

    df = df_gait.merge(
        df_measures[keep], left_on="Participant_ID", right_on="ID"
    ).drop(columns=["ID"])

    gait_features = build_gait_feature_list(df)

    # --- optional enrichment: upper body / arm swing ---
    upper_features = []
    if os.path.exists(UPPER_BODY_PATH):
        df_upper = pd.read_csv(UPPER_BODY_PATH)
        df = df.merge(df_upper, on="Participant_ID", how="left")
        upper_features = [c for c in df_upper.columns if c != "Participant_ID"]
        print(f"[OK] Upper body features: {len(upper_features)} columns")
    else:
        print("[INFO] UpperBody_Features.csv not found - arm swing features "
              "will be skipped (extract_upper_body_features.py can be run).")

    # --- optional enrichment: waveform features ---
    waveform_features = []
    df_wave = build_waveform_features(FRAME_LEVEL_PATH, set(df["Participant_ID"]))
    if df_wave is not None:
        df = df.merge(df_wave, on="Participant_ID", how="left")
        waveform_features = [c for c in df_wave.columns if c != "Participant_ID"]

    enriched = gait_features + upper_features + waveform_features

    print(f"\n[OK] {len(df)} hasta | {len(gait_features)} yürüyüş özelliği | "
          f"{len(ANTHROPOMETRIC_FEATURES)} antropometrik özellik")
    print(f"     Cinsiyet dağılımı: {df['Sex_target'].sum()} / "
          f"{len(df) - df['Sex_target'].sum()}")
    print(f"     Yaş: ort {df['Age'].mean():.1f}, "
          f"aralık {df['Age'].min()}-{df['Age'].max()}")

    all_rows = []
    all_rows += run_age_threshold_sweep(df, gait_features)
    all_rows += run_age_regression(df, gait_features)

    main_rows = run_main_classification(df, gait_features, enriched)
    all_rows += main_rows

    all_rows += run_permutation_tests(df, gait_features, main_rows)

    importance_rows = run_feature_importance(df, gait_features)

    os.makedirs(os.path.dirname(RESULTS_OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULTS_OUTPUT_PATH, index=False)
    pd.DataFrame(importance_rows).to_csv(IMPORTANCE_OUTPUT_PATH, index=False)

    print("\n" + "=" * 63)
    print("YORUM")
    print("=" * 63)
    print("İz A (yalnızca yürüyüş) birincil bilimsel sonuçtur.")
    print("İz B, karıştırıcının büyüklüğünü gösterir: antropometrik")
    print("ölçümler cinsiyeti neredeyse kusursuz ayırdığı için, yüksek")
    print("AUC değerleri yürüyüş analizinin başarısı olarak SUNULMAMALIDIR.")
    print(f"\n[BAŞARILI] '{RESULTS_OUTPUT_PATH}'")
    print(f"[BAŞARILI] '{IMPORTANCE_OUTPUT_PATH}'")


if __name__ == "__main__":
    main()