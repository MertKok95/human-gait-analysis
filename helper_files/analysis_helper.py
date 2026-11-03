"""
analysis_helper.py 
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


# =====================================================================
# 1. FILE AND SCREENS
# =====================================================================

def files_ready(paths):
    """File check"""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print("[ERROR] The following file(s) were not found:")
        for path in missing:
            print(f"        - {path}")
        return False
    return True


def title(text):
    print("\n" + "=" * 70)
    print(f"   {text}")
    print("=" * 70)


def read_table(path):
    """It handles name changes."""
    table = pd.read_csv(path)
    return table.rename(columns=config.INPUT_COLUMN_NAMES)


def write_table(table, path, note=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    renamed = table.rename(columns=config.OUTPUT_COLUMN_NAMES)
    renamed.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {path.name}  ({len(renamed)} rows) {note}")


def append_to_log(new_rows, path):
    """It adds a log entry for eliminated participants."""
    if not new_rows:
        return
    existing = read_table(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    write_table(combined.drop_duplicates(subset=["Participant_ID"]), path)


# =====================================================================
# 2. ERROR AND STATISTICAL METRICS
# =====================================================================

def errors(actual_values, predicted_values):
    """
    It measures the error between two sets of data.
    """
    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    percent_errors = np.abs((actual - predicted) / actual) * 100
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(np.mean(percent_errors)),
        "MedAPE": float(np.median(percent_errors)),
        "R2": float(r2_score(actual, predicted)),
    }


def paired_test(reference_errors, model_errors):
    """
    The model compares its performance to a reference value.
    It determines whether it is better or worse.
    In addition, it identifies whether this performance is due to chance.

    (Wilcoxon signed-rank) is used. A difference is significant if p < 0.05.
    """
    try:
        return float(wilcoxon(reference_errors, model_errors)[1])
    except ValueError:      # If all differences are zero, the test cannot be performed.
        return np.nan


def correlation(first_values, second_values):
    """Pearson correlation. Returns: (r, p_value, number_of_participants)"""
    valid = pd.DataFrame({"a": np.asarray(first_values),
                          "b": np.asarray(second_values)}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, 0
    r, p_value = pearsonr(valid["a"], valid["b"])
    return float(r), float(p_value), len(valid)


def variation(values):
    """
    Coefficient of variation (%): standard deviation / mean. 
    Indicates how regular the walk is; low = regular.
    """
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2 or series.mean() <= 0:
        return np.nan
    return float(series.std(ddof=1) / series.mean() * 100.0)


def symmetry_index(left_value, right_value):
    """Left-right difference divided by the average (%). perfect symmetry = 0"""
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    average = (left_value + right_value) / 2.0
    return (abs(left_value - right_value) / average * 100.0
            if average > 0 else np.nan)


# =====================================================================
# 3. Mean OF TESTS
# =====================================================================

def average_trials(trial_table, measure_columns, jacket_trials=None):
    """
    Averages trial-level data to participant-level.

    Input: each row = 1 participant x condition x trial  (Video_ID = 'ugs_WoJ_1')
    Output: each row = 1 participant x condition           (trials are averaged)

    If jacket_trials is provided, trials with jackets will also be included 
    in the average; the number of trials will increase from 2 to 4, and measurement
    noise will decrease.
    """
    table = trial_table.copy()
    table["Condition"] = table["Video_ID"].str.split("_").str[0]

    columns = [c for c in measure_columns if c in table.columns]
    combined = table[["Subject_ID", "Condition"] + columns]

    if jacket_trials is not None and not jacket_trials.empty:
        extra = jacket_trials.copy()
        for column in columns:
            if column not in extra.columns:
                extra[column] = np.nan
        combined = pd.concat(
            [combined, extra[["Subject_ID", "Condition"] + columns]],
            ignore_index=True)

    return (combined.groupby(["Subject_ID", "Condition"])[columns]
            .mean().reset_index())


def one_row_per_patient(condition_level_table):
    """
    Converts a patient x condition table to a single row per patient format.
    """
    wide = condition_level_table.pivot(index="Subject_ID", columns="Condition")
    wide.columns = [f"{measure}_{condition}" for measure, condition in wide.columns]
    return wide.reset_index()


# =====================================================================
# 4. MODEL (Single Source)
# =====================================================================

def build_model(feature_count):
    """
    The main model of the project: Ridge regression. It consists of four steps:
    1) SelectKBest: selects the k most informative features (learns k from the data)
    2) StandardScaler: scales all features to the same scale
    3) RidgeCV: penalized linear regression (prevents memorization)
    4) log/exp: targets are positive and right-skewed, learned in the log space

    Feature selection is done inside the pipeline so that in each cross-validation
    fold only the training data is used -> no data leakage.
    """
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=f_regression)),
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    log_wrapped = TransformedTargetRegressor(
        regressor=pipeline, func=np.log, inverse_func=np.exp)

    options = [k for k in config.FEATURE_COUNT_OPTIONS if k <= feature_count]
    return GridSearchCV(
        log_wrapped, {"regressor__select__k": options},
        cv=KFold(config.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=config.RANDOM_SEED),
        scoring="neg_mean_absolute_error", n_jobs=-1)


def fit_and_score(features, target, seed_count=None):
    """
    It builds the model and measures the R2 score over multiple random decays.
    """
    seed_count = seed_count if seed_count is not None else config.STABILITY_SPLIT_COUNT
    search = build_model(features.shape[1])
    search.fit(features, target)
    model = search.best_estimator_

    low, high = np.percentile(target, 0.5), np.percentile(target, 99.5)
    scores = []
    for seed in range(seed_count):
        predictions = cross_val_predict(
            model, features, target,
            cv=KFold(config.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=seed),
            n_jobs=-1)
        scores.append(r2_score(target, np.clip(predictions, low, high)))

    return float(np.mean(scores)), float(np.std(scores)), scores, model
