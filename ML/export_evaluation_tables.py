"""
export_evaluation_tables.py - UC TABLE EVALUATION REPORT

THIS FILE IS NOT PART OF THE PIPELINE.

It has been included separately because the existing Table2/Table3 files do not contain RMSE, mean MAPE, and root mean square error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import config
from helper import read_table, write_table

MAIN_MODEL = "Ridge"
REFERENCE_MODEL = "Median_Reference"
PRIMARY_TRACK = "All_Participants"

PREVIEW_ROWS = 10

METRIC_ORDER = [
    "Baseline MAE", "MAE", "RMSE", "MAPE (%)", "MedAPE (%)",
    "R2 Score", "R2 SD (20 splits)",
    "MAE Improvement (%)", "Model Better Patient (%)",
]

UNIT_OF_METRIC = {
    "Baseline MAE": "target unit", "MAE": "target unit", "RMSE": "target unit",
    "MAPE (%)": "%", "MedAPE (%)": "%",
    "R2 Score": "-", "R2 SD (20 splits)": "-",
    "MAE Improvement (%)": "%", "Model Better Patient (%)": "%",
}


# =====================================================================
# 1. COLLECT THE DATA
# =====================================================================

def collect_target_data(predictions, baseline_table, target_code):
    """
    For a given target, collect the actual values, baseline predictions, and model predictions
    into a single table.
    """
    rows = predictions[(predictions["Target"] == target_code)
                       & (predictions["Model"] == MAIN_MODEL)
                       & (predictions["Track"] == PRIMARY_TRACK)]
    if rows.empty:
        return None

    data = pd.DataFrame({
        "Subject_ID": rows["Subject_ID"].values,
        "Actual": rows["Sensor_Value"].values,
        "Model_Prediction": rows["Model_Prediction"].values,
    })

    # Baseline: Camera method at 8 main targets, median at 4 support phase targets.
    if target_code in baseline_table.columns:
        lookup = baseline_table.set_index("ID")[target_code]
        data["Baseline"] = data["Subject_ID"].map(lookup)
        data.attrs["reference_name"] = "Baseline camera method"
    else:
        reference = predictions[(predictions["Target"] == target_code)
                                & (predictions["Model"] == REFERENCE_MODEL)
                                & (predictions["Track"] == PRIMARY_TRACK)]
        lookup = reference.set_index("Subject_ID")["Model_Prediction"]
        data["Baseline"] = data["Subject_ID"].map(lookup)
        data.attrs["reference_name"] = "Median predictor"

    return data[["Subject_ID", "Actual", "Baseline", "Model_Prediction"]]


# =====================================================================
# 2. BUILD THE TABLES
# =====================================================================

def build_table_1(data):
    """Table 1 - prediction comparison."""
    return data.round(4)


def build_table_2(data):
    """
    Table 2 - patient-level error analysis. Both baseline and model in the same table.
    
      - y       = actual value measured by the sensor
      - y_base  = baseline camera method's prediction
      - y_pred  = our model's prediction
    """
    actual = data["Actual"]
    baseline = data["Baseline"]
    model = data["Model_Prediction"]

    baseline_difference = actual - baseline
    model_difference = actual - model
    baseline_absolute = baseline_difference.abs()
    model_absolute = model_difference.abs()

    # Patient-level improvement: percentage of baseline error eliminated
    improvement = np.where(baseline_absolute > 0,
                           (baseline_absolute - model_absolute) / baseline_absolute * 100,
                           np.nan)

    table = pd.DataFrame({
        "Subject_ID": data["Subject_ID"],
        "Sensor_Value (y)": actual.round(4),
        "Baseline_Prediction (y_base)": baseline.round(4),
        "Model_Prediction (y_pred)": model.round(4),

        "Baseline_Absolute_Error (|y - y_base|)": baseline_absolute.round(4),
        "Model_Absolute_Error (|y - y_pred|)": model_absolute.round(4),

        "Baseline_Squared_Error ((y - y_base)^2)": (baseline_difference ** 2).round(4),
        "Model_Squared_Error ((y - y_pred)^2)": (model_difference ** 2).round(4),

        "Baseline_Percentage_Error (%)": (baseline_absolute / actual * 100).round(2),
        "Model_Percentage_Error (%)": (model_absolute / actual * 100).round(2),

        "Improvement (%)": np.round(improvement, 2),
        "Model_Is_Better": np.where(baseline_absolute.isna(), "No baseline",
                            np.where(model_absolute < baseline_absolute, "Yes", "No")),
    })
    return table


def build_table_3(data, unit, thesis_r2, thesis_r2_sd):
    """
    Table 3 - Summary of overall performance. Seven metrics listed one below the other.
    """
    actual = data["Actual"]
    model = data["Model_Prediction"]
    baseline = data["Baseline"]

    model_error = (actual - model).abs()
    model_mae = model_error.mean()
    model_rmse = np.sqrt(((actual - model) ** 2).mean())
    model_percent_error = model_error / actual * 100

    shared = baseline.notna()
    baseline_error = (actual[shared] - baseline[shared]).abs()

    baseline_mae = round(baseline_error.mean(), 4)
    improvement = (baseline_mae - round(model_error[shared].mean(), 4)) / baseline_mae * 100
    better = (model_error[shared] < baseline_error).mean() * 100

    return pd.DataFrame([
        {"Metric": "Baseline MAE", "Value": round(baseline_mae, 4), "Unit": unit},
        {"Metric": "MAE", "Value": round(model_mae, 4), "Unit": unit},
        {"Metric": "RMSE", "Value": round(model_rmse, 4), "Unit": unit},
        {"Metric": "MAPE (%)", "Value": round(model_percent_error.mean(), 2), "Unit": "%"},
        {"Metric": "MedAPE (%)", "Value": round(model_percent_error.median(), 2), "Unit": "%"},
        {"Metric": "R2 Score", "Value": thesis_r2, "Unit": "-"},
        {"Metric": "R2 SD (20 splits)", "Value": thesis_r2_sd, "Unit": "-"},
        {"Metric": "MAE Improvement (%)", "Value": round(improvement, 2), "Unit": "%"},
        {"Metric": "Model Better Patient (%)", "Value": round(better, 2), "Unit": "%"},
    ])


# =====================================================================
# 3. PRESSING THE CONSOLE
# =====================================================================

def print_target_report(target_code, table_1, table_2, table_3, reference_name):
    label = config.readable_name(target_code)
    line = "=" * 78

    print("\n" + line)
    print(f"  TARGET: {label}   ({target_code})")
    print(f"  Reference method: {reference_name}   |   Participants: {len(table_1)}")
    print(line)

    print(f"\n  TABLE 1 - PREDICTION COMPARISON   (first {PREVIEW_ROWS} of {len(table_1)} rows)")
    print("  " + "-" * 62)
    print(table_1.head(PREVIEW_ROWS).to_string(index=False).replace("\n", "\n  "))

    print(f"\n  TABLE 2 - PER-PARTICIPANT ERROR ANALYSIS   "
          f"(first {PREVIEW_ROWS} of {len(table_2)} rows)")
    print("  " + "-" * 62)
    compact = table_2[["Subject_ID", "Sensor_Value (y)",
                       "Baseline_Absolute_Error (|y - y_base|)",
                       "Model_Absolute_Error (|y - y_pred|)",
                       "Model_Squared_Error ((y - y_pred)^2)",
                       "Model_Percentage_Error (%)", "Model_Is_Better"]]
    compact.columns = ["Subject_ID", "y", "|y-y_base|", "|y-y_pred|",
                       "(y-y_pred)^2", "Error %", "Better?"]
    print(compact.head(PREVIEW_ROWS).to_string(index=False).replace("\n", "\n  "))
    print("  (CSV dosyasinda baseline karesel/yuzdesel hata ve Improvement da vardir)")

    print("\n  TABLE 3 - OVERALL PERFORMANCE SUMMARY")
    print("  " + "-" * 62)
    for _, row in table_3.iterrows():
        unit = "" if row["Unit"] == "-" else f" {row['Unit']}"
        print(f"    {row['Metric']:<28s} {row['Value']:>12.4f}{unit}")


# =====================================================================
# 4. MAIN STREAM
# =====================================================================

def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--list" in sys.argv:
        print("\nAvailable targets:\n")
        for target in config.ALL_TARGETS:
            print(f"  {target:14s} {config.readable_name(target)}")
        print()
        return

    if not config.CHECK_6_PREDICTIONS.exists():
        print(f"[ERROR] {config.CHECK_6_PREDICTIONS} not found.")
        print("        Run ML/train_models.py first.")
        return

    predictions = read_table(config.CHECK_6_PREDICTIONS)
    baseline_table = pd.read_csv(config.BASELINE_ESTIMATES)

    stability = read_table(config.CHECK_7_STABILITY)
    stability = stability[stability["Track"] == PRIMARY_TRACK]
    r2_lookup = stability.set_index("Target_Code")[["R2_Mean", "R2_Deviation"]]

    targets = arguments if arguments else config.ALL_TARGETS
    unknown = [t for t in targets if t not in config.ALL_TARGETS]
    if unknown:
        print(f"[ERROR] Unknown target(s): {unknown}")
        print("        Use --list to see available targets.")
        return

    print("=" * 78)
    print("   THREE-TABLE EVALUATION REPORT")
    print("=" * 78)

    all_table_1, all_table_2, all_table_3 = [], [], []

    for target_code in targets:
        data = collect_target_data(predictions, baseline_table, target_code)
        if data is None:
            print(f"\n[SKIPPED] {target_code}: no predictions found.")
            continue

        unit = config.unit_of(target_code)
        table_1 = build_table_1(data)
        table_2 = build_table_2(data)
        thesis_r2 = float(r2_lookup.loc[target_code, "R2_Mean"])
        thesis_r2_sd = float(r2_lookup.loc[target_code, "R2_Deviation"])
        table_3 = build_table_3(data, unit, thesis_r2, thesis_r2_sd)

        print_target_report(target_code, table_1, table_2, table_3,
                            data.attrs.get("reference_name", "-"))

        label = config.readable_name(target_code)
        for table, store in [(table_1, all_table_1), (table_2, all_table_2),
                             (table_3, all_table_3)]:
            tagged = table.copy()
            tagged.insert(0, "Target", label)
            store.append(tagged)

    if not all_table_1:
        print("\n[WARNING] No target could be evaluated.")
        return

    folder = config.THESIS_TABLES
    write_table(pd.concat(all_table_1, ignore_index=True), folder / "Eval1_Predictions.csv")
    write_table(pd.concat(all_table_2, ignore_index=True), folder / "Eval2_Error_Analysis.csv")

    summary_long = pd.concat(all_table_3, ignore_index=True)
    summary_wide = summary_long.pivot(index="Metric", columns="Target", values="Value")
    summary_wide = summary_wide.reindex(METRIC_ORDER)          # metrik sirasi sabit
    summary_wide = summary_wide[[config.readable_name(t) for t in targets
                                 if config.readable_name(t) in summary_wide.columns]]
    summary_wide.insert(0, "Unit", [UNIT_OF_METRIC.get(m, "") for m in summary_wide.index])
    summary_wide = summary_wide.reset_index()
    write_table(summary_wide, folder / "Eval3_Performance_Summary.csv")

    print("\n" + "=" * 78)
    print("  TABLE 3 - PERFORMANCE SUMMARY (metrics x targets)")
    print("=" * 78)
    print(summary_wide.set_index("Metric").to_string())
    print("\n  NOTE: R2 is read from Check7_Model_Stability.csv (mean of 20 splits),")
    print("  so it matches Table2_Main_Results.csv exactly. All other metrics are")
    print("  computed from the cross-validated predictions in Check6.")
    print("\n[DONE] Three tables written to data/outputs/thesis_tables/")


if __name__ == "__main__":
    main()
