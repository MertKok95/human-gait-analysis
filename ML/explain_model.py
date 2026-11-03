"""
STEP 7 - EXPLAINING THE MODEL'S DECISIONS
============================================================
This is the "SHAP - explainable artificial intelligence" (XAI) section of the thesis.

The question it answers:
- Which measurement does the model look at when making this prediction?
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from helper import *

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings("ignore")

MAX_EVALUATIONS = 201   
TOP_N = 10              # number of features to be shown in the graphs


def explain_one_target(model, features, target_name):
    """
    Calculates SHAP values ​​for a target and extracts the model structure. 
    
    Returns: (importance table, shap object, aggregation validation passed)
    """
    masker = shap.maskers.Independent(features, max_samples=len(features))
    explainer = shap.Explainer(model.predict, masker, algorithm="permutation",
                               seed=config.RANDOM_SEED)
    shap_values = explainer(features, max_evals=MAX_EVALUATIONS, silent=True)

    # Additivity check: estimation of the sum of contributions
    rebuilt = shap_values.base_values + shap_values.values.sum(axis=1)
    additive = bool(np.allclose(rebuilt, model.predict(features.values), atol=1e-6))

    # Which features the model selected and in what direction it used them.
    pipeline = model.regressor_
    selected_mask = pipeline.named_steps["select"].get_support()
    coefficients = np.zeros(len(features.columns))
    coefficients[selected_mask] = pipeline.named_steps["model"].coef_

    importance = pd.DataFrame({
        "Feature": [config.readable_name(c) for c in features.columns],
        "Mean_Contribution": np.abs(shap_values.values).mean(axis=0),
        "Used_By_Model": np.where(selected_mask, "Yes", "No"),
        "Direction": np.where(coefficients > 0, "Increases",
                        np.where(coefficients < 0, "Decreases", "-")),
    }).sort_values("Mean_Contribution", ascending=False)

    total = importance["Mean_Contribution"].sum()
    importance["Importance_Percent"] = (importance["Mean_Contribution"] / total * 100
                                  if total > 0 else 0.0)
    importance.insert(0, "Rank", range(1, len(importance) + 1))
    importance.insert(0, "Target", config.readable_name(target_name))

    return importance, shap_values, additive, int(selected_mask.sum())


def plot_importance(importance, target_name, unit, path):
    """Feature importance plot. Blue = increases the target, red = decreases it."""
    top = importance.head(TOP_N)[::-1]
    if top["Mean_Contribution"].sum() == 0:
        return

    colors = ["#B03A2E" if direction == "Decreases" else "#1F618D"
              for direction in top["Direction"]]

    figure, axis = plt.subplots(figsize=(9, 0.45 * len(top) + 1.6))
    axis.barh(top["Feature"], top["Mean_Contribution"], color=colors)
    axis.set_xlabel(f"Mean contribution ({unit})")
    axis.set_title(f"Feature importance - {config.readable_name(target_name)}\n"
                   f"blue: increases the target  |  red: decreases it",
                   fontsize=11, fontweight="bold")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_distribution(shap_values, target_name, unit, path):
    """
    Beeswarm: each point is a participant. 
    Horizontal position is the contribution to the prediction,
    color is the feature value (red = high, blue = low).
    Shows how the contribution varies from patient to patient.
    """
    
    if np.abs(shap_values.values).sum() == 0:
        return
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=TOP_N, show=False)
    plt.title(f"Contribution distribution - {config.readable_name(target_name)}",
              fontsize=11, fontweight="bold")
    plt.xlabel(f"Contribution ({unit})")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_all_targets_heatmap(all_importance, path):
    """
    All features in a single visualization. 
    Each column is normalized within itself; 
    a dark color indicates that the model is not using the feature at all.
    """
    
    table = pd.concat(all_importance, ignore_index=True)
    grid = table.pivot_table(index="Feature", columns="Target",
                             values="Mean_Contribution", aggfunc="first").fillna(0.0)
    if grid.empty:
        return

    column_maximum = grid.max(axis=0).replace(0, np.nan)
    normalized = (grid / column_maximum).fillna(0.0)
    normalized = normalized.loc[normalized.sum(axis=1).sort_values().index]

    figure, axis = plt.subplots(figsize=(1.15 * len(normalized.columns) + 4,
                                         0.34 * len(normalized) + 2))
    image = axis.imshow(normalized.values, aspect="auto", cmap="viridis",
                        vmin=0, vmax=1)
    axis.set_xticks(range(len(normalized.columns)))
    axis.set_xticklabels(normalized.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(normalized)))
    axis.set_yticklabels(normalized.index, fontsize=8)
    axis.set_title("Which measurement is used for which target?\n"
                   "(each column normalised independently; "
                   "dark blue = not used at all)",
                   fontsize=11, fontweight="bold", pad=12)
    figure.colorbar(image, ax=axis, label="relative importance")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    config.create_folders()

    if shap is None:
        print("\n[SKIPPED] The 'shap' library is not installed; this step cannot run.")
        print("          To install:  pip install shap")
        print("          Kurduktan sonra:  python run_pipeline.py --from 7")
        return

    if not files_ready([config.CHECK_5_MODEL_INPUT]):
        return
    if not config.MODEL_FOLDER.exists():
        print(f"[ERROR] {config.MODEL_FOLDER} yok. Run step5_train_models.py first.")
        return

    title("STEP 7 - MODEL EXPLANATION (SHAP)")

    dataset = read_table(config.CHECK_5_MODEL_INPUT)
    feature_names = config.model_feature_columns()
    features = dataset[feature_names]

    plot_folder = config.figure_folder("importance")
    plot_folder.mkdir(parents=True, exist_ok=True)

    all_importance, problems = [], []

    for target in config.ALL_TARGETS:
        model_path = config.MODEL_FOLDER / f"{target}.joblib"
        if not model_path.exists():
            print(f"  [SKIPPED] {target}: model file missing.")
            continue

        model = joblib.load(model_path)
        importance, shap_values, additive, k = explain_one_target(model, features, target)
        all_importance.append(importance)

        if not additive:
            problems.append(target)

        unit = config.unit_of(target)
        plot_importance(importance, target, unit,
                        plot_folder / f"importance_{target}.png")
        plot_distribution(shap_values, target, unit,
                          plot_folder / f"distribution_{target}.png")

        top_three = importance.head(3)
        top_three = top_three[top_three["Mean_Contribution"] > 0]
        print(f"\n  {config.readable_name(target)}  "
              f"(model uses {k} feature(s)) {'[OK]' if additive else '[PROBLEM]'}")
        for _, row in top_three.iterrows():
            print(f"      {row['Rank']}. {row['Feature'][:45]:45s} "
                  f"%{row['Importance_Percent']:5.1f}  ({row['Direction']})")

    if not all_importance:
        print("\n[WARNING] Hiçbir model açıklanamadı.")
        return

    plot_all_targets_heatmap(all_importance,
                             plot_folder / "all_targets_heatmap.png")

    combined = pd.concat(all_importance, ignore_index=True)
    output = combined[["Target", "Rank", "Feature", "Importance_Percent",
                       "Used_By_Model", "Direction"]].copy()
    output["Importance_Percent"] = output["Importance_Percent"].round(1)

    keep = (output["Used_By_Model"] == "Yes") | (output["Rank"] <= TOP_N)
    write_table(output[keep], config.TABLE_6_FEATURE_IMPORTANCE)

    used = combined[combined["Used_By_Model"] == "Yes"]
    print("\n  Number of targets each feature is used in:")
    counts = used.groupby("Feature")["Target"].nunique().sort_values(ascending=False)
    for feature, count in counts.head(10).items():
        print(f"      used in {count:2d} target(s):  {feature}")

    if problems:
        print(f"\n  [WARNING] Additivity check failed for: {problems}")

    print("\n[DONE] All steps finished. Thesis tables: data/outputs/thesis_tables/")


if __name__ == "__main__":
    main()
