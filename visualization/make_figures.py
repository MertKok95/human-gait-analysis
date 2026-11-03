import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch

import config
from helper import read_table

BONES = [
    ("nose", "mid_shoulder", "#8c8c8c", 3), ("mid_shoulder", "mid_hip", "#8c8c8c", 3),
    ("l_shoulder", "r_shoulder", "#8c8c8c", 3), ("l_shoulder", "l_hip", "#8c8c8c", 3),
    ("r_shoulder", "r_hip", "#8c8c8c", 3), ("l_hip", "r_hip", "#8c8c8c", 3),
    ("l_shoulder", "l_elbow", "#b0b0b0", 2.5), ("l_elbow", "l_wrist", "#b0b0b0", 2.5),
    ("r_shoulder", "r_elbow", "#b0b0b0", 2.5), ("r_elbow", "r_wrist", "#b0b0b0", 2.5),
    ("r_hip", "r_knee", "#d64161", 4), ("r_knee", "r_ankle", "#d64161", 4),
    ("l_hip", "l_knee", "#005b96", 4), ("l_knee", "l_ankle", "#005b96", 4),
]

JOINTS = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
          "nose", "mid_shoulder", "mid_hip"]

LEFT_LEG = {"l_hip", "l_knee", "l_ankle"}
RIGHT_LEG = {"r_hip", "r_knee", "r_ankle"}


# =====================================================================
# 1. SKELETON DRAWINGS
# =====================================================================

def pick_best_frame(df):
    """
    it selects the most meaningful frame for the drawing: the moment when the knee is bent the most (mid-swing).
    Anatomically impossible frames (those with absurd bone lengths) are eliminated.
    """
    frame_data = df.copy()
    for side in ["l", "r"]:
        frame_data[f"{side}_thigh"] = np.hypot(
            frame_data[f"{side}_hip_x"] - frame_data[f"{side}_knee_x"],
            frame_data[f"{side}_hip_y"] - frame_data[f"{side}_knee_y"])
        frame_data[f"{side}_calf"] = np.hypot(
            frame_data[f"{side}_knee_x"] - frame_data[f"{side}_ankle_x"],
            frame_data[f"{side}_knee_y"] - frame_data[f"{side}_ankle_y"])

    tolerance = 0.35
    reliable = frame_data
    for column in ["l_thigh", "l_calf", "r_thigh", "r_calf"]:
        median = frame_data[column].median()
        reliable = reliable[reliable[column].between(median * (1 - tolerance),
                                                     median * (1 + tolerance))]
    if reliable.empty:
        reliable = frame_data

    reliable = reliable.copy()
    reliable["max_flexion"] = reliable[["l_knee_flexion",
                                        "r_knee_flexion"]].max(axis=1)
    safe = reliable[reliable["max_flexion"] < 100]
    source = safe if not safe.empty else reliable
    return source.loc[source["max_flexion"].idxmax()]


def draw_one_skeleton(axis, row, title):
    """Draws a single frame's skeleton."""
    row = row.copy()
    for name, left, right in [("mid_shoulder", "l_shoulder", "r_shoulder"),
                              ("mid_hip", "l_hip", "r_hip")]:
        for axis_name in ["x", "y"]:
            row[f"{name}_{axis_name}"] = (row[f"{left}_{axis_name}"]
                                          + row[f"{right}_{axis_name}"]) / 2

    for start, end, color, width in BONES:
        points = [row.get(f"{start}_x"), row.get(f"{start}_y"),
                  row.get(f"{end}_x"), row.get(f"{end}_y")]
        if all(pd.notna(p) for p in points):
            axis.plot([points[0], points[2]], [points[1], points[3]],
                      color=color, linewidth=width, solid_capstyle="round")

    for joint in JOINTS:
        x, y = row.get(f"{joint}_x"), row.get(f"{joint}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        color = ("#005b96" if joint in LEFT_LEG else
                 "#d64161" if joint in RIGHT_LEG else "#333333")
        axis.scatter(x, y, color=color, s=40, edgecolors="white",
                     linewidths=1, zorder=6)

    valid_x = [row[f"{j}_x"] for j in JOINTS if pd.notna(row.get(f"{j}_x"))]
    valid_y = [row[f"{j}_y"] for j in JOINTS if pd.notna(row.get(f"{j}_y"))]
    if valid_x and valid_y:
        height = max(valid_y) - min(valid_y)
        axis.set_xlim(min(valid_x) - height * 0.4, max(valid_x) + height * 0.4)
        axis.set_ylim(min(valid_y) - height * 0.15, max(valid_y) + height * 0.15)
        axis.axhline(max(valid_y) + height * 0.05, color="#404040",
                     linestyle="--", linewidth=1.5, alpha=0.5)

    axis.set_title(title, fontweight="bold", fontsize=10)
    axis.set_aspect("equal")
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid(True, linestyle="-", color="#eeeeee")
    axis.invert_yaxis()


def draw_skeletons(count=20, columns=4, name="skeleton"):
    """
    Draws the skeletons of the requested number of participants in a single figure.
    count=1 single participant, count=3 triple panel, count=20 bird's eye view.
    """
    files = sorted((config.CLEAN_POSE_FOLDER / "ugs").glob("*_ugs_WoJ_1_measurements.csv"))
    if not files:
        print("  [SKIPPED] Measurement files not found. Run step 2 first.")
        return

    chosen = files[:count] if len(files) <= count else list(
        np.random.choice(files, count, replace=False))

    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 5.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for axis, path in zip(axes, chosen):
        df = pd.read_csv(path)
        patient_id = path.name.split("_")[0]
        try:
            best = pick_best_frame(df)
            draw_one_skeleton(axis, best,
                              f"{patient_id} | frame {int(best['frame'])}\n"
                              f"knee flexion: {best['max_flexion']:.1f}°")
        except Exception:
            axis.text(0.5, 0.5, f"{patient_id}\nnot drawn", ha="center")
            axis.axis("off")

    for axis in axes[len(chosen):]:
        axis.axis("off")

    figure.suptitle("2D skeletal kinematics - mid-swing",
                    fontsize=16, fontweight="bold")
    figure.legend(handles=[
        Line2D([0], [0], color="#005b96", lw=4, label="Left leg"),
        Line2D([0], [0], color="#d64161", lw=4, label="Right leg"),
        Line2D([0], [0], color="#8c8c8c", lw=3, label="Trunk and arms"),
        Line2D([0], [0], color="#404040", lw=1.5, ls="--", label="Gait surface")],
        loc="lower center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(figure, f"skeleton/{name}.png")


# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

def draw_eda():
    """Draws the distributions, correlations, and differences between conditions."""
    if not config.CHECK_4_TRIAL_MEASURES.exists():
        print("  [SKIPPED] Trial measurements not found.")
        return

    df = read_table(config.CHECK_4_TRIAL_MEASURES)
    df = df[df["Spatial_Valid_Cycle_Count"] > 0].copy()
    df["Condition"] = np.where(df["Video_ID"].str.startswith("fgs"),
                           "Fast gait", "Usual gait")
    palette = {"Usual gait": "#1f77b4", "Fast gait": "#ff7f0e"}
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # --- Distributions ---
    metrics = [("MeanSpeed", "Velocity (m/s)"), ("Cadence", "Cadence (step/min)"),
               ("StepLength_Mean", "Step length (cm)"),
               ("StrideLength_Mean", "Stride length (cm)"),
               ("KneeROM_Mean", "Knee range of motion (degrees)"),
               ("StancePercent_Mean", "Stance phase (%)")]

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, (column, label) in zip(axes.flatten(), metrics):
        if column not in df.columns:
            continue
        sns.histplot(data=df, x=column, hue="Condition", kde=True, ax=axis,
                     element="step", alpha=0.35, palette=palette)
        axis.set_title(label, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Number")
    figure.suptitle("Distribution of gait measurements", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(figure, "exploratory/measurement_distributions.png")

    # --- Correlation heat map ---
    columns = [c for c, _ in metrics if c in df.columns]
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=[label for c, label in metrics if c in df.columns],
                yticklabels=[label for c, label in metrics if c in df.columns])
    plt.title("Correlation between measurements", fontweight="bold", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(figure, "exploratory/correlation_matrix.png")

    # --- Box graphs between conditions ---
    box_metrics = [("MeanSpeed", "Velocity (m/s)"), ("Cadence", "Cadence (step/min)"),
                   ("StepLength_Mean", "Step length (cm)"),
                   ("StancePercent_Mean", "Stance phase (%)")]
    box_metrics = [(c, l) for c, l in box_metrics if c in df.columns]
    if box_metrics:
        figure, axes = plt.subplots(1, len(box_metrics), figsize=(3.6 * len(box_metrics), 6))
        for axis, (column, label) in zip(np.atleast_1d(axes), box_metrics):
            sns.boxplot(data=df, x="Condition", y=column, hue="Condition",
                        palette=palette, legend=False, ax=axis)
            axis.set_title(label, fontweight="bold")
            axis.set_xlabel("")
            axis.set_ylabel("")
        figure.suptitle("Usual vs fast gait comparison",
                        fontsize=15, fontweight="bold")
        plt.tight_layout()
        _save(figure, "exploratory/condition_comparison.png")

    # --- Principal component analysis (PCA) ---
    pca_columns = [c for c in ["MeanSpeed", "Cadence", "StepLength_Mean",
                               "KneeROM_Mean", "StancePercent_Mean",
                               "StepLength_CV", "StanceSymmetryIndex_Pct"]
                   if c in df.columns]
    pca_data = df.dropna(subset=pca_columns)
    if len(pca_data) >= 3 and len(pca_columns) >= 2:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(pca_data[pca_columns])
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(scaled)
        plot_data = pca_data.copy()
        plot_data["Component_1"] = coordinates[:, 0]
        plot_data["Component_2"] = coordinates[:, 1]

        figure = plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data, x="Component_1", y="Component_2",
                        hue="Condition", palette=palette, alpha=0.7, s=60)
        variance = pca.explained_variance_ratio_
        plt.title(f"Principal component analysis of gait measurements\n"
                  f"Explained variance: PC1 {variance[0]*100:.0f}%, "
                  f"PC2 {variance[1]*100:.0f}%", fontweight="bold")
        _save(figure, "exploratory/principal_components.png")

    # --- Symmetry and variability ---
    if {"StanceSymmetryIndex_Pct", "StepLength_CV", "MeanSpeed"} <= set(df.columns):
        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.kdeplot(data=df, x="StanceSymmetryIndex_Pct", hue="Condition",
                    fill=True, palette=palette, ax=axes[0])
        axes[0].set_title("Distribution of gait symmetry", fontweight="bold")
        axes[0].set_xlabel("Symmetry index (%) - lower is better")

        sns.scatterplot(data=df, x="MeanSpeed", y="StepLength_CV", hue="Condition",
                        palette=palette, alpha=0.6, ax=axes[1])
        axes[1].set_title("Velocity vs Step Length Variability", fontweight="bold")
        axes[1].set_xlabel("Mean speed (m/s)")
        axes[1].set_ylabel("Step length variability (%)")
        plt.tight_layout()
        _save(figure, "exploratory/symmetry_and_variability.png")



# =====================================================================
# 3. METHOD DIAGRAMS
# =====================================================================

def draw_foreshortening():
    """
    Foreshortening effect: if a bone is angled towards the camera, 
    it will appear shorter on screen. This is the reason why
    we use the 95th percentile.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 6))
    for axis in (left, right):
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 10)
        axis.axis("off")
        axis.plot([2, 2], [1, 9], color="gray", linestyle="--", linewidth=2.5)
        axis.text(1.4, 5, "Camera / image plane", rotation=90, va="center",
                  ha="center", fontsize=11, color="gray", fontweight="bold")

    left.plot([6, 6], [2.5, 7.5], color="#1f77b4", linewidth=8, solid_capstyle="round")
    left.text(6.4, 5, "True bone length", va="center", fontsize=11,
              fontweight="bold", color="#1f77b4")
    left.plot([2, 2], [2.5, 7.5], color="#d62728", linewidth=5, solid_capstyle="round")
    left.text(2.6, 5, "Projected length\n(accurate)", rotation=90, va="center",
              ha="center", fontsize=10, color="#d62728", fontweight="bold")
    for y in (2.5, 7.5):
        left.plot([2, 6], [y, y], color="black", linestyle=":", alpha=0.5)
    left.set_title("A) Parallel to image plane\n(no foreshortening)", fontsize=13,
                   fontweight="bold")

    right.plot([6, 8.5], [7.5, 4.0], color="#1f77b4", linewidth=8,
               solid_capstyle="round")
    right.text(7.6, 6.0, "True bone length", fontsize=11,
               fontweight="bold", color="#1f77b4", rotation=-54)
    right.plot([2, 2], [4.0, 7.5], color="#d62728", linewidth=5,
               solid_capstyle="round")
    right.text(2.6, 5.75, "Appears SHORTER\n(foreshortening error)", rotation=90,
               va="center", ha="center", fontsize=10, color="#d62728",
               fontweight="bold")
    right.plot([2, 6], [7.5, 7.5], color="black", linestyle=":", alpha=0.5)
    right.plot([2, 8.5], [4.0, 4.0], color="black", linestyle=":", alpha=0.5)
    right.annotate("", xy=(6, 4.0), xytext=(8.5, 4.0),
                   arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    right.text(7.25, 3.5, "Depth (Z) deviation", ha="center", fontsize=10,
               color="green", fontweight="bold")
    right.set_title("B) Angled to image plane\n(foreshortening)", fontsize=13,
                    fontweight="bold")

    plt.tight_layout()
    _save(figure, "diagrams/foreshortening.png")


def draw_knee_angle():
    """Draws a diagram showing how the knee angle and flexion are defined."""
    figure, axis = plt.subplots(figsize=(6, 8))
    axis.axis("off")

    hip, knee, ankle = np.array([0.0, 10.0]), np.array([0.0, 5.0]), np.array([-2.5, 0.0])
    for point, label, offset in [(hip, "Hip (A)", 0.3), (knee, "Knee (B)", 0.3),
                                 (ankle, "Ankle (C)", -0.3)]:
        axis.plot(point[0], point[1], "ko", markersize=10, zorder=5)
        axis.text(point[0] + offset, point[1], label, fontsize=12,
                  fontweight="bold", va="center",
                  ha="left" if offset > 0 else "right")

    axis.plot([hip[0], knee[0]], [hip[1], knee[1]], "#1f77b4", lw=5, label="Thigh")
    axis.plot([knee[0], ankle[0]], [knee[1], ankle[1]], "#2ca02c", lw=5, label="Shank")
    axis.plot([knee[0], knee[0]], [knee[1], knee[1] - 3], "k--", lw=2,
              label="Reference (0° flexion)")

    calf_angle = np.degrees(np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]))
    axis.add_patch(Arc(tuple(knee), 2, 2, theta1=calf_angle, theta2=90,
                       color="#1f77b4", lw=2))
    axis.text(knee[0] - 0.5, knee[1] + 1.2, r"$\theta$", fontsize=16,
              color="#1f77b4", fontweight="bold")
    axis.add_patch(Arc(tuple(knee), 3, 3, theta1=calf_angle, theta2=270,
                       color="#d62728", lw=2))
    axis.text(knee[0] - 2.8, knee[1] - 1.8, r"fleksiyon $= 180° - \theta$",
              fontsize=12, color="#d62728", fontweight="bold")

    axis.set_xlim(-4, 3)
    axis.set_ylim(-1, 11)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    _save(figure, "diagrams/knee_angle.png")


# =====================================================================
# 4. BASELINE ERROR GRAPH
# =====================================================================

def draw_baseline_errors():
    """Draws the average error of the baseline camera method for step and stride measurements."""
    sensor = pd.read_csv(config.SENSOR_MEASUREMENTS)
    baseline = pd.read_csv(config.BASELINE_ESTIMATES)
    merged = pd.merge(sensor, baseline, on="ID", suffixes=("_real", "_prediction"))

    parameters = ["Step_UGS", "Stride_UGS", "Step_FGS", "Stride_FGS"]
    labels = ["Step\n(usual)", "Stride\n(usual)",
              "Step\n(fast)", "Stride\n(fast)"]
    errors = [float(np.abs(merged[f"{p}_real"] - merged[f"{p}_prediction"]).mean())
              for p in parameters]

    figure = plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    bars = plt.bar(labels, errors, color=colors, edgecolor="black", alpha=0.85)
    for bar, value in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.4,
                 f"{value:.1f} cm", ha="center", fontsize=12, fontweight="bold")

    plt.title("Baseline camera methodnin ortalama hatası\n"
              "(values computed from the source CSV files)",
              fontsize=13, fontweight="bold")
    plt.ylabel("Mean absolute error (cm)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(errors) + 5)
    plt.legend(handles=[Patch(facecolor="#1f77b4", label="Usual gait"),
                        Patch(facecolor="#ff7f0e", label="Fast gait")],
               loc="upper left")
    plt.tight_layout()
    _save(figure, config.BASELINE_ERROR_FIGURE)


# =====================================================================

def _save(figure, relative_path):
    path = config.FIGURE_FOLDER / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  [SAVED] {relative_path}")


def main():
    config.create_folders()
    what = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    print(f"\nGenerating figures: {what}\n")

    if what in ("all", "skeleton"):
        draw_skeletons(count=1, columns=1, name="single_participants")
        draw_skeletons(count=3, columns=3, name="three_participants")
        draw_skeletons(count=20, columns=4, name="twenty_participants")
    if what in ("all", "eda"):
        draw_eda()
    if what in ("all", "diagram"):
        draw_foreshortening()
        draw_knee_angle()
    if what in ("all", "baseline"):
        draw_baseline_errors()

    print(f"\n[DONE] Figures: {config.FIGURE_FOLDER}")


if __name__ == "__main__":
    main()
