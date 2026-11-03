"""
visualization_helper.py - PIPELINE GRAPHS

  1. Heel strike detection graph   -> for each trial, shows the accuracy of the detection
  2. Clinical kinematics graph     -> normalized trunk/hip/knee angles
  3. Sensor data overview          -> distribution of the sensor data

Thesis figures (skeleton drawings, EDA, diagrams) are in a separate file:
make_figures.py
"""

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

import config


def plot_heel_strikes(raw_signal, smoothed_signal, accepted_events,
                      subject_id, trial_name):
    """
    It traces the ankle distance signal and the heel strikes that occur on it.
    """
    time_axis = np.arange(len(raw_signal)) / config.FRAMES_PER_SECOND

    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, raw_signal, alpha=0.4, label="Raw signal")
    plt.plot(time_axis, smoothed_signal, linewidth=2, label="Filtered signal")

    styles = {"L": ("blue", "o", "Left heel strike"),
              "R": ("red", "s", "Right heel strike")}
    for leg, (color, marker, label) in styles.items():
        positions = np.array([e["position"] for e in accepted_events
                              if e.get("leg") == leg], dtype=int)
        if len(positions):
            plt.scatter(time_axis[positions], smoothed_signal[positions],
                        color=color, marker=marker, s=60, zorder=5, label=label)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Ankle distance (cm)")
    plt.title(f"{subject_id} - {trial_name} gait signal")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(True, alpha=0.3)

    folder = config.figure_folder("signals")
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / f"{subject_id}_{trial_name}_heel_strikes.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_clinical_kinematics(df_pose, df_cycles, subject_id, trial_name):
    """
    It traces the trunk, hip, and knee angles in a single flawless gait cycle,
    normalizing them to the range %0-100. 
    """
    try:
        if df_cycles is None or df_cycles.empty:
            return
        valid = df_cycles[df_cycles["Spatial_Valid"]]
        if valid.empty:
            return

        cycle = valid.iloc[0]
        start, end = int(cycle["Start_HS_Frame"]), int(cycle["End_HS_Frame"])
        side = str(cycle["Leg"]).lower()

        df = df_pose.copy()
        if "frame" in df.columns:
            df = df.set_index("frame")
        if start not in df.index or end not in df.index:
            return

        segment = df.loc[start:end]
        columns = {"Trunk sway": f"{side}_trunk_sway",
                   "Hip flexion": f"{side}_hip_flexion",
                   "Knee flexion": f"{side}_knee_flexion"}
        if not all(c in segment.columns for c in columns.values()):
            return

        source = np.linspace(0, 100, len(segment))
        target = np.linspace(0, 100, 101)

        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        figure.suptitle(f"Sagittal plane kinematics - {subject_id} ({trial_name})\n"
                        f"Normalised gait cycle (0-100%)",
                        fontsize=14, fontweight="bold")

        for axis, (title, column) in zip(axes, columns.items()):
            curve = interp1d(source, segment[column].ffill().bfill(), kind="cubic")
            axis.plot(target, curve(target), color="#3C763D", linewidth=2.5)
            axis.axhline(0, color="black", linewidth=1)
            axis.set_ylabel(f"{title}\n(degrees)")
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Percent of gait cycle (%)")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        folder = config.figure_folder("kinematics")
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / f"{subject_id}_{trial_name}_kinematic.png",
                    dpi=200, bbox_inches="tight")
        plt.close(figure)

    except Exception as error:
        print(f"      [INFO] Kinematics figure for {subject_id} could not be drawn: {error}")


def plot_sensor_overview(sensor_path):
    """Sensor data overview: comparison of usual and fast gait."""
    
    df = pd.read_csv(sensor_path)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df["Step_UGS"].dropna(), kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Step length distribution (usual gait)")

    sns.scatterplot(x="Velocity_UGS", y="Velocity_FGS", data=df,
                    ax=axes[0, 1], color="coral")
    axes[0, 1].plot([0, 5], [0, 5], ls="--", color="gray")
    axes[0, 1].set_title("Gait speed: usual vs fast")

    sns.boxplot(y=df["MonoSP_UGS"].dropna(), ax=axes[1, 0], color="lightgreen")
    axes[1, 0].set_title("Single support time (usual gait)")

    sns.scatterplot(x="Cadence_UGS", y="Cadence_FGS", data=df,
                    ax=axes[1, 1], color="purple")
    axes[1, 1].set_title("Cadence: usual vs fast")

    plt.tight_layout()
    config.FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURE_FOLDER / "sensor_overview.png", dpi=200)
    plt.close()
