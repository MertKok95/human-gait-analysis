"""
compare_distance_definitions.py — TESTING THE ANKLE DISTANCE DEFINITION

This script is not part of the main pipeline. 
It tests three different definitions of ankle distance.

Tested three definitions:
----------------

A) HORIZONTAL      |Δx|                        (current method)
B) EUCLIDEAN       √(Δx² + Δy²)                (straight-line distance between two points)
C) PROJECTED       |(Δx, Δy) · u|              (projection onto the progression axis)
                   u = unit vector fitted to the trunk

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, wilcoxon

import config
from helper import (files_ready, read_table, write_table, title,
                    read_pose_table, clean_pose_path)
from helper_files import data_cleaning_helper

DEFINITIONS = ["HORIZONTAL", "EUCLIDEAN", "PROJECTED"]

# Sensör karşılıkları: (bizim ölçümümüz, sensör sütunu kökü)
SENSOR_PAIRS = [("step_length", "Step"), ("stride_length", "Stride"),
                ("cadence", "Cadence"), ("velocity", "Velocity")]


# =====================================================================
# 1. GENERATE THREE SIGNALS
# =====================================================================

def progression_axis(pose_table):
    """
    It aligns the camera's center of gravity with its trajectory and rotates the unit vector of 
    the trajectory axis. It also gives the angle the axis makes with the horizontal. 
    If the camera is perfectly perpendicular to the trajectory plane, the angle becomes 0 degrees, 
    and the projection definition produces the same result as the horizontal definition.
    """
    
    x = pose_table["mid_hip_x_cm"].interpolate(limit_direction="both").to_numpy()
    y = pose_table["mid_hip_y_cm"].interpolate(limit_direction="both").to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.array([1.0, 0.0]), 0.0

    # The line y = a·x + b gives the direction of the slope of the axis of progression.
    slope = np.polyfit(x[valid], y[valid], 1)[0]
    direction = np.array([1.0, slope])
    direction = direction / np.linalg.norm(direction)
    angle_degrees = abs(np.degrees(np.arctan(slope)))
    return direction, angle_degrees


def build_signals(pose_table):
    """It generates an ankle distance signal based on three definitions."""
    delta_x = (pose_table["l_ankle_x_cm"] - pose_table["r_ankle_x_cm"]).to_numpy()
    delta_y = (pose_table["l_ankle_y_cm"] - pose_table["r_ankle_y_cm"]).to_numpy()

    axis, angle = progression_axis(pose_table)

    signals = {
        "HORIZONTAL": np.abs(delta_x),
        "EUCLIDEAN": np.hypot(delta_x, delta_y),
        # Projection: the component of the difference vector on the axis of progress.
        "PROJECTED": np.abs(delta_x * axis[0] + delta_y * axis[1]),
    }
    return signals, angle


# =====================================================================
# 2. SIGNAL QUALITY
# =====================================================================

def signal_quality(signal):
    """
    Three metrics:
      modulation depth = (peak - bottom) / peak   -> closer to 1 is better
      bottom (trough) level      -> lower is better (legs really close)
      peak prominence   -> higher is better (peak detection easier)
    """
    smoothed = data_cleaning_helper.smooth_array(signal, cutoff=3.0)
    if len(smoothed) < 32 or not np.isfinite(smoothed).all():
        return None

    peaks, properties = find_peaks(
        smoothed,
        distance=max(1, int(config.FRAMES_PER_SECOND / 4)),
        prominence=4.0, height=8.0)
    if len(peaks) < 3:
        return None

    peak_level = float(np.median(smoothed[peaks]))
    troughs, _ = find_peaks(-smoothed, distance=max(1, int(config.FRAMES_PER_SECOND / 4)))
    bottom_level = float(np.median(smoothed[troughs])) if len(troughs) else float(smoothed.min())

    return {
        "peak_level_cm": peak_level,
        "trough_level_cm": bottom_level,
        "modulation_depth": (peak_level - bottom_level) / peak_level if peak_level > 0 else np.nan,
        "mean_prominence": float(np.mean(properties["prominences"])),
        "relative_prominence": float(np.mean(properties["prominences"]) / peak_level)
        if peak_level > 0 else np.nan,
        "peak_count": len(peaks),
        "_smoothed": smoothed,
        "_peaks": peaks,
    }


# =====================================================================
# 3. THE CHAIN ​​REACTION: HEEL STRIKE AND PARAMETERS
# =====================================================================

def detect_heel_strikes_from_signal(pose_table, signal):
    """
    The detection logic in the pipeline is exactly the same, 
    but it works on an externally provided signal. Thus, the three definitions are 
    compared with the same algorithm and the same thresholds.
    """
    smoothed = data_cleaning_helper.smooth_array(signal, cutoff=3.0)
    peaks, properties = find_peaks(
        smoothed,
        distance=max(1, int(config.FRAMES_PER_SECOND / 4)),
        prominence=4.0, height=8.0)
    if len(peaks) < 3:
        return []

    # Gait direction: inclination of the trunk center
    hip_x = pose_table["mid_hip_x_cm"].interpolate(limit_direction="both").to_numpy()
    slope = np.polyfit(np.arange(len(hip_x)), hip_x, 1)[0]
    direction = 1 if slope > 0 else -1

    left_x = pose_table["l_ankle_x_cm"].to_numpy()
    right_x = pose_table["r_ankle_x_cm"].to_numpy()

    events = []
    for position, prominence in zip(peaks, properties["prominences"]):
        if not (np.isfinite(left_x[position]) and np.isfinite(right_x[position])):
            continue
        ahead_is_left = ((left_x[position] > right_x[position]) if direction > 0
                         else (left_x[position] < right_x[position]))
        events.append({"position": int(position), "leg": "L" if ahead_is_left else "R",
                       "prominence": float(prominence)})

    # Sequence control: L,R,L,R must be true.
    accepted = []
    for event in events:
        if not accepted or event["leg"] != accepted[-1]["leg"]:
            accepted.append(event)
        elif event["prominence"] > accepted[-1]["prominence"]:
            accepted[-1] = event
    return accepted


def trial_parameters(signal, events):
    """
    It calculates four parameters at the trial level from heel strikes. 
    Event-based averaging logic is used in the pipeline.
    """
    if len(events) < 3:
        return None

    positions = np.array([e["position"] for e in events])
    gaps = np.diff(positions)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return None

    # step length: signal value at the moment of heel strike.
    step_lengths = signal[positions]
    step_lengths = step_lengths[np.isfinite(step_lengths)]
    if len(step_lengths) == 0:
        return None

    cadence_per_gap = 60.0 / (gaps / config.FRAMES_PER_SECOND)
    mean_step = float(np.mean(step_lengths))
    mean_cadence = float(np.mean(cadence_per_gap))

    # stride length: sum of two consecutive steps
    stride_lengths = [step_lengths[i] + step_lengths[i + 1]
                      for i in range(len(step_lengths) - 1)]

    return {
        "step_length": mean_step,
        "stride_length": float(np.mean(stride_lengths)) if stride_lengths else np.nan,
        "cadence": mean_cadence,
        "velocity": (mean_step / 100.0) * (mean_cadence / 60.0),
        "heel_strike_count": len(events),
    }


# =====================================================================
# 4. MAIN STREAM
# =====================================================================

def process_all_trials():
    """it analyzes each experiment with three definitions and summarizes the results."""
    dataset = read_table(config.CHECK_5_MODEL_INPUT)
    participants = dataset["Subject_ID"].tolist()

    quality_rows, parameter_rows, alignment_rows = [], [], []
    example_saved = False

    for number, participant in enumerate(participants, start=1):
        for condition in config.WALKING_CONDITIONS:
            for trial in config.REQUIRED_TRIALS:
                path = clean_pose_path(participant, condition, trial)
                if not path.exists():
                    continue
                try:
                    pose_table = read_pose_table(path)
                except Exception:
                    continue
                if "l_ankle_x_cm" not in pose_table.columns:
                    # If the clean exposure file does not contain cm columns, it will be rescaled.
                    pose_table = _rescale(pose_table, participant)
                    if pose_table is None:
                        continue

                signals, angle = build_signals(pose_table)
                alignment_rows.append({"Subject_ID": participant,
                                       "Condition": condition, "Trial": trial,
                                       "Progression_Axis_Tilt_deg": round(angle, 3)})

                for name in DEFINITIONS:
                    quality = signal_quality(signals[name])
                    if quality is None:
                        continue
                    quality_rows.append({
                        "Subject_ID": participant, "Condition": condition,
                        "Trial": trial, "Definition": name,
                        **{k: v for k, v in quality.items() if not k.startswith("_")}})

                    events = detect_heel_strikes_from_signal(pose_table, signals[name])
                    parameters = trial_parameters(signals[name], events)
                    if parameters:
                        parameter_rows.append({
                            "Subject_ID": participant, "Condition": condition,
                            "Trial": trial, "Definition": name, **parameters})

                if not example_saved and quality_rows:
                    _plot_example(signals, participant, condition, trial)
                    example_saved = True

        if number % 25 == 0:
            print(f"  ... {number}/{len(participants)} participants processed")

    return (pd.DataFrame(quality_rows), pd.DataFrame(parameter_rows),
            pd.DataFrame(alignment_rows))


def _rescale(pose_table, participant):
    """If the clean pose file is in pixels, it converts it to centimeters."""
    from business_logic import get_patient_height_cm
    from helper import (measure_bone_lengths, pixel_to_cm_scale,
                        add_cm_coordinates, remove_impossible_coordinates)
    try:
        info = pd.read_csv(config.PARTICIPANT_INFO)
        bones = measure_bone_lengths(pose_table)
        scale = pixel_to_cm_scale(bones, get_patient_height_cm(info, participant))
        pose_table = add_cm_coordinates(pose_table, bones, scale)
        return remove_impossible_coordinates(pose_table)
    except Exception:
        return None


def _plot_example(signals, participant, condition, trial):
    """It shows all three signals on the same graph — visual evidence."""
    figure, axis = plt.subplots(figsize=(13, 5))
    time_axis = np.arange(len(signals["HORIZONTAL"])) / config.FRAMES_PER_SECOND
    styles = {"HORIZONTAL": ("#1F618D", "-", 2.2),
              "EUCLIDEAN": ("#B03A2E", "--", 1.8),
              "PROJECTED": ("#1E8449", ":", 2.2)}
    for name, (colour, style, width) in styles.items():
        smoothed = data_cleaning_helper.smooth_array(signals[name], cutoff=3.0)
        axis.plot(time_axis, smoothed, color=colour, linestyle=style,
                  linewidth=width, label=name)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Ankle separation (cm)")
    axis.set_title(f"Ankle separation under three distance definitions\n"
                   f"{participant} — {condition} {trial}",
                   fontsize=12, fontweight="bold")
    axis.legend()
    axis.grid(alpha=0.3)
    folder = config.FIGURE_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / "distance_definition_comparison.png",
                   dpi=200, bbox_inches="tight")
    plt.close(figure)


def summarise_quality(quality):
    """It summarizes the signal quality metrics by definition and performs paired tests."""
    metrics = ["modulation_depth", "trough_level_cm", "relative_prominence",
               "peak_count"]
    rows = []
    for metric in metrics:
        wide = quality.pivot_table(index=["Subject_ID", "Condition", "Trial"],
                                   columns="Definition", values=metric).dropna()
        row = {"Metric": metric, "Trials_Compared": len(wide)}
        for name in DEFINITIONS:
            row[f"{name}_mean"] = round(float(wide[name].mean()), 4)

        # HORIZONTAL matched test against others
        for other in ["EUCLIDEAN", "PROJECTED"]:
            better = (wide["HORIZONTAL"] > wide[other] if metric != "trough_level_cm"
                      else wide["HORIZONTAL"] < wide[other])
            row[f"HORIZONTAL_better_vs_{other}"] = f"{int(better.sum())}/{len(wide)}"
            try:
                row[f"p_vs_{other}"] = float(wilcoxon(wide["HORIZONTAL"], wide[other])[1])
            except ValueError:
                row[f"p_vs_{other}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_downstream(parameters):
    """
    It summarizes the downstream parameters calculated with each definition and correlates them with the sensor data.
    The decision table: correlation of parameters calculated with each definition and the sensor.
    Final method preference is based on this table.
    """
    sensor = read_table(config.CHECK_2_CLEAN_SENSOR)
    rows = []

    for name in DEFINITIONS:
        subset = parameters[parameters["Definition"] == name]
        averaged = (subset.groupby(["Subject_ID", "Condition"])
                    [[m for m, _ in SENSOR_PAIRS]].mean().reset_index())

        for condition_lower, condition_upper in [("ugs", "UGS"), ("fgs", "FGS")]:
            per_condition = averaged[averaged["Condition"] == condition_lower]
            merged = per_condition.merge(sensor, left_on="Subject_ID",
                                         right_on="ID", how="inner")
            for measure, sensor_root in SENSOR_PAIRS:
                sensor_column = f"{sensor_root}_{condition_upper}"
                if sensor_column not in merged.columns:
                    continue
                valid = merged[[measure, sensor_column]].dropna()
                if len(valid) < 3:
                    continue
                r, p = pearsonr(valid[measure], valid[sensor_column])
                error = (valid[measure] - valid[sensor_column]).abs().mean()
                rows.append({
                    "Definition": name, "Parameter": sensor_root,
                    "Condition": "usual gait" if condition_lower == "ugs" else "fast gait",
                    "Correlation_r": round(float(r), 4),
                    "p_value": float(p),
                    "Mean_Absolute_Error": round(float(error), 4),
                    "Participants": len(valid)})
    return pd.DataFrame(rows)


def main():
    config.create_folders()
    if not files_ready([config.CHECK_5_MODEL_INPUT, config.CHECK_2_CLEAN_SENSOR]):
        print("        Run the main pipeline (steps 1-3) first.")
        return

    title("ANKLE DISTANCE DEFINITION — EVIDENCE TEST")
    print("  Comparing three definitions on every trial:")
    print("    A) HORIZONTAL  |dx|                  (current method)")
    print("    B) EUCLIDEAN   sqrt(dx^2 + dy^2)     (straight-line distance)")
    print("    C) PROJECTED   |(dx,dy) . u|         (projection onto progression axis)\n")

    quality, parameters, alignment = process_all_trials()
    if quality.empty:
        print("[ERROR] No trial could be processed. Check clean pose files.")
        return

    # ---- 1. Camera alignment ----
    print("\n" + "=" * 74)
    print("1. CAMERA ALIGNMENT — tilt of the progression axis")
    print("=" * 74)
    tilt = alignment["Progression_Axis_Tilt_deg"]
    print(f"  Trials analysed : {len(tilt)}")
    print(f"  Median tilt     : {tilt.median():.3f} degrees")
    print(f"  90th percentile : {tilt.quantile(0.90):.3f} degrees")
    print(f"  Maximum tilt    : {tilt.max():.3f} degrees")
    print("\n  A tilt near zero means the image x-axis IS the progression axis,")
    print("  so the PROJECTED definition reduces mathematically to HORIZONTAL.")

    # ---- 2. Signal quality ----
    print("\n" + "=" * 74)
    print("2. SIGNAL QUALITY — paired comparison across all trials")
    print("=" * 74)
    quality_summary = summarise_quality(quality)
    print(quality_summary.to_string(index=False))

    # ---- 3. Chain effect (DECISION TABLE) ----
    print("\n" + "=" * 74)
    print("3. DOWNSTREAM EFFECT — correlation with the sensor reference")
    print("   (this is the decisive evidence)")
    print("=" * 74)
    downstream = summarise_downstream(parameters)
    pivot = downstream.pivot_table(index=["Parameter", "Condition"],
                                  columns="Definition", values="Correlation_r")
    print(pivot.round(4).to_string())

    print("\n  Mean absolute error against the sensor:")
    error_pivot = downstream.pivot_table(index=["Parameter", "Condition"],
                                         columns="Definition",
                                         values="Mean_Absolute_Error")
    print(error_pivot.round(3).to_string())

    # ---- Save ----
    folder = config.THESIS_TABLES
    write_table(quality_summary, folder / "Distance_Definition_Signal_Quality.csv")
    write_table(downstream, folder / "Distance_Definition_Downstream.csv")
    write_table(alignment, folder / "Distance_Definition_Camera_Alignment.csv")

    # ---- Decision ----
    print("\n" + "=" * 74)
    print("4. CONCLUSION")
    print("=" * 74)
    wins = {name: int((pivot.idxmax(axis=1) == name).sum()) for name in DEFINITIONS}
    print(f"  Highest correlation count out of {len(pivot)} parameter-condition pairs:")
    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"    {name:12s} {count}")
    print(f"\n  Median progression-axis tilt: {tilt.median():.3f} degrees")
    print("  Figure: data/outputs/figures/distance_definition_comparison.png")


if __name__ == "__main__":
    main()
