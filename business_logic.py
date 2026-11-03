import sys

sys.dont_write_bytecode = True

import numpy as np

import config
from helper import *

CONSTANT_COLUMNS = [
    "step_length_cm", "stride_length_cm",
    "cadence", "velocity",
    "step_symmetry", "step_asymmetry_flag",
]

def get_patient_height_cm(patient_info_table, patient_id, default_height=170.0):
    """
    It reads the patient's actual height; if it can't find it, 
    it uses the default value.
    """
    row = patient_info_table[
        patient_info_table["ID"].str.strip() == patient_id.strip()]
    return float(row["Height"].values[0]) if not row.empty else default_height


def extract_trial_measurements(pose_file, patient_info_table,
                               patient_id, trial_label):
    """
    It extracts all measurements from a single trial.
    """
    try:
        pose_table = read_pose_table(pose_file)
    except Exception as error:
        print(f"      [ERROR] Could not read {pose_file.name}: {error}")
        return None, None

    # --- 1. Find the pixel scale and convert it to centimeters. ---
    bone_lengths = measure_bone_lengths(pose_table)
    height_cm = get_patient_height_cm(patient_info_table, patient_id)
    scale = pixel_to_cm_scale(bone_lengths, height_cm)

    pose_table = add_cm_coordinates(pose_table, bone_lengths, scale)
    pose_table = remove_impossible_coordinates(pose_table)

    # --- 2. Find heel strikes and calculate step lengths ---
    pose_table = add_ankle_distance(pose_table)
    pose_table = detect_heel_strikes(pose_table, patient_id, trial_label)

    # --- 3. Cadence and velocity (each calculated using two different methods) ---
    pose_table = add_cadence(pose_table)              # from heel strike intervals
    pose_table = add_velocity(pose_table)             # step x kadans
    pose_table = add_spectral_cadence(pose_table)     # from frequency analysis
    pose_table = add_displacement_velocity(pose_table)  # with displacement
    pose_table = fill_forward(pose_table, CONSTANT_COLUMNS)

    # --- 4. Joint angles ---
    pose_table = add_joint_angles(pose_table)
    pose_table = add_trunk_sway(pose_table)
    if has_impossible_knee_bend(pose_table):
        print(f"      [REJECTED] Knee flexion above {config.MAXIMUM_KNEE_FLEXION_DEGREES}° "
              f"detected (pose estimation error).")
        return None, None
    pose_table = add_asymmetry(pose_table)

    # --- 5. Gait cycles ---
    cycle_table = extract_cycles(pose_table, patient_id, trial_label)
    if cycle_table.empty:
        print("      [REJECTED] No gait cycle found.")
        return cycle_table, cycle_table

    # --- 6. Trial-level summary ---
    cadence, mean_speed = average_over_heel_strike_intervals(pose_table)
    trial_summary = summarise_trial(
        cycle_table,
        cadence=cadence,
        mean_speed=mean_speed,
        displacement_speed=trial_level_value(pose_table, "displacement_velocity"),
        spectral_cadence=trial_level_value(pose_table, "spectral_cadence"))

    # --- 7. Save and generate clinical graph ---
    condition = trial_label.split("_")[0]
    trial = trial_label[len(condition) + 1:]
    save_pose_table(pose_table, measurement_path(patient_id, condition, trial))
    plot_clinical_kinematics(pose_table, cycle_table, patient_id, trial_label)

    return cycle_table, trial_summary


def trial_level_value(pose_table, column_name):
    """
    It retrieves the single value of a column that remains constant throughout the trial.
    (Spectral cadence and displacement velocity are written the same for each frame.)
    """
    if column_name not in pose_table.columns:
        return np.nan
    values = pose_table[column_name].dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def average_over_heel_strike_intervals(pose_table):
    """
    The trial average is calculated based on
    EVENTS (heel strike intervals), not on SQUARES.
    """
    heel_strike_positions = np.flatnonzero(
        pose_table["is_heel_strike"].to_numpy() == 1)
    frame_gaps = np.diff(heel_strike_positions)
    frame_gaps = frame_gaps[frame_gaps > 0]

    if len(frame_gaps) == 0:
        return np.nan, np.nan

    cadence_per_interval = 60.0 / (frame_gaps / config.FRAMES_PER_SECOND)
    step_lengths = pose_table["step_length_cm"].to_numpy()[
        heel_strike_positions[:len(frame_gaps)]]
    speed_per_interval = (step_lengths / 100.0) * (cadence_per_interval / 60.0)

    return (float(np.mean(cadence_per_interval)),
            float(np.nanmean(speed_per_interval)))
