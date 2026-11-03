"""
file_process_helper.py - FILE READING AND WRITING OPERATIONS
"""

import json

import numpy as np
import pandas as pd

import config


# =====================================================================
# 1. READING RAW POSE DATA
# =====================================================================

def load_pose_json(json_path):
    with open(json_path, "r") as file:
        frames = json.load(file)

    rows = []
    for frame in frames:
        row = {"frame": frame["frame"]}
        joints = frame.get("joints")
        if joints:
            for joint_name, coordinates in joints.items():
                x = coordinates.get("x", np.nan)
                y = coordinates.get("y", np.nan)
                if x == 0.0 and y == 0.0:
                    x, y = np.nan, np.nan
                row[f"{joint_name}_x"] = x
                row[f"{joint_name}_y"] = y
        rows.append(row)

    pose_table = pd.DataFrame(rows)
    if not pose_table.empty:
        pose_table = pose_table.set_index("frame").sort_index()
    return pose_table


def raw_pose_json_path(patient_id, condition, trial):
    return (config.RAW_POSE_FOLDER / "pose" / patient_id / condition.upper()
            / f"{trial}_AlphaPose.json")


# =====================================================================
# 2. CLEAN POSE FILES
# =====================================================================

def clean_pose_path(patient_id, condition, trial):
    return config.clean_pose_file(patient_id, condition, trial)


def measurement_path(patient_id, condition, trial):
    return (config.CLEAN_POSE_FOLDER / condition
            / f"{patient_id}_{condition}_{trial}_measurements.csv")


def save_pose_table(pose_table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pose_table.reset_index().to_csv(path, index=False)
    return path


def read_pose_table(path):
    table = pd.read_csv(path, index_col=False)
    table = table.loc[:, ~table.columns.str.contains("^Unnamed")]
    if "frame" in table.columns:
        table = table.set_index("frame")
    return table


# =====================================================================
# 3. REJECTED PATIENT LOG
# =====================================================================

def load_rejected_patients():
    """List of patients previously rejected (to prevent them from being processed again)."""
    if not config.CHECK_1_EXCLUDED.exists():
        return set()
    log = pd.read_csv(config.CHECK_1_EXCLUDED)
    id_column = log.columns[0]
    return set(log[id_column].dropna().astype(str).str.strip())


def add_rejected_patients(new_rejections):
    """
    Adds rejected patients to the log. Same participants are not written twice.
    """
    if not new_rejections:
        return
    path = config.CHECK_1_EXCLUDED
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(new_rejections)],
                         ignore_index=True)
    combined = combined.drop_duplicates(subset=["Participant_ID"], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


# =====================================================================
# 4. FRAME-LEVEL DATASET 
# =====================================================================

FRAME_LEVEL_COLUMNS = [
    "frame", "is_heel_strike",
    "l_knee_angle", "r_knee_angle",
    "l_hip_angle", "r_hip_angle",
    "l_trunk_sway", "r_trunk_sway",
    "knee_asymmetry", "step_symmetry",
    "step_length_cm", "stride_length_cm", "ankle_distance_cm",
    "cadence", "velocity",
]


def build_frame_level_dataset(patient_ids, output_path):
    """
    All participants' frame-level measurements are combined into a single file.
    """
    parts = []
    for patient_id in patient_ids:
        for condition in config.WALKING_CONDITIONS:
            for trial in config.REQUIRED_TRIALS:
                path = measurement_path(patient_id, condition, trial)
                if not path.exists():
                    continue
                table = pd.read_csv(path)
                available = [c for c in FRAME_LEVEL_COLUMNS if c in table.columns]
                part = table[available].copy()
                part.insert(0, "Participant_ID", patient_id)
                part.insert(1, "Fast_Gait", 1 if condition == "fgs" else 0)
                part.insert(2, "Walking_Direction", 0 if trial == "WoJ_1" else 1)
                parts.append(part)

    if not parts:
        return None

    dataset = pd.concat(parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    return dataset
