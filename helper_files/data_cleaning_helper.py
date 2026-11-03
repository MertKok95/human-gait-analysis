"""
data_cleaning_helper.py - CLEARING SENSOR AND EXPOSURE DATA

TWO SEPARATE CLEANING PROCESSES
  A) SENSOR DATA  -> removes physically impossible records (ground truth)
  B) POSE DATA     -> makes AlphaPose output analyzable

Pose data cleaning order (clean_trial function applies this order)
  1. Read the JSON
  2. Remove first/last frames where the person is not fully in camera view
  3. Remove completely empty frames at the start and end
  4. Make missing frame numbers visible (add empty rows)
  5. If the missing-data ratio is too high, REJECT the trial
  6. Remove frames with bone lengths that are nonsense (tracking glitches)
  7. Fill remaining gaps (short gaps linear, long gaps spline interpolation)
  8. If there are still gaps, REJECT the trial
  9. Filter out jitter (high-frequency noise)
"""

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config
from helper_files import file_process_helper


# =====================================================================
# =====================================================================
# 2. POSE TEMİZLİK ADIMLARI
# =====================================================================

def trim_camera_edges(pose_table, condition):
    """
    Camera edges are trimmed because pose estimation is unreliable 
    when entering and exiting the frame.
    Fewer frames are trimmed for fast gait (FGS) 
    because the person enters and exits the frame more quickly.
    """
    
    if pose_table.empty:
        return pose_table

    burn_in, burn_out = (20, 10) if "FGS" in str(condition).upper() else (24, 15)
    start = pose_table.index.min() + burn_in
    end = pose_table.index.max() - burn_out
    return pose_table[(pose_table.index >= start) & (pose_table.index <= end)]


def trim_empty_frames(pose_table):
    """It discards the completely empty squares at the beginning and end."""
    if pose_table.empty:
        return pose_table
    x_columns = [c for c in pose_table.columns if c.endswith("_x")]
    filled = pose_table.dropna(how="all", subset=x_columns)
    if filled.empty:
        return pd.DataFrame()
    return pose_table.loc[int(filled.index.min()): int(filled.index.max())].copy()


def show_missing_frames(pose_table):
    """It adds empty rows for missing frame numbers (makes them visible)."""
    if pose_table.empty:
        return pose_table
    return pose_table.reindex(range(int(pose_table.index.min()), int(pose_table.index.max()) + 1))


def missing_ratio(pose_table):
    if pose_table.empty:
        return 1.0
    return pose_table.isna().sum().sum() / (pose_table.shape[0] * pose_table.shape[1])


def remove_tracking_glitches(pose_table, tolerance=config.BONE_LENGTH_TOLERANCE):
    """
    If the bone length in a frame deviates by more than 40% from the overall median of the video, the knee and ankle points in that frame are removed for later interpolation.
    """
    if pose_table.empty:
        return pose_table

    result = pose_table.copy()
    for side in ["l", "r"]:
        hip_x, hip_y = f"{side}_hip_x", f"{side}_hip_y"
        knee_x, knee_y = f"{side}_knee_x", f"{side}_knee_y"
        ankle_x, ankle_y = f"{side}_ankle_x", f"{side}_ankle_y"
        needed = [hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y]
        if not all(c in result.columns for c in needed):
            continue

        thigh = np.hypot(result[hip_x] - result[knee_x],
                         result[hip_y] - result[knee_y])
        calf = np.hypot(result[knee_x] - result[ankle_x],
                        result[knee_y] - result[ankle_y])

        bad = (_out_of_range(thigh, tolerance) | _out_of_range(calf, tolerance))
        result.loc[bad, [knee_x, knee_y, ankle_x, ankle_y]] = np.nan

    return result


def _out_of_range(lengths, tolerance):
    """It marks the elements in a length array that deviate significantly from the median."""
    median = lengths.median()
    return (lengths < median * (1 - tolerance)) | (lengths > median * (1 + tolerance))


def find_gaps(pose_table):
    """
    It lists the length of continuous gaps for each joint.
    Output: [{'joint': 'l_ankle', 'start': 40, 'end': 43, 'length': 4}, ...]
    """
    if pose_table.empty:
        return []

    gaps = []
    for column in [c for c in pose_table.columns if c.endswith("_x")]:
        joint = column[:-2]
        missing = pose_table[column].isna()
        start, length = None, 0

        for frame, is_missing in missing.items():
            if is_missing:
                start = frame if start is None else start
                length += 1
            elif start is not None:
                gaps.append({"joint": joint, "start": start,
                             "end": frame - 1, "length": length})
                start, length = None, 0

        if start is not None:
            gaps.append({"joint": joint, "start": start,
                         "end": pose_table.index.max(), "length": length})
    return gaps


def fill_gaps(pose_table, gaps):
    """
    It fills the gaps:
      1-3 frames  -> straight line (linear)
      4-10 frames -> smooth curve (cubic spline)
    Gaps longer than 10 frames cannot be reliably filled -> None is returned (test is rejected).
    """
    if pose_table.empty or not gaps:
        return pose_table.copy()

    result = pose_table.copy()
    linear = result.interpolate(method="linear", limit_direction="both")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)
            spline = result.interpolate(method="spline", order=3,
                                        limit_direction="both")
    except (UserWarning, ValueError):
        print("      [REJECTED] Spline fitting failed.")
        return None

    for gap in gaps:
        columns = [f"{gap['joint']}_x", f"{gap['joint']}_y"]
        span = slice(gap["start"], gap["end"])

        if gap["length"] <= 3:
            result.loc[span, columns] = linear.loc[span, columns]
        elif gap["length"] <= config.MAXIMUM_GAP_FRAMES:
            result.loc[span, columns] = spline.loc[span, columns]
        else:
            print(f"      [REJECTED] Gap too long ({gap['joint']}): "
                  f"{gap['length']} frames")
            return None

    return result


def smooth_signal(pose_table, cutoff=6.0, order=4):
    """
    Butterworth low-pass filter: removes high-frequency tremor (pose estimation noise)
    """
    if pose_table.empty or len(pose_table) < 20:
        print(f"      [INFO] Too short to filter ({len(pose_table)} frames), skipped.")
        return pose_table.copy()

    result = pose_table.copy()
    b, a = butter(N=order, Wn=cutoff / (config.FRAMES_PER_SECOND / 2), btype="low")
    for column in [c for c in result.columns if c != "frame"]:
        if not result[column].isna().any():
            result[column] = filtfilt(b, a, result[column])
    return result


def smooth_array(values, cutoff=6.0, order=4):
    """The same filter for a single array of numbers (used in signal analysis)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 20 or np.isnan(values).any():
        return values.copy()
    b, a = butter(N=order, Wn=cutoff / (config.FRAMES_PER_SECOND / 2), btype="low")
    return filtfilt(b, a, values)


# =====================================================================
# 3. CLEAR THE SINGLE ATTEMPT FROM START TO FINISH
# =====================================================================

def clean_trial(json_path, condition):
    """
    Clears a single trial (video) from start to finish.
    """
    pose_table = file_process_helper.load_pose_json(json_path)
    pose_table = trim_camera_edges(pose_table, condition)
    pose_table = trim_empty_frames(pose_table)
    pose_table = show_missing_frames(pose_table)

    if missing_ratio(pose_table) > config.MAXIMUM_MISSING_RATIO:
        print(f"      [REJECTED] Missing-data ratio too high: {missing_ratio(pose_table):.1%}")
        return None

    pose_table = remove_tracking_glitches(pose_table)

    if missing_ratio(pose_table) > config.MAXIMUM_MISSING_RATIO:
        print(f"      [REJECTED] Missing ratio still too high after glitch removal: "
              f"{missing_ratio(pose_table):.1%}")
        return None

    gaps = find_gaps(pose_table)
    too_long = [g for g in gaps if g["length"] > config.MAXIMUM_GAP_FRAMES]
    if too_long:
        print(f"      [REJECTED] {len(too_long)} gaps are too long to fill.")
        return None

    pose_table = fill_gaps(pose_table, gaps)
    if pose_table is None:
        return None

    if pose_table.isna().sum().sum() != 0:
        print("      [REJECTED] Missing values remain after gap filling.")
        return None

    return smooth_signal(pose_table)


# =====================================================================
# 3. SENSOR DATA QUALITY CONTROL
# =====================================================================

def find_sensor_problem(sensor_row):
    """
    It filters a patient's sensor line according to the rules of logic and physics.
    """
    if sensor_row.isnull().any():
        return "Missing sensor reading"

    try:
        normal_speed = float(sensor_row["Velocity_UGS"])   # normal yürüyüş hızı
        fast_speed = float(sensor_row["Velocity_FGS"])     # hızlı yürüyüş hızı

        if normal_speed <= 0 or normal_speed > 2.5:
            return f"Implausible gait speed ({normal_speed} m/s)"
        if normal_speed > fast_speed:
            return "Contradiction: usual gait faster than fast gait"
        if (normal_speed == fast_speed
                and float(sensor_row["Step_UGS"]) == float(sensor_row["Step_FGS"])):
            return "Duplicate record: usual and fast gait identical"
    except (ValueError, TypeError, KeyError):
        return "Value could not be converted to a number"

    return None


def clean_sensor_data(sensor_table, patient_ids):
    """
    Cleans the sensor table.
    Returns: (cleaned table, list of excluded patients)
    """
    sensor_table_copy = sensor_table.copy()
    sensor_table_copy["ID"] = sensor_table_copy["ID"].str.strip()
    sensor_table_copy = sensor_table_copy.set_index("ID")

    accepted_ids, rejections = [], []
    for patient_id in patient_ids:
        if patient_id not in sensor_table_copy.index:
            rejections.append({"Participant_ID": patient_id,
                               "Exclusion_Reason": "Not present in the sensor file"})
            continue

        problem = find_sensor_problem(sensor_table_copy.loc[patient_id])
        if problem:
            rejections.append({"Participant_ID": patient_id, "Exclusion_Reason": problem})
        else:
            accepted_ids.append(patient_id)

    return sensor_table_copy.loc[accepted_ids].reset_index(), rejections
