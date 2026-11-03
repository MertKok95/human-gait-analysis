"""
gait_cycle_helper.py - EXTRACT AND SUMMARISE GAIT CYCLES

A gait cycle is the time between two heel strikes of the same leg. 

This file:
  1. Extracts each cycle as a separate row (duration, step length, phases)
  2. Filters each cycle through quality control
  3. Summarizes all cycles from a trial into a single row

"""

import numpy as np
import pandas as pd

import config
from helper_files import analysis_helper


# =====================================================================
# 1. BASE EVENTS
# =====================================================================

def heel_strike_frames(df, leg):
    """Returns the frame numbers where the specified leg strikes the ground."""
    return df.index[df[config.HEEL_STRIKE_COLUMN[leg]] == 1].to_numpy()


def alternation_errors(df):
    """
    It checks the order of steps. 
    If there is a repetition like L, L, then both squares in that pattern are marked as incorrect.
    """
    left = pd.Series("L", index=df.index[df[config.HEEL_STRIKE_COLUMN["L"]] == 1])
    right = pd.Series("R", index=df.index[df[config.HEEL_STRIKE_COLUMN["R"]] == 1])
    ordered = pd.concat([left, right]).sort_index()

    errors = set()
    for i in range(1, len(ordered)):
        if ordered.iloc[i] == ordered.iloc[i - 1]:
            errors.update([ordered.index[i - 1], ordered.index[i]])
    return errors


def find_toe_off(df, start_frame, end_frame, leg):
    """
    Toe-off frame: the moment when the foot leaves the ground.

    The foot lifts off the ground when it is at its furthest point from the body.
    Therefore, we look for the frame where the horizontal distance between the hip and ankle is MAXIMUM.
    The search window is 40%-75% of the gait cycle (according to literature, toe-off occurs around 60%).
    """
    total_frames = end_frame - start_frame
    if total_frames <= 0:
        return np.nan, "invalid_cycle"

    window_start = max(start_frame + 1, start_frame + int(np.floor(total_frames * 0.40)))
    window_end = min(end_frame - 1, start_frame + int(np.ceil(total_frames * 0.75)))
    if window_start > window_end:
        return np.nan, "invalid_window"

    return _max_hip_ankle_distance(df, window_start, window_end, leg, "bulundu")


def find_leading_toe_off(df, start_frame, total_frames, opposite_leg,
                         search_fraction=0.30):
    """
    In the first cycle of the trial, the opposite leg's toe-off frame might fall outside the recording window.
    In such cases, we search within a safety margin of 30% of the cycle duration from the beginning.
    """
    if total_frames <= 0:
        return np.nan, "invalid_cycle"

    window_end = min(start_frame + int(np.ceil(total_frames * search_fraction)),
                     int(df.index.max()))
    if start_frame >= window_end:
        return np.nan, "invalid_window"

    return _max_hip_ankle_distance(df, start_frame, window_end,
                                   opposite_leg, "leading_toe_off")


def _max_hip_ankle_distance(df, start, end, leg, method_name):
    """
    Hip-ankle horizontal distance is calculated, and 
    the frame where it peaks is found.
    """
    side = leg.lower()
    hip, ankle = f"{side}_hip_x_cm", f"{side}_ankle_x_cm"
    if hip not in df.columns or ankle not in df.columns:
        return np.nan, "invalid_columns"

    window = df.loc[start:end]
    if window.empty:
        return np.nan, "invalid_window"

    distance = (window[hip] - window[ankle]).abs().replace([np.inf, -np.inf], np.nan)
    smoothed = distance.rolling(window=3, min_periods=1, center=True).mean()

    if not smoothed.notna().any():
        return np.nan, "invalid_result"
    return smoothed.idxmax(), method_name


# =====================================================================
# 2. PHASE CALCULATIONS
# =====================================================================

def stance_and_swing(start_frame, end_frame, toe_off_frame):
    """
    Stance phase = when the foot is on the ground -> from heel strike to toe-off
    Swing phase = when the foot is in the air -> from toe-off to the next heel strike
    In healthy walking, stance is ~60% and swing is ~40%.
    """
    
    total = end_frame - start_frame
    invalid = pd.isna(toe_off_frame) or not (start_frame < toe_off_frame < end_frame)
    if invalid:
        return dict.fromkeys(["stance_frames", "stance_time", "swing_time",
                              "stance_percent", "swing_percent"], np.nan) | \
               {"error": "TOE_OFF_FRAME_NOT_FOUND"}

    stance = toe_off_frame - start_frame
    swing = end_frame - toe_off_frame
    return {
        "stance_frames": stance,
        "stance_time": stance / config.FRAMES_PER_SECOND, "swing_time": swing / config.FRAMES_PER_SECOND,
        "stance_percent": stance / total * 100, "swing_percent": swing / total * 100,
        "error": None,
    }


def support_phases(start_frame, opposite_hs_frame, opposite_toe_offs,
                   toe_off_frame, stance_frames, total_frames):
    """
    Double support (çift destek) = iki ayak da yerde
    Single support (tek destek)  = tek ayak yerde
    """
    empty = dict.fromkeys(["first_double_support", "second_double_support", "double_support_time",
                           "single_support_time", "double_support_percent",
                           "single_support_percent"], np.nan)

    if pd.isna(opposite_hs_frame) or pd.isna(toe_off_frame) or pd.isna(stance_frames):
        return empty | {"error": "DESTEK_OLAYLARI_EKSIK"}

    usable = opposite_toe_offs[(opposite_toe_offs >= start_frame)
                               & (opposite_toe_offs <= toe_off_frame)]
    if len(usable) == 0:
        return empty | {"error": "OPPOSITE_TOE_OFF_FRAME_NOT_FOUND"}

    ds1 = np.min(usable) - start_frame
    ds2 = toe_off_frame - opposite_hs_frame
    total_double = ds1 + ds2

    if ds1 < 0 or ds2 < 0:
        return empty | {"error": "NEGATIVE_DOUBLE_SUPPORT"}
    if total_double > stance_frames:
        return empty | {"error": "DOUBLE_SUPPORT_LONGER_THAN_STANCE"}

    single = stance_frames - total_double
    return {
        "first_double_support": ds1 / config.FRAMES_PER_SECOND, "second_double_support": ds2 / config.FRAMES_PER_SECOND,
        "double_support_time": total_double / config.FRAMES_PER_SECOND,
        "single_support_time": single / config.FRAMES_PER_SECOND,
        "double_support_percent": total_double / total_frames * 100,
        "single_support_percent": single / total_frames * 100,
        "error": None,
    }


def knee_range_of_motion(cycle_df, leg):
    """
    The difference between the maximum and minimum values of the knee angle throughout the cycle.
    """
    column = f"{leg.lower()}_knee_angle"
    if column not in cycle_df.columns:
        return np.nan
    angles = cycle_df[column].replace([np.inf, -np.inf], np.nan).dropna()
    return angles.max() - angles.min() if len(angles) >= 2 else np.nan



# =====================================================================
# 3. CYCLES EXTRACT
# =====================================================================

def extract_cycles(df, subject_id, video_id):
    """
    Finds all walking cycles in the trial and creates a row for each cycle.
    
    Each line pass quality control; the result is reported with two flags:
        Spatial_Valid: Is step/stride length measurement available?
        Support_Valid: Is pressure/oscillation/double support measurement available?
    """
    if "frame" in df.columns:
        df = df.set_index("frame")
    if df.empty:
        return pd.DataFrame()

    df = df.sort_index()
    _check_required_columns(df)

    hs = {"L": heel_strike_frames(df, "L"), "R": heel_strike_frames(df, "R")}
    bad_order = alternation_errors(df)
    both_legs_at_once = df.index[
        (df[config.HEEL_STRIKE_COLUMN["L"]] == 1)
        & (df[config.HEEL_STRIKE_COLUMN["R"]] == 1)].to_numpy()

    records = []
    for leg in ["L", "R"]:
        other = "R" if leg == "L" else "L"

        # Pre-identify all toe-ups on the opposite leg (for double support).
        opposite_toe_offs = np.array([
            frame for start, end in zip(hs[other][:-1], hs[other][1:])
            for frame, _ in [find_toe_off(df, start, end, other)]
            if pd.notna(frame)])

        for order, (start, end) in enumerate(zip(hs[leg][:-1], hs[leg][1:]), start=1):
            total_frames = end - start
            if total_frames <= 0:
                continue

            record = _build_one_cycle(
                df, subject_id, video_id, leg, order, start, end, total_frames,
                hs[other], opposite_toe_offs, bad_order, both_legs_at_once)
            records.append(record)

    if not records:
        return pd.DataFrame()

    return (pd.DataFrame(records)
            .sort_values(["Subject_ID", "Video_ID", "Start_HS_Frame", "Leg"])
            .reset_index(drop=True))


def _check_required_columns(df):
    required = [config.HEEL_STRIKE_COLUMN["L"], config.HEEL_STRIKE_COLUMN["R"],
                "l_hip_x_cm", "r_hip_x_cm",
                "l_ankle_x_cm", "r_ankle_x_cm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if not df.index.is_unique:
        raise ValueError("Duplicate frame numbers detected.")


def _build_one_cycle(df, subject_id, video_id, leg, order, start, end,
                     total_frames, opposite_hs, opposite_toe_offs,
                     bad_order, both_legs_at_once):
    """A single gait cycle's measurements and QC result are produced."""
    cycle_time = total_frames / config.FRAMES_PER_SECOND
    cycle_df = df.loc[start:end]

    toe_off, toe_off_method = find_toe_off(df, start, end, leg)

    inside = opposite_hs[(opposite_hs > start) & (opposite_hs < end)]
    opposite_hs_frame = inside[0] if len(inside) == 1 else np.nan

    stance = stance_and_swing(start, end, toe_off)
    support = support_phases(start, opposite_hs_frame, opposite_toe_offs,
                             toe_off, stance["stance_frames"], total_frames)

    # If counter-finger lift wasn't found, try again in a wider window.
    if support.get("error") == "OPPOSITE_TOE_UP_NOT_FOUND":
        rescue_frame, rescue_method = find_leading_toe_off(
            df, start, total_frames, "R" if leg == "L" else "L")
        if pd.notna(rescue_frame):
            retry = support_phases(start, opposite_hs_frame,
                                   np.array([rescue_frame]), toe_off,
                                   stance["stance_frames"], total_frames)
            if retry.get("error") is None:
                support, toe_off_method = retry, rescue_method

    # --- Spatial measurements ---
    valid_frames = [f for f in [opposite_hs_frame, end]
                    if pd.notna(f) and f in df.index]
    step_length = (df.loc[valid_frames, "step_length_cm"].mean()
                   if valid_frames else np.nan)
    stride_length = cycle_df["stride_length_cm"].dropna().median()

    # --- Quality control ---
    spatial_problems, support_problems = [], []

    for problem, applies in [
        ("STEP_ORDER_INVALID", start in bad_order or end in bad_order),
        ("BOTH_LEGS_AT_ONCE", np.any((both_legs_at_once >= start)
                                      & (both_legs_at_once <= end))),
        ("CYCLE_TOO_SHORT", cycle_time < config.MINIMUM_CYCLE_SECONDS),
        ("CYCLE_TOO_LONG", cycle_time > config.MAXIMUM_CYCLE_SECONDS),
    ]:
        if applies:                      
            spatial_problems.append(problem)
            support_problems.append(problem)

    if pd.notna(step_length) and not (config.MINIMUM_STEP_LENGTH_CM <= step_length <= config.MAXIMUM_STEP_LENGTH_CM):
        spatial_problems.append("STEP_LENGTH_INVALID")
    if pd.notna(stride_length) and not (config.MINIMUM_STRIDE_LENGTH_CM <= stride_length <= config.MAXIMUM_STRIDE_LENGTH_CM):
        spatial_problems.append("STRIDE_LENGTH_INVALID")

    for phase in [stance, support]:
        if phase.get("error"):
            support_problems.append(phase["error"])

    return {
        "Subject_ID": subject_id, "Video_ID": video_id,
        "Cycle_ID": f"{subject_id}_{video_id}_{leg}_{order}", "Leg": leg,
        "Start_HS_Frame": start, "Opposite_HS_Frame": opposite_hs_frame,
        "End_HS_Frame": end, "TO_Frame": toe_off,
        "Cycle_Time": cycle_time,
        "Stance_Time": stance["stance_time"], "Swing_Time": stance["swing_time"],
        "Stance_Percent": stance["stance_percent"],
        "Swing_Percent": stance["swing_percent"],
        "Double_Support_Time": support["double_support_time"],
        "Single_Support_Time": support["single_support_time"],
        "Double_Support_Percent": support["double_support_percent"],
        "Single_Support_Percent": support["single_support_percent"],
        "Step_Length": step_length, "Stride_Length": stride_length,
        "Knee_ROM": knee_range_of_motion(cycle_df, leg),
        "TO_Detection_Method": toe_off_method,
        "Spatial_Valid": len(spatial_problems) == 0,
        "Spatial_Problem": "OK" if not spatial_problems else ";".join(sorted(set(spatial_problems))),
        "Support_Valid": len(support_problems) == 0,
        "Support_Problem": "OK" if not support_problems else ";".join(sorted(set(support_problems))),
    }


# =====================================================================
# 4. TEST SUMMARY (all cycles -> single rows)
# =====================================================================

def summarise_trial(df_cycles, cadence, mean_speed,
                    displacement_speed=np.nan, spectral_cadence=np.nan):
    """
    A test's cycles are summarised into a single row.
    Spatial measurements are only taken from spatially valid cycles,
    support phase measurements are only taken from support valid cycles.
    """
    if df_cycles.empty:
        return pd.DataFrame()

    rows = []
    for (subject_id, video_id), group in df_cycles.groupby(["Subject_ID", "Video_ID"]):
        spatial = group[group["Spatial_Valid"]]
        support = group[group["Support_Valid"]]

        row = {
            "Subject_ID": subject_id, "Video_ID": video_id,
            "Cadence": cadence, "MeanSpeed": mean_speed,
            "DisplacementSpeed": displacement_speed,
            "SpectralCadence": spectral_cadence,
            "Total_Cycle_Count": len(group),
            "Spatial_Valid_Cycle_Count": len(spatial),
            "Support_Valid_Cycle_Count": len(support),
        }

        row.update(_spatial_summary(spatial))
        row.update(_support_summary(support))
        rows.append(row)

    return pd.DataFrame(rows)


def _spatial_summary(spatial):
    """Mean steps/stride/duration and their variability."""
    
    keys = ["StrideDuration_Mean", "StrideDuration_CV", "StepDuration_Mean",
            "StepLength_Mean", "StepLength_CV", "StrideLength_Mean",
            "StrideLength_CV", "KneeROM_Mean"]
    if spatial.empty:
        return dict.fromkeys(keys, np.nan)

    return {
        "StrideDuration_Mean": spatial["Cycle_Time"].mean(),
        "StrideDuration_CV": analysis_helper.variation(spatial["Cycle_Time"]),
        "StepDuration_Mean": (spatial["Cycle_Time"] / 2).mean(),
        "StepLength_Mean": spatial["Step_Length"].mean(),
        "StepLength_CV": analysis_helper.variation(spatial["Step_Length"]),
        "StrideLength_Mean": spatial["Stride_Length"].mean(),
        "StrideLength_CV": analysis_helper.variation(spatial["Stride_Length"]),
        "KneeROM_Mean": spatial["Knee_ROM"].mean(),
    }


def _support_summary(support):
    """Stance/swing/double support means and their variability."""
    
    keys = ["StanceTime_Mean", "SwingTime_Mean", "DoubleSupportTime_Mean",
            "SingleSupportTime_Mean", "StancePercent_Mean", "SwingPercent_Mean",
            "DoubleSupportPercent_Mean", "SingleSupportPercent_Mean",
            "StanceSymmetryIndex_Pct"]
    if support.empty:
        return dict.fromkeys(keys, np.nan)

    summary = {f"{name}_Mean": support[column].mean() for name, column in [
        ("StanceTime", "Stance_Time"), ("SwingTime", "Swing_Time"),
        ("DoubleSupportTime", "Double_Support_Time"),
        ("SingleSupportTime", "Single_Support_Time"),
        ("StancePercent", "Stance_Percent"), ("SwingPercent", "Swing_Percent"),
        ("DoubleSupportPercent", "Double_Support_Percent"),
        ("SingleSupportPercent", "Single_Support_Percent")]}

    summary["StanceSymmetryIndex_Pct"] = analysis_helper.symmetry_index(
        support.loc[support["Leg"] == "L", "Stance_Percent"].mean(),
        support.loc[support["Leg"] == "R", "Stance_Percent"].mean())
    return summary
