import sys

sys.dont_write_bytecode = True

import pandas as pd

import config
from business_logic import extract_trial_measurements
from pre_data_cleaning import append_pose_cleaning_summary
from helper import *

def process_required_trials(pose_patient_id, patient_info):
    """
    It processes mandatory (jacket-less) attempts. 
    It returns: (cycle_list, summary_list, error_reason)
    If the error reason is None, participants are eliminated.
    """
    cycles, summaries = [], []

    for condition in config.WALKING_CONDITIONS:
        for trial in config.REQUIRED_TRIALS:
            trial_label = f"{condition}_{trial}"
            pose_file = clean_pose_path(pose_patient_id, condition, trial)

            if not pose_file.exists():
                return [], [], f"Clean pose file missing ({trial_label})"

            cycle_table, summary = extract_trial_measurements(
                pose_file, patient_info, pose_patient_id, trial_label)

            if cycle_table is None:
                return [], [], f"Kinematik hata ({trial_label})"
            if cycle_table.empty:
                return [], [], f"No valid gait cycle ({trial_label})"
            if summary["Spatial_Valid_Cycle_Count"].iloc[0] == 0:
                return [], [], f"No cycle passed quality control ({trial_label})"

            summary["Jacket_Trial"] = 0
            cycles.append(cycle_table)
            summaries.append(summary)

    return cycles, summaries, None


def process_optional_trials(patient_id, patient_info):
    """
    The trials with the jacket work.
    """
    summaries = []

    for condition in config.WALKING_CONDITIONS:
        for trial in config.OPTIONAL_TRIALS:
            json_path = raw_pose_json_path(patient_id, condition, trial)
            if not json_path.exists():
                continue

            try:
                cleaned = clean_trial(json_path, condition)
                if cleaned is None:
                    continue
                pose_file = clean_pose_path(patient_id, condition, trial)
                save_pose_table(cleaned, pose_file)

                _, summary = extract_trial_measurements(
                    pose_file, patient_info, patient_id, f"{condition}_{trial}")
                if summary is not None and not summary.empty:
                    summary["Jacket_Trial"] = 1
                    summaries.append(summary)
            except Exception as error:
                print(f"      [SKIPPED] {patient_id} {condition} {trial}: {error}")

    return summaries


def main():
    config.create_folders()
    if not files_ready([config.CHECK_2_CLEAN_SENSOR, config.PARTICIPANT_INFO]):
        print("        Run pre_data_cleaning.py first.")
        return

    title("STEP 2 - FEATURE EXTRACTION")

    sensor_table = read_table(config.CHECK_2_CLEAN_SENSOR)
    patient_info = pd.read_csv(config.PARTICIPANT_INFO)

    already_rejected = load_rejected_patients()
    patient_ids = [p for p in sensor_table["ID"].unique()
                   if p not in already_rejected]
    print(f"Participants to process: {len(patient_ids)}\n")

    all_cycles, all_summaries = [], []
    accepted_patients, rejections = [], []
    jacket_trial_count = 0

    for number, patient_id in enumerate(patient_ids, start=1):
        cycles, summaries, failure = process_required_trials(patient_id, patient_info)

        if failure:
            print(f"  [{number}/{len(patient_ids)}] {patient_id} EXCLUDED - {failure}")
            rejections.append({"Participant_ID": patient_id, "Exclusion_Reason": failure})
            continue

        jacket_summaries = process_optional_trials(patient_id, patient_info)
        jacket_trial_count += len(jacket_summaries)

        accepted_patients.append(patient_id)
        all_cycles.extend(cycles)
        all_summaries.extend(summaries + jacket_summaries)

        if number % 25 == 0:
            print(f"  ... {number}/{len(patient_ids)} participants processed "
                  f"({len(accepted_patients)} kabul)")

    add_rejected_patients(rejections)
    
    append_pose_cleaning_summary(len(patient_ids), len(accepted_patients), rejections)

    if all_cycles:
        write_table(pd.concat(all_cycles, ignore_index=True),
                    config.CHECK_3_GAIT_CYCLES)
    if all_summaries:
        write_table(pd.concat(all_summaries, ignore_index=True),
                    config.CHECK_4_TRIAL_MEASURES)

    build_frame_level_dataset(accepted_patients, config.CHECK_9_FRAME_DATASET)

    print(f"\n  Accepted participants   : {len(accepted_patients)}")
    print(f"  Jacket trials processed: {jacket_trial_count}")
    print("\n[DONE] Next step: python ML/prepare_ml_dataset.py")


if __name__ == "__main__":
    main()
