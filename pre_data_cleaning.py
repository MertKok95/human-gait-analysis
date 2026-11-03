import sys

sys.dont_write_bytecode = True

import pandas as pd

import config
from helper import *


def clean_all_sensor_records():
    """It filters the sensor data and returns a list of remaining patients."""
    title("A) SENSOR DATA QUALITY CONTROL")

    patient_info = pd.read_csv(config.PARTICIPANT_INFO)
    patient_ids = patient_info["ID"].str.strip().unique().tolist()
    sensor_table = pd.read_csv(config.SENSOR_MEASUREMENTS)

    clean_table, rejections = clean_sensor_data(sensor_table, patient_ids)

    write_table(clean_table, config.CHECK_2_CLEAN_SENSOR)
    add_rejected_patients(rejections)
    write_cleaning_summary(len(patient_ids), len(clean_table), rejections)

    print(f"\n  Initial   : {len(patient_ids)} participants")
    print(f"  Excluded  : {len(rejections)} participants")
    print(f"  Remaining : {len(clean_table)} participants")

    if rejections:
        reasons = pd.DataFrame(rejections)["Exclusion_Reason"].value_counts()
        print("\n  Elenme sebepleri:")
        for reason, count in reasons.items():
            print(f"    {count:4d}  {reason}")

    return clean_table["ID"].tolist()


def write_cleaning_summary(starting_count, remaining_count, rejections):
    rows = [{
        "Source_File": config.SENSOR_MEASUREMENTS.name,
        "Participants_Before": starting_count,
        "Participants_After": remaining_count,
        "Participants_Excluded": len(rejections),
        "Exclusion_Reason": "TOTAL",
        "Processing_Note": ("Rule-based exclusion applied (physical and logical constraints). "
                       "IQR winsorization was deliberately NOT applied in order to "
                       "preserve the integrity of the ground-truth data."),
    }]

    # Sebep bazlı döküm
    if rejections:
        reasons = pd.DataFrame(rejections)["Exclusion_Reason"].value_counts()
        for reason, count in reasons.items():
            rows.append({
                "Source_File": config.SENSOR_MEASUREMENTS.name,
                "Participants_Before": starting_count,
                "Participants_After": remaining_count,
                "Participants_Excluded": int(count),
                "Exclusion_Reason": reason,
                "Processing_Note": "",
            })

    write_table(pd.DataFrame(rows), config.CHECK_10_CLEANING_SUMMARY)


def append_pose_cleaning_summary(starting_count, remaining_count, rejections):
    """
    It appends the pose cleaning summary to the existing cleaning summary file.
    """
    rows = [{
        "Stage": "2. Pose and kinematic quality control",
        "Source_File": "raw_data/pose/*/AlphaPose JSON",
        "Participants_Before": starting_count,
        "Participants_After": remaining_count,
        "Participants_Excluded": starting_count - remaining_count,
        "Exclusion_Reason": "ALL REASONS (stage total)",
        "Processing_Note": ("Atomicity rule applied: a participant is removed if ANY "
                            "of the four required trials fails pose cleaning, gait "
                            "cycle detection or the anatomical knee-flexion check."),
    }]

    if rejections:
        # Sebepleri kategoriye indir (deneme adi cikarilir), yoksa liste cok uzar
        labels = []
        for record in rejections:
            reason = record.get("Exclusion_Reason", "Unknown")
            labels.append(reason.split(" (")[0])
        for reason, count in pd.Series(labels).value_counts().items():
            rows.append({
                "Stage": "2. Pose and kinematic quality control",
                "Source_File": "raw_data/pose/*/AlphaPose JSON",
                "Participants_Before": "",
                "Participants_After": "",
                "Participants_Excluded": int(count),
                "Exclusion_Reason": reason,
                "Processing_Note": "Breakdown of the stage total above.",
            })

    new_rows = pd.DataFrame(rows)
    path = config.CHECK_10_CLEANING_SUMMARY
    if path.exists():
        existing = read_table(path)
        # Eski surumde Stage sutunu yoktu; varsa 1. asama olarak etiketle
        if "Stage" not in existing.columns:
            existing.insert(0, "Stage", "1. Sensor quality control")
        else:
            # 2. asama satirlari varsa temizle (tekrar calistirmada birikmesin)
            existing = existing[~existing["Stage"].astype(str).str.startswith("2.")]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    # Sutun sirasi sabit: Stage en basta olmali
    order = ["Stage", "Source_File", "Participants_Before", "Participants_After",
             "Participants_Excluded", "Exclusion_Reason", "Processing_Note"]
    combined = combined[[c for c in order if c in combined.columns]]

    # Sayi sutunlarindaki bos degerler NaN yerine bos metin olarak gorunsun
    for column in ["Participants_Before", "Participants_After"]:
        if column in combined.columns:
            combined[column] = (combined[column]
                                .apply(lambda v: "" if pd.isna(v) or v == "" else int(float(v))))

    write_table(combined, path)

def clean_all_pose_files(patient_ids):
    """
    It cleans the mandatory videos of each patient. 
    If any fails, the patient is excluded.
    """
    title("B) POSE FILE CLEANING")

    accepted_patients, rejections = [], []

    for number, patient_id in enumerate(patient_ids, start=1):
        cleaned_trials, failure_reason = [], None

        for condition in config.WALKING_CONDITIONS:
            for trial in config.REQUIRED_TRIALS:
                json_path = raw_pose_json_path(patient_id, condition, trial)

                if not json_path.exists():
                    failure_reason = f"Video file missing ({condition} {trial})"
                    break

                cleaned = clean_trial(json_path, condition)
                if cleaned is None:
                    failure_reason = f"Failed pose cleaning ({condition} {trial})"
                    break

                cleaned_trials.append((cleaned, condition, trial))
            if failure_reason:
                break

        if failure_reason:
            rejections.append({"Participant_ID": patient_id,
                               "Exclusion_Reason": failure_reason})
            print(f"  [{number}/{len(patient_ids)}] {patient_id} EXCLUDED - {failure_reason}")
            continue

        for cleaned, condition, trial in cleaned_trials:
            save_pose_table(cleaned, clean_pose_path(patient_id, condition, trial))
        accepted_patients.append(patient_id)

        if number % 25 == 0:
            print(f"  ... {number}/{len(patient_ids)} participants scanned "
                  f"({len(accepted_patients)} kabul, {len(rejections)} red)")

    add_rejected_patients(rejections)
    print(f"\n  Passed pose cleaning: {len(accepted_patients)} participants")
    return accepted_patients


def main():
    config.create_folders()
    if not files_ready([config.SENSOR_MEASUREMENTS, config.BASELINE_ESTIMATES,
                        config.PARTICIPANT_INFO]):
        return

    title("STEP 1 - DATA CLEANING")

    plot_sensor_overview(config.SENSOR_MEASUREMENTS)
    patient_ids = clean_all_sensor_records()
    clean_all_pose_files(patient_ids)

    print("\n[DONE] Next step: python main.py")


if __name__ == "__main__":
    main()
