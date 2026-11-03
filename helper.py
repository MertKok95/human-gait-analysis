from helper_files import (
    analysis_helper,
    calculation_helper,
    data_cleaning_helper,
    file_guide,
    file_process_helper,
    gait_cycle_helper,
    visualization_helper,
)

import config

import numpy as np
import pandas as pd


# =====================================================================
# File and data management
# =====================================================================

def raw_pose_json_path(patient_id, condition, trial):
    return file_process_helper.raw_pose_json_path(patient_id, condition, trial)


def clean_pose_path(patient_id, condition, trial):
    return file_process_helper.clean_pose_path(patient_id, condition, trial)


def measurement_path(patient_id, condition, trial):
    return file_process_helper.measurement_path(patient_id, condition, trial)


def save_pose_table(pose_table, path):
    return file_process_helper.save_pose_table(pose_table, path)


def read_pose_table(path):
    return file_process_helper.read_pose_table(path)


def load_rejected_patients():
    return file_process_helper.load_rejected_patients()


def add_rejected_patients(new_rejections):
    return file_process_helper.add_rejected_patients(new_rejections)


def build_frame_level_dataset(patient_ids, output_path):
    return file_process_helper.build_frame_level_dataset(patient_ids, output_path)


def write_file_guide():
    return file_guide.write_guide()


# =====================================================================
# DATA CLEANING
# =====================================================================

def clean_sensor_data(sensor_table, patient_ids):
    return data_cleaning_helper.clean_sensor_data(sensor_table, patient_ids)


def clean_trial(json_path, condition):
    return data_cleaning_helper.clean_trial(json_path, condition)


# =====================================================================
# BIOMECHANICAL CALCULATIONS
# =====================================================================

def measure_bone_lengths(pose_table):
    return calculation_helper.measure_bone_lengths(pose_table)


def pixel_to_cm_scale(bone_lengths, real_height_cm):
    return calculation_helper.pixel_to_cm_scale(bone_lengths, real_height_cm)


def add_cm_coordinates(pose_table, bone_lengths, scale):
    return calculation_helper.add_cm_coordinates(pose_table, bone_lengths, scale)


def remove_impossible_coordinates(pose_table):
    return calculation_helper.remove_impossible_coordinates(pose_table)


def add_ankle_distance(pose_table):
    return calculation_helper.add_ankle_distance(pose_table)


def detect_heel_strikes(pose_table, patient_id, trial_label, save_plot=True):
    return calculation_helper.detect_heel_strikes(
        pose_table, patient_id, trial_label, save_plot)


def add_cadence(pose_table):
    return calculation_helper.add_cadence(pose_table)


def add_spectral_cadence(pose_table):
    return calculation_helper.add_spectral_cadence(pose_table)


def add_velocity(pose_table):
    return calculation_helper.add_velocity(pose_table)


def add_displacement_velocity(pose_table):
    return calculation_helper.add_displacement_velocity(pose_table)


def add_joint_angles(pose_table):
    return calculation_helper.add_joint_angles(pose_table)


def add_trunk_sway(pose_table):
    return calculation_helper.add_trunk_sway(pose_table)


def add_asymmetry(pose_table):
    return calculation_helper.add_asymmetry(pose_table)


def has_impossible_knee_bend(pose_table):
    return calculation_helper.has_impossible_knee_bend(pose_table)


def fill_forward(pose_table, columns):
    return calculation_helper.fill_forward(pose_table, columns)


# =====================================================================
# GAIT CYCLES
# =====================================================================

def extract_cycles(pose_table, patient_id, trial_label):
    return gait_cycle_helper.extract_cycles(pose_table, patient_id, trial_label)


def summarise_trial(cycle_table, cadence, mean_speed,
                    displacement_speed, spectral_cadence):
    return gait_cycle_helper.summarise_trial(
        cycle_table, cadence, mean_speed, displacement_speed, spectral_cadence)


# =====================================================================
# COMMON TOOLS
# =====================================================================

def files_ready(paths):
    return analysis_helper.files_ready(paths)


def title(text):
    return analysis_helper.title(text)


def read_table(path):
    return analysis_helper.read_table(path)


def write_table(table, path, note=""):
    return analysis_helper.write_table(table, path, note)


def errors(actual_values, predicted_values):
    return analysis_helper.errors(actual_values, predicted_values)


def paired_test(reference_errors, model_errors):
    return analysis_helper.paired_test(reference_errors, model_errors)


def correlation(first_values, second_values):
    return analysis_helper.correlation(first_values, second_values)


def average_trials(trial_table, measure_columns, jacket_trials=None):
    return analysis_helper.average_trials(trial_table, measure_columns, jacket_trials)


def one_row_per_patient(condition_level_table):
    return analysis_helper.one_row_per_patient(condition_level_table)


def build_model(feature_count):
    return analysis_helper.build_model(feature_count)


def fit_and_score(features, target, seed_count=None):
    return analysis_helper.fit_and_score(features, target, seed_count)


# =====================================================================
# VISUALIZATIONS
# =====================================================================

def plot_sensor_overview(sensor_path):
    return visualization_helper.plot_sensor_overview(sensor_path)


def plot_clinical_kinematics(pose_table, cycle_table, patient_id, trial_label):
    return visualization_helper.plot_clinical_kinematics(
        pose_table, cycle_table, patient_id, trial_label)
