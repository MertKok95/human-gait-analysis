from pathlib import Path

# =====================================================================
# 1. FOLDER PATHS
# =====================================================================


ROOT = Path(__file__).resolve().parent

DATA_FOLDER = ROOT / "data"
RAW_POSE_FOLDER = DATA_FOLDER / "raw_data"         
OUTPUT_FOLDER = DATA_FOLDER / "outputs"

THESIS_TABLES = OUTPUT_FOLDER / "thesis_tables"
VALIDATION_TABLES = OUTPUT_FOLDER / "validation"
CLEAN_POSE_FOLDER = OUTPUT_FOLDER / "clean_pose"
FIGURE_FOLDER = OUTPUT_FOLDER / "figures"
MODEL_FOLDER = OUTPUT_FOLDER / "models"

# --- INPUTS ---
SENSOR_MEASUREMENTS = DATA_FOLDER / "gait_parameters.csv"            # ground truth
BASELINE_ESTIMATES = DATA_FOLDER / "gait_parameters_estimation.csv"  # article's predictions
PARTICIPANT_INFO = DATA_FOLDER / "patients_measures.csv"             # height, weight, gender

# --- THESIS TABLES ---
TABLE_1_SAMPLE_CHAIN = THESIS_TABLES / "Table1_Sample_Chain.csv"
TABLE_2_MAIN_RESULTS = THESIS_TABLES / "Table2_Main_Results.csv"
TABLE_3_PER_PARTICIPANT = THESIS_TABLES / "Table3_Per_Participant_Comparison.csv"
TABLE_4_METHOD_COMPARISON = THESIS_TABLES / "Table4_Method_Comparison.csv"
TABLE_5_MEASUREMENT_RELIABILITY = THESIS_TABLES / "Table5_Measurement_Reliability.csv"
TABLE_6_FEATURE_IMPORTANCE = THESIS_TABLES / "Table6_Feature_Importance.csv"
TABLE_7_INFORMATION_CEILING = THESIS_TABLES / "Table7_Information_Ceiling.csv"

# --- VALIDATION TABLES ---
CHECK_1_EXCLUDED = VALIDATION_TABLES / "Check1_Excluded_Participants.csv"
CHECK_2_CLEAN_SENSOR = VALIDATION_TABLES / "Check2_Clean_Sensor_Data.csv"
CHECK_3_GAIT_CYCLES = VALIDATION_TABLES / "Check3_Gait_Cycles.csv"
CHECK_4_TRIAL_MEASURES = VALIDATION_TABLES / "Check4_Trial_Measurements.csv"
CHECK_5_MODEL_INPUT = VALIDATION_TABLES / "Check5_Model_Dataset.csv"
CHECK_6_PREDICTIONS = VALIDATION_TABLES / "Check6_Model_Predictions.csv"
CHECK_7_STABILITY = VALIDATION_TABLES / "Check7_Model_Stability.csv"
CHECK_8_OVERFITTING = VALIDATION_TABLES / "Check8_Overfitting_Check.csv"
CHECK_9_FRAME_DATASET = VALIDATION_TABLES / "Check9_Frame_Level_Dataset.csv"
CHECK_10_CLEANING_SUMMARY = VALIDATION_TABLES / "Check10_Cleaning_Summary.csv"

# A guide showing which output was stored under what name in the old project.
FILE_GUIDE = OUTPUT_FOLDER / "00_FILE_GUIDE.csv"


# Graphics subfolders
FIGURE_SUBFOLDERS = {
    "signals": "signals",            # heel strike signals
    "kinematics": "kinematics",       # clinical kinematic curves
    "exploratory": "exploratory",      # exploratory analysis
    "skeleton": "skeleton",            # skeleton drawings
    "diagrams": "diagrams",            # method diagrams
    "importance": "feature_importance",# SHAP graphs
}

BASELINE_ERROR_FIGURE = "baseline_error.png"

def figure_folder(key):
    return FIGURE_FOLDER / FIGURE_SUBFOLDERS[key]


def create_folders():
    """It creates all the necessary folders (leaving any existing ones untouched)."""
    for folder in [THESIS_TABLES, VALIDATION_TABLES, CLEAN_POSE_FOLDER,
                   FIGURE_FOLDER, MODEL_FOLDER,
                   *[FIGURE_FOLDER / name for name in FIGURE_SUBFOLDERS.values()]]:
        folder.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 2. MEASUREMENT CONSTANTS
# =====================================================================

FRAMES_PER_SECOND = 30.0

HEEL_STRIKE_COLUMN = {"L": "left_heel_strike", "R": "right_heel_strike"}

# ugs = usual gait speed (normal walking) | fgs = fast gait speed (fast walking)
WALKING_CONDITIONS = ["ugs", "fgs"]

# WoJ = without jacket (without jacket) -> required trials
# WJ  = with jacket    (with jacket)  -> optional, reduces measurement noise
REQUIRED_TRIALS = ["WoJ_1", "WoJ_2"]
OPTIONAL_TRIALS = ["WJ_1", "WJ_2"]

# How long can a walking cycle last? (seconds)
# Typical values ​​in the literature are 0.8-1.3 seconds; anything outside of this range is considered an incorrect measurement.
MINIMUM_CYCLE_SECONDS = 0.4
MAXIMUM_CYCLE_SECONDS = 2.0

# Step and stride length limits (centimeters)
MINIMUM_STEP_LENGTH_CM = 15.0
MAXIMUM_STEP_LENGTH_CM = 150.0
MINIMUM_STRIDE_LENGTH_CM = 30.0
MAXIMUM_STRIDE_LENGTH_CM = 250.0

# Quality control thresholds
MAXIMUM_MISSING_RATIO = 0.20          # Reject if more than 20% of the frames are missing.
MAXIMUM_GAP_FRAMES = 10               # A gap longer than 10 squares cannot be filled.
BONE_LENGTH_TOLERANCE = 0.40          # Bone length is considered abnormal if it deviates 40% from the median.
MAXIMUM_KNEE_FLEXION_DEGREES = 110.0  # The knee cannot bend that much -> posture error

# =====================================================================
# 3. MODEL FEATURES AND TARGETS
# =====================================================================

# Measurements extracted from the video
VIDEO_MEASUREMENTS = [
    "Cadence",             # cadence, from interval of consecutive heel strikes
    "SpectralCadence",     # cadence, from frequency analysis (NEW, more reliable)
    "MeanSpeed",           # velocity, step length x cadence
    "DisplacementSpeed",   # velocity, from trunk displacement (NEW)
    "StepLength_Mean",     # average step length
    "StrideLength_Mean",   # average stride length
    "StrideDuration_Mean", # average cycle time
    "KneeROM_Mean",        # ROM = range of motion (knee range of motion)
]

# Mathematically derived step length from speed and cadence
DERIVED_MEASUREMENTS = ["DerivedStep"]

DEMOGRAPHIC_FEATURES = ["Height", "Sex"]   # height (cm), gender

# Values measured by the sensor, which the model aims to predict
MAIN_TARGETS = [        # The baseline camera method also produces 8 parameters.
    "Step_UGS", "Stride_UGS", "Cadence_UGS", "Velocity_UGS",
    "Step_FGS", "Stride_FGS", "Cadence_FGS", "Velocity_FGS",
]

# Support phase parameters that baseline did not produce.
SUPPORT_PHASE_TARGETS = [   
    "MonoSP_UGS", "BiSP_UGS", "MonoSP_FGS", "BiSP_FGS",
]
ALL_TARGETS = MAIN_TARGETS + SUPPORT_PHASE_TARGETS

# Support phase measurements: not used in the model, only for reporting
SUPPORT_PHASE_MEASUREMENTS = [
    "StanceTime_Mean", "SwingTime_Mean",
    "DoubleSupportTime_Mean", "SingleSupportTime_Mean",
    "StancePercent_Mean", "SwingPercent_Mean",
    "DoubleSupportPercent_Mean", "SingleSupportPercent_Mean",
]

# Measurements to be averaged with jacket trials.
# Increasing the number of trials from 2 to 4 reduces measurement noise.
MEASUREMENTS_AVERAGED_WITH_JACKET_TRIALS = VIDEO_MEASUREMENTS.copy()

# Physiological limits for the sensor's support phase readings (seconds)
MONO_SUPPORT_RANGE_SECONDS = (0.25, 0.60)
BILATERAL_SUPPORT_RANGE_SECONDS = (0.05, 0.40)


def model_feature_columns():
    """The complete list of 20 features used by the model."""
    measurements = VIDEO_MEASUREMENTS + DERIVED_MEASUREMENTS
    per_condition = [f"{name}_{condition}"
                     for name in measurements for condition in WALKING_CONDITIONS]
    return per_condition + DEMOGRAPHIC_FEATURES


def support_phase_columns():
    return [f"{name}_{condition}"
            for name in SUPPORT_PHASE_MEASUREMENTS for condition in WALKING_CONDITIONS]


# =====================================================================
# 4. MODEL SETTINGS
# =====================================================================

CROSS_VALIDATION_FOLDS = 5    # How many parts should the data be divided into?
RANDOM_SEED = 42              # Constant randomness for repeatability
STABILITY_SPLIT_COUNT = 20    # How many different divisions should R2 be measured in?
HOLDOUT_TEST_RATIO = 0.20     # Share allocated for memorization control.
FEATURE_COUNT_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 16]  # How many features should be selected?

# =====================================================================
# 5. OUTPUT COLUMN NAMES
# =====================================================================
# The column name in the code -> The name that will appear in the CSV.

OUTPUT_COLUMN_NAMES = {
    "Subject_ID": "Participant_ID", "Video_ID": "Trial",
    "Start_HS_Frame": "Start_Frame",         
    "End_HS_Frame": "End_Frame",
    "Opposite_HS_Frame": "Opposite_Foot_Frame",
    "TO_Frame": "Toe_Off_Frame",              
    "TO_Detection_Method": "Toe_Off_Detection_Method",
    "Cycle_Time": "Cycle_Duration_s",
    "Stance_Time": "Stance_Time_s", "Swing_Time": "Swing_Time_s",
    "Double_Support_Time": "Double_Support_Time_s",
    "Single_Support_Time": "Single_Support_Time_s",
    "Step_Length": "Step_Length_cm", "Stride_Length": "Stride_Length_cm",
    "Knee_ROM": "Knee_Range_Of_Motion_deg",
    "Spatial_Valid": "Spatial_Measures_Valid",
    "Spatial_Problem": "Spatial_Measures_Issue",
    "Support_Valid": "Support_Phase_Valid",
    "Support_Problem": "Support_Phase_Issue",
    "Total_Cycle_Count": "Total_Cycles",
    "Spatial_Valid_Cycle_Count": "Valid_Spatial_Cycles",
    "Support_Valid_Cycle_Count": "Valid_Support_Cycles",
    "Cadence": "Cadence_steps_min",
    "SpectralCadence": "Spectral_Cadence_steps_min",
    "MeanSpeed": "Speed_Step_x_Cadence_m_s",
    "DisplacementSpeed": "Speed_Displacement_m_s",
    "DerivedStep": "Derived_Step_Length_cm",
    "StepLength_Mean": "Step_Length_Mean_cm",
    "StrideLength_Mean": "Stride_Length_Mean_cm",
    "StrideDuration_Mean": "Cycle_Duration_Mean_s",
    "KneeROM_Mean": "Knee_Range_Of_Motion_Mean_deg",
    "StepLength_CV": "Step_Length_Variability_Percent", 
    "StrideLength_CV": "Stride_Length_Variability_Percent",
    "StrideDuration_CV": "Cycle_Duration_Variability_Percent",
    "StepDuration_Mean": "Step_Duration_Mean_s",
    "StanceTime_Mean": "Stance_Time_Mean_s",
    "SwingTime_Mean": "Swing_Time_Mean_s",
    "DoubleSupportTime_Mean": "Double_Support_Time_Mean_s",
    "SingleSupportTime_Mean": "Single_Support_Time_Mean_s",
    "StancePercent_Mean": "Stance_Percent_Mean",
    "SwingPercent_Mean": "Swing_Percent_Mean",
    "DoubleSupportPercent_Mean": "Double_Support_Percent_Mean",
    "SingleSupportPercent_Mean": "Single_Support_Percent_Mean",
    "StanceSymmetryIndex_Pct": "Stance_Symmetry_Index_Percent",
    "Height": "Height_cm",
    "R2_Deviation": "R2_SD", "R2_Lowest": "R2_Min", "R2_Highest": "R2_Max",
}


INPUT_COLUMN_NAMES = {written: code
                      for code, written in OUTPUT_COLUMN_NAMES.items()}

# =====================================================================
# 6. DISPLAY LABELS
# =====================================================================

MEASUREMENT_LABELS = {
    # targets
    "Step": "Step Length", "Stride": "Stride Length",
    "Cadence": "Cadence", "Velocity": "Gait Speed",
    "MonoSP": "Single Support Time", "BiSP": "Double Support Time",
    # video measurements
    "SpectralCadence": "Cadence (spectral)",
    "DisplacementSpeed": "Speed (displacement)",
    "MeanSpeed": "Speed (step x cadence)",
    "DerivedStep": "Step Length (derived)",
    "StepLength_Mean": "Step Length (measured)",
    "StrideLength_Mean": "Stride Length (measured)",
    "StrideDuration_Mean": "Cycle Duration",
    "KneeROM_Mean": "Knee Range of Motion",
    "Height": "Height", "Sex": "Sex",
}

CONDITION_LABELS = {"UGS": "usual gait", "FGS": "fast gait",
                    "ugs": "usual gait", "fgs": "fast gait"}

MEASUREMENT_UNITS = {"Step": "cm", "Stride": "cm", "Velocity": "m/s",
                     "Cadence": "steps/min", "MonoSP": "s", "BiSP": "s"}


def readable_name(column_name):
    """
    Converts the technical column name to a readable label.
      'DisplacementSpeed_ugs' -> 'Speed (displacement) - usual gait'
      'Step_FGS'              -> 'Step Length - fast gait'
    """
    for suffix in ["_ugs", "_fgs", "_UGS", "_FGS"]:
        if column_name.endswith(suffix):
            base = column_name[: -len(suffix)]
            return (f"{MEASUREMENT_LABELS.get(base, base)} - "
                    f"{CONDITION_LABELS[suffix[1:]]}")
    return MEASUREMENT_LABELS.get(column_name, column_name)


def unit_of(target_name):
    """'Step_UGS' -> 'cm'"""
    return MEASUREMENT_UNITS.get(target_name.split("_")[0], "")


def clean_pose_file(participant_id, condition, trial):
    """Returns the path to the cleaned pose file."""
    return (CLEAN_POSE_FOLDER / condition
            / f"{participant_id}_{condition}_{trial}_clean_pose.csv")
