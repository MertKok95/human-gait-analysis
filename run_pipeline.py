"""
run_pipeline.py - IT OPERATES THE ENTIRE PIPELINE IN SEQUENCE.
"""

import argparse
import subprocess
import sys
import time

import config

STEPS = [
    ("Data Cleaning", "pre_data_cleaning.py",
     [config.CHECK_2_CLEAN_SENSOR], True),

    ("Feature Extraction", "main.py",
     [config.CHECK_3_GAIT_CYCLES, config.CHECK_4_TRIAL_MEASURES], True),

    ("Model Data Set", "ML/prepare_ml_dataset.py",
     [config.CHECK_5_MODEL_INPUT], True),

    ("Model Training", "ML/train_models.py",
     [config.CHECK_6_PREDICTIONS, config.CHECK_7_STABILITY], True),

    ("Results Evaluation", "ML/evaluate_results.py",
     [config.TABLE_1_SAMPLE_CHAIN, config.TABLE_2_MAIN_RESULTS,
      config.TABLE_3_PER_PARTICIPANT], True),

    ("Reliability Analysis", "ML/reliability_analysis.py",
     [config.TABLE_4_METHOD_COMPARISON, config.TABLE_5_MEASUREMENT_RELIABILITY,
      config.TABLE_7_INFORMATION_CEILING], True),

    ("Figure Generation", "visualization/make_figures.py",
     [config.FIGURE_FOLDER / config.BASELINE_ERROR_FIGURE], True),

    ("SHAP", "ML/explain_model.py",
     [config.TABLE_6_FEATURE_IMPORTANCE], False),
]


def list_steps():
    print("\nPipeline steps:\n")
    for number, (name, script, outputs, required) in enumerate(STEPS, start=1):
        print(f"  {number}. {name:32s} {script}")
        for output in outputs:
            print(f"       -> {output.name}")
    print()


def missing_outputs(outputs, started_at):
    """Missing outputs check"""
    problems = []
    for output in outputs:
        if not output.exists():
            problems.append(f"{output.name} was not created")
        elif output.stat().st_mtime < started_at - 1:
            problems.append(f"{output.name} was not updated (stale file present)")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="Run from this step onwards")
    parser.add_argument("--only", type=int, nargs="+",
                        help="Run only the given steps")
    parser.add_argument("--list", action="store_true", help="List the steps")
    arguments = parser.parse_args()

    if arguments.list:
        list_steps()
        return

    selected = arguments.only or range(arguments.start, len(STEPS) + 1)
    skipped = []
    pipeline_start = time.time()

    for number in selected:
        if not 1 <= number <= len(STEPS):
            print(f"[WARNING] Invalid step number skipped: {number}")
            continue

        name, script, outputs, required = STEPS[number - 1]
        print(f"\n{'=' * 70}")
        print(f">>> ADIM {number}/{len(STEPS)}: {name}")
        print(f"    {script}")
        print("=" * 70)

        step_start = time.time()
        result = subprocess.run([sys.executable, script], cwd=config.ROOT)

        if result.returncode != 0:
            if not required:
                print(f"\n[WARNING] Step {number} ({name}) could not complete, skipped.")
                skipped.append(number)
                continue
            print(f"\n[FAILED] Step {number} ({script}) hata verdi.")
            print(f"            After fixing: python run_pipeline.py --from {number}")
            sys.exit(1)

        problems = missing_outputs(outputs, step_start)
        if problems and not required:
            print(f"\n[WARNING] Step {number} ({name}) skipped - no output produced.")
            print("        This step is optional; the pipeline continues.")
            skipped.append(number)
            continue
        if problems:
            print(f"\n[FAILED] Step {number} finished without error but produced no output:")
            for problem in problems:
                print(f"            - {problem}")
            print("            Its input file was probably missing.")
            print(f"            After fixing: python run_pipeline.py --from {number}")
            sys.exit(1)

        print(f"[DONE] {name} — {time.time() - step_start:.1f} seconds")

    from helper import write_file_guide
    write_file_guide()

    print(f"\n{'=' * 70}")
    print(f"[FINISHED] Total time: {(time.time() - pipeline_start) / 60:.1f} minutes")
    print("=" * 70)
    show_outputs()


def show_outputs():
    """Lists every produced file so that no output is overlooked."""
    print("\nTHESIS TABLES")
    print(f"  {config.THESIS_TABLES}")
    for path in sorted(config.THESIS_TABLES.glob("*.csv")):
        print(f"    + {path.name}")

    print("\nVALIDATION FILES (for checking a specific number)")
    print(f"  {config.VALIDATION_TABLES}")
    for path in sorted(config.VALIDATION_TABLES.glob("*.csv")):
        print(f"    + {path.name}")

    print("\nFIGURES")
    print(f"  {config.FIGURE_FOLDER}")
    for folder in sorted(p for p in config.FIGURE_FOLDER.iterdir() if p.is_dir()):
        count = len(list(folder.glob("*.png")))
        print(f"    + {folder.name}/  ({count} figures)")
    for path in sorted(config.FIGURE_FOLDER.glob("*.png")):
        print(f"    + {path.name}")


if __name__ == "__main__":
    main()
