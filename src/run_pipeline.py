from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    key: str
    title: str
    script_name: str
    default_enabled: bool = True
    expected_outputs: tuple[str, ...] = field(default_factory=tuple)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

STEPS = [
    PipelineStep(
        key="load",
        title="Load and clean raw IOC data",
        script_name="load_data.py",
        expected_outputs=(
            "data/processed/processed_iocs.csv",
        ),
    ),
    PipelineStep(
        key="label",
        title="Apply weak labels",
        script_name="label_data.py",
        expected_outputs=(
            "data/labeled/labeled_iocs.csv",
        ),
    ),
    PipelineStep(
        key="features",
        title="Build model features",
        script_name="features.py",
        expected_outputs=(
            "data/processed/featured_iocs.csv",
        ),
    ),
    PipelineStep(
        key="train",
        title="Train family-based production model",
        script_name="train_model.py",
        expected_outputs=(
            "models/ioc_priority_family_bundle_v1.joblib",
            "reports/training_summary_family_v1.csv",
            "reports/training_summary_family_v1.txt",
        ),
    ),
    PipelineStep(
        key="score",
        title="Score latest IOCs",
        script_name="score_iocs.py",
        expected_outputs=(
            "reports/scored_iocs_family_v4.csv",
        ),
    ),
    PipelineStep(
        key="export",
        title="Export ranked analyst-facing outputs",
        script_name="export_ranked_results.py",
        expected_outputs=(
            "reports/exports/all_ranked_iocs.csv",
            "reports/exports/high_priority_iocs.csv",
            "reports/exports/medium_priority_iocs.csv",
            "reports/exports/top_100_for_review.csv",
            "reports/exports/summary.txt",
        ),
    ),
    PipelineStep(
        key="benchmark",
        title="Run family-based benchmark",
        script_name="benchmark_family_model_v2.py",
        default_enabled=False,
        expected_outputs=(
            "reports/benchmark_family_by_source_group_v3.csv",
        ),
    ),
]

STEP_MAP = {step.key: step for step in STEPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SwiftIOC ML scoring pipeline."
    )
    parser.add_argument(
        "--from-step",
        choices=[step.key for step in STEPS],
        help="Start running from this step onward.",
    )
    parser.add_argument(
        "--only",
        choices=[step.key for step in STEPS],
        help="Run only a single step.",
    )
    parser.add_argument(
        "--include-benchmark",
        action="store_true",
        help="Include the benchmark step in the normal pipeline run.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Show available pipeline steps and exit.",
    )
    return parser.parse_args()


def get_default_steps(include_benchmark: bool) -> list[PipelineStep]:
    steps = [step for step in STEPS if step.default_enabled]
    if include_benchmark:
        steps.append(STEP_MAP["benchmark"])
    return steps


def get_steps_to_run(
    from_step: str | None,
    only: str | None,
    include_benchmark: bool,
) -> list[PipelineStep]:
    if only:
        return [STEP_MAP[only]]

    selected_steps = get_default_steps(include_benchmark)

    if from_step:
        start_index = next(
            (index for index, step in enumerate(selected_steps) if step.key == from_step),
            None,
        )
        if start_index is None:
            raise ValueError(
                f"Step '{from_step}' is not in the selected pipeline path. "
                "Use --include-benchmark if you want to start from benchmark."
            )
        return selected_steps[start_index:]

    return selected_steps


def validate_scripts(steps: list[PipelineStep]) -> None:
    missing = []
    for step in steps:
        script_path = SRC_DIR / step.script_name
        if not script_path.exists():
            missing.append(str(script_path))

    if missing:
        raise FileNotFoundError(
            "The following pipeline scripts are missing:\n- " + "\n- ".join(missing)
        )


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}m {remaining:.2f}s"


def print_header() -> None:
    print("=" * 80)
    print("SwiftIOC ML Scoring Pipeline")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print()


def print_available_steps() -> None:
    print("Available steps:")
    for step in STEPS:
        default_marker = "default" if step.default_enabled else "optional"
        print(f"- {step.key:<10} {step.title} ({step.script_name}) [{default_marker}]")


def run_step(step: PipelineStep) -> float:
    script_path = SRC_DIR / step.script_name

    print("-" * 80)
    print(f"Running step : {step.key}")
    print(f"Title        : {step.title}")
    print(f"Script       : {script_path}")
    print("-" * 80)

    start_time = time.perf_counter()

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Completed    : {step.key} in {format_seconds(elapsed)}")
    print()
    return elapsed


def collect_expected_outputs(steps_run: list[PipelineStep]) -> list[Path]:
    seen: set[Path] = set()
    outputs: list[Path] = []

    for step in steps_run:
        for relative_path in step.expected_outputs:
            full_path = PROJECT_ROOT / relative_path
            if full_path not in seen:
                seen.add(full_path)
                outputs.append(full_path)

    return outputs


def print_final_summary(
    steps_run: list[PipelineStep],
    timings: list[tuple[str, float]],
) -> None:
    total_time = sum(elapsed for _, elapsed in timings)
    expected_outputs = collect_expected_outputs(steps_run)

    print("=" * 80)
    print("Pipeline finished successfully")
    print("=" * 80)
    print(f"Steps run    : {len(steps_run)}")
    print(f"Total time   : {format_seconds(total_time)}")
    print()

    print("Step timings:")
    for step_key, elapsed in timings:
        print(f"- {step_key:<10} {format_seconds(elapsed)}")
    print()

    print("Expected outputs:")
    for output_path in expected_outputs:
        status = "OK" if output_path.exists() else "MISSING"
        print(f"- [{status:<7}] {output_path}")

    print()
    print("Useful commands:")
    print(f"- Full production pipeline : {sys.executable} src/run_pipeline.py")
    print(f"- Production from score    : {sys.executable} src/run_pipeline.py --from-step score")
    print(f"- Export only              : {sys.executable} src/run_pipeline.py --only export")
    print(f"- Benchmark only           : {sys.executable} src/run_pipeline.py --only benchmark")
    print(f"- Full + benchmark         : {sys.executable} src/run_pipeline.py --include-benchmark")


def main() -> None:
    args = parse_args()

    if args.list_steps:
        print_available_steps()
        return

    if args.from_step and args.only:
        raise ValueError("Use either --from-step or --only, not both.")

    steps_to_run = get_steps_to_run(
        from_step=args.from_step,
        only=args.only,
        include_benchmark=args.include_benchmark,
    )
    validate_scripts(steps_to_run)

    print_header()
    print("Steps selected:")
    for step in steps_to_run:
        print(f"- {step.key:<10} {step.title}")
    print()

    timings: list[tuple[str, float]] = []

    try:
        for step in steps_to_run:
            elapsed = run_step(step)
            timings.append((step.key, elapsed))
    except subprocess.CalledProcessError as exc:
        print()
        print("=" * 80)
        print("Pipeline failed")
        print("=" * 80)
        print(f"Step failed  : {step.key}")
        print(f"Exit code    : {exc.returncode}")
        print(f"Command      : {' '.join(map(str, exc.cmd))}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print()
        print("=" * 80)
        print("Pipeline failed before completion")
        print("=" * 80)
        print(f"Error        : {exc}")
        sys.exit(1)

    print_final_summary(steps_to_run, timings)


if __name__ == "__main__":
    main()