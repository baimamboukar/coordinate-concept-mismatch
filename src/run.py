import argparse
from pathlib import Path

from dotenv import load_dotenv

from core.config import load_config
from core.constants import PIPELINE_STAGES, PROJECT_ROOT
from pipeline.config import materialize_stage
from pipeline.panel import select_task, task_variants
from pipeline.runner import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stage of a configured research study.")
    parser.add_argument("config", type=Path, help="Path to a study YAML configuration.")
    parser.add_argument("stage", choices=PIPELINE_STAGES, help="Pipeline stage to run.")
    parser.add_argument("--model", help="Model key for preflight, extraction, or symmetry workers.")
    parser.add_argument("--task", help="Task key within a configured multi-task panel.")
    parser.add_argument("--fit", help="Label-free fit condition for panel alignment.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--validate-only",
        action="store_true",
        help="Compose and validate the selected stage without executing it.",
    )
    modes.add_argument(
        "--publish-only",
        action="store_true",
        help="Retry worker-to-Hugging-Face publication without recomputing the stage.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    study = load_config(args.config)
    if args.fit is not None and (args.stage != "align" or args.task is None):
        parser.error("--fit requires the align stage and --task.")
    if args.validate_only:
        variants = (
            task_variants(study, args.stage)
            if args.task is None
            else [select_task(study, args.task, args.fit)]
        )
        for variant in variants:
            materialize_stage(variant, args.stage)
        print(f"Validated {study['name']}:{args.stage}")
        return
    study = select_task(study, args.task, args.fit)
    run_stage(study, args.stage, model=args.model, publish_only=args.publish_only)


if __name__ == "__main__":
    main()
