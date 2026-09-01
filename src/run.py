import argparse
from pathlib import Path

from dotenv import load_dotenv

from core.config import load_config
from core.constants import PIPELINE_STAGES, PROJECT_ROOT
from pipeline.adaptation_batch import run_task_adaptation_panel, task_adaptation_variants
from pipeline.batch import run_alignment_panel
from pipeline.config import materialize_stage
from pipeline.materials import validate_material_preparation
from pipeline.panel import select_task, task_variants
from pipeline.runner import run_stage
from probe_transfer.alignment.contrasts import validate_contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stage of a configured research study.")
    parser.add_argument("config", type=Path, help="Path to a study YAML configuration.")
    parser.add_argument("stage", choices=PIPELINE_STAGES, help="Pipeline stage to run.")
    parser.add_argument("--model", help="Model key for preflight, extraction, or symmetry workers.")
    parser.add_argument("--task", help="Task key within a configured multi-task panel.")
    parser.add_argument("--fit", help="Label-free fit condition for panel alignment.")
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Run an alignment panel, optionally preparing its configured materials first.",
    )
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
    if args.panel and (
        args.stage != "align" or args.task or args.fit or args.model or args.publish_only
    ):
        parser.error("--panel requires align without task, fit, model, or publish-only selectors.")
    if args.fit is not None and (args.stage != "align" or args.task is None):
        parser.error("--fit requires the align stage and --task.")
    if args.panel:
        validate_material_preparation(study)
        validate_contrasts(study)
    if args.validate_only:
        if args.panel and study.get("execution", {}).get("panel_mode") == "task_adaptation":
            variants = task_adaptation_variants(study)
        else:
            variants = (
                task_variants(study, args.stage)
                if args.task is None
                else [select_task(study, args.task, args.fit)]
            )
        for variant in variants:
            materialize_stage(variant, args.stage)
        print(f"Validated {study['name']}:{args.stage}")
        return
    if args.panel:
        if study.get("execution", {}).get("panel_mode") == "task_adaptation":
            run_task_adaptation_panel(study, args.config)
        else:
            run_alignment_panel(study, args.config)
        return
    study = select_task(study, args.task, args.fit)
    run_stage(study, args.stage, model=args.model, publish_only=args.publish_only)


if __name__ == "__main__":
    main()
