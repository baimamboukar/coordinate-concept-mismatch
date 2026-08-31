import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.constants import PROJECT_NAME, PROJECT_ROOT


@dataclass
class Tracker:
    experiment: str
    report_path: Path | None
    wandb_run: Any = None

    @classmethod
    def start(cls, config: dict[str, Any], root: Path = PROJECT_ROOT) -> "Tracker":
        experiment = config["name"]
        stage = config.get("stage", "run")

        tracking = config.get("tracking", {})
        wandb_enabled = bool(tracking.get("wandb", False))
        if config.get("training", False) and not wandb_enabled:
            raise ValueError("Training experiments must enable W&B tracking.")

        report_path = None
        if tracking.get("local_report", True):
            report_dir = root / "logs" / experiment
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "report.md"

        wandb_run = None
        if wandb_enabled:
            import wandb

            labels = [
                config.get("study"),
                config.get("task"),
                stage,
                config.get("fit_condition"),
                config.get("worker_model"),
            ]
            run_name = "-".join(str(label) for label in labels if label)
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", PROJECT_NAME),
                entity=os.getenv("WANDB_ENTITY") or None,
                name=run_name,
                group=experiment,
                config=config,
                mode=tracking.get("mode"),
            )

        tracker = cls(
            experiment=experiment,
            report_path=report_path,
            wandb_run=wandb_run,
        )
        if report_path is not None and not report_path.exists():
            report_path.write_text(f"# {experiment}\n\n")
        return tracker

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def report(self, heading: str, body: str) -> None:
        if self.report_path is None:
            return
        with self.report_path.open("a") as handle:
            handle.write(f"## {heading}\n\n{body.strip()}\n\n")

    def finish(self, status: str = "completed") -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=0 if status == "completed" else 1)
