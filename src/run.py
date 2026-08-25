import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from core.config import load_config
from core.constants import PROJECT_ROOT
from core.reproducibility import require_process_hash_seed, seed_everything
from core.tracking import Tracker

Runner = Callable[[dict[str, Any], Tracker], None]


def resolve_runner(spec: str) -> Runner:
    module_name, function_name = spec.split(":", maxsplit=1)
    runner = getattr(import_module(module_name), function_name, None)
    if not callable(runner):
        raise TypeError(f"Runner is not callable: {spec}")
    return cast(Runner, runner)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named probe-transfer experiment.")
    parser.add_argument("config", type=Path, help="Path to a semantic .yaml configuration.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(args.config)
    if config.get("deterministic", True):
        require_process_hash_seed(config["seed"])
    seed_everything(config["seed"], config.get("deterministic", True))
    runner = resolve_runner(config["runner"])
    tracker = Tracker.start(config)

    try:
        runner(config, tracker)
    except Exception as error:
        tracker.report("Failure", f"`{type(error).__name__}`: {error}")
        tracker.finish("failed")
        raise
    tracker.finish()


if __name__ == "__main__":
    main()
