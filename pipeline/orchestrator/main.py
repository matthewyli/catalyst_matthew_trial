from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core.loader import ToolLoader
from .core.pipeline import Pipeline
from .core.weights import WeightManager

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "tools_registry.yaml"
WEIGHTS_PATH = PACKAGE_ROOT / "state" / "weights.json"


def build_pipeline() -> Pipeline:
    loader = ToolLoader(CONFIG_PATH)
    registry = loader.load()
    manager = WeightManager(WEIGHTS_PATH)
    return Pipeline(registry, manager)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run orchestrator pipeline on a natural language prompt.")
    parser.add_argument("prompt", type=str, help="Strategy prompt to analyze.")
    parser.add_argument("--asset", type=str, default=None, help="Explicit asset symbol override.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    pipeline = build_pipeline()
    result = pipeline.run(args.prompt, asset=args.asset)
    payload = result.as_dict()
    output = json.dumps(payload, indent=2 if args.json else None)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
