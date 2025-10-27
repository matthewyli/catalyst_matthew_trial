from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cli import _build_detector
from pipeline.strategy_pipeline import StrategyPipeline
from pipeline.tool_registry import ToolRegistry
from pipeline.usage_tracker import UsageTracker
from tools import default_tools

os.environ.setdefault("PIPELINE_DEBUG", "false")


def load_library(path: Path) -> List[Dict[str, str]]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list):
        raise ValueError("Prompt library must be a list")
    for entry in prompts:
        if "prompt" not in entry:
            raise ValueError(f"Invalid entry without prompt: {entry}")
        entry.setdefault("id", entry["prompt"][:40].replace(" ", "_"))
    return prompts


def build_pipeline_for(alpha: float, blend_mode: str, usage_path: Path) -> StrategyPipeline:
    registry = ToolRegistry()
    registry.bulk_register(default_tools())

    detector = _build_detector()
    usage_tracker = UsageTracker(usage_path, decay_half_life_hours=24.0)

    return StrategyPipeline(
        registry=registry,
        detector=detector,
        tracker=usage_tracker,
        keyword_weight_alpha=alpha,
        blend_mode=blend_mode,
    )


def evaluate_combo(
    prompts: Iterable[Dict[str, str]],
    alpha: float,
    blend_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    usage_path = output_dir / f"usage_alpha_{alpha}_mode_{blend_mode}.json"
    pipeline = build_pipeline_for(alpha, blend_mode, usage_path)

    combo_dir = output_dir / f"alpha_{alpha}_mode_{blend_mode}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    prompt_results: List[Dict[str, object]] = []
    alert_counts: Dict[str, int] = {}
    top_match = 0

    for entry in prompts:
        prompt_id = entry.get("id")
        prompt_text = entry["prompt"]
        asset = entry.get("asset")

        start = time.perf_counter()
        output = pipeline.run(prompt_text, asset=asset)
        elapsed = time.perf_counter() - start

        router = output.metadata.get("router", {})
        tool_ranking = router.get("tool_ranking", [])
        executed = router.get("executed_tools", [])
        executed_names = {rec.get("tool") for rec in executed}
        top_tool = tool_ranking[0] if tool_ranking else None
        if top_tool and top_tool in executed_names:
            top_match += 1

        for alert in output.phase_alerts:
            alert_counts[alert] = alert_counts.get(alert, 0) + 1

        (combo_dir / f"{prompt_id}_summary.json").write_text(
            json.dumps(output.as_dict(), indent=2), encoding="utf-8"
        )

        prompt_results.append(
            {
                "id": prompt_id,
                "prompt": prompt_text,
                "alpha": alpha,
                "blend_mode": blend_mode,
                "duration_sec": elapsed,
                "top_tool": top_tool,
                "executed_tools": list(executed_names),
                "alerts": output.phase_alerts,
            }
        )

    aggregate = {
        "alpha": alpha,
        "blend_mode": blend_mode,
        "n_prompts": len(prompt_results),
        "top_match_rate": top_match / len(prompt_results) if prompt_results else 0.0,
        "alerts": alert_counts,
        "entries": prompt_results,
    }

    (combo_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune router blending parameters using the prompt library.")
    parser.add_argument("--library", type=Path, default=Path("prompts/library.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/router_tuning"))
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.3,0.5,0.7,0.9",
        help="Comma-separated keyword alpha values to test.",
    )
    parser.add_argument(
        "--blend-modes",
        type=str,
        default="linear,sqrt,softmax",
        help="Comma-separated blend modes to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_library(args.library)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    blend_modes = [mode.strip().lower() for mode in args.blend_modes.split(",") if mode.strip()]

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "timestamp": timestamp,
        "alphas": alphas,
        "blend_modes": blend_modes,
        "results": [],
    }

    for alpha, mode in itertools.product(alphas, blend_modes):
        aggregate = evaluate_combo(prompts, alpha, mode, run_dir)
        summary["results"].append(aggregate)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
