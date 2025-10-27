from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cli import build_pipeline  # noqa: E402

os.environ.setdefault("PIPELINE_DEBUG", "false")


def load_library(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Prompt library must be a list of prompt objects.")
    for entry in data:
        if "prompt" not in entry:
            raise ValueError(f"Library entry missing prompt: {entry}")
        entry.setdefault("id", entry["prompt"][:40].replace(" ", "_"))
    return data


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_library(library: List[Dict[str, str]], output_root: Path) -> Dict[str, object]:
    pipeline = build_pipeline()
    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
    run_dir = output_root / "library" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    entries_summary: List[Dict[str, object]] = []

    for entry in library:
        prompt_id = entry.get("id")
        prompt_text = entry["prompt"]
        asset = entry.get("asset")
        start = time.perf_counter()
        output = pipeline.run(prompt_text, asset=asset)
        elapsed = time.perf_counter() - start

        entry_dir = run_dir / prompt_id
        entry_dir.mkdir(parents=True, exist_ok=True)
        write_json(entry_dir / "summary.json", output.as_dict())
        write_json(entry_dir / "tool_runs.json", output.tool_runs)

        entries_summary.append(
            {
                "id": prompt_id,
                "description": entry.get("description"),
                "asset": output.primary_asset,
                "duration_sec": elapsed,
                "recommendation": output.recommendation,
                "score": output.score,
                "alerts": output.phase_alerts,
                "detected_keywords": output.detected_keywords,
            }
        )

    aggregated = {
        "timestamp": timestamp,
        "n_prompts": len(library),
        "entries": entries_summary,
    }
    write_json(run_dir / "aggregate.json", aggregated)
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt library against the strategy pipeline.")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("prompts/library.json"),
        help="Path to prompt library JSON (default: prompts/library.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory to store aggregated outputs (default: runs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_library(args.library)
    aggregated = run_library(library, args.output_dir)
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
