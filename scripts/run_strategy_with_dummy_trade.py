from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.cli import build_pipeline  # noqa: E402
from pipeline.strategy_pipeline import StrategyPipeline  # noqa: E402
from scripts.execute_strategy import build_strategy_spec, execute_strategy, find_latest_run  # noqa: E402


SYMBOL_TO_SLUG: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ARB": "arbitrum",
    "OP": "optimism",
    "MATIC": "polygon",
    "AVAX": "avalanche",
    "BNB": "bnb-chain",
}


def _resolve_asset_slug(asset: str | None) -> str | None:
    if not asset:
        return None
    asset = asset.strip()
    if not asset:
        return None
    return SYMBOL_TO_SLUG.get(asset.upper(), asset.lower())


def _summarise_strict_values(
    tool_runs: List[Dict[str, object]]
) -> Tuple[List[Tuple[str, float, Dict[str, object]]], Dict[str, float]]:
    summary: List[Tuple[str, float, Dict[str, object]]] = []
    mapping: Dict[str, float] = {}
    for item in tool_runs:
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        if "value" not in payload:
            continue
        value = payload.get("value")
        if not isinstance(value, (int, float)):
            continue
        raw = payload.get("raw")
        raw_payload = raw if isinstance(raw, dict) else payload
        name = str(item.get("name"))
        numeric = float(value)
        summary.append((name, numeric, raw_payload))
        mapping[name] = numeric
    return summary, mapping


def run_strategy(
    prompt: str,
    asset: Optional[str],
    strict: bool,
    pipeline: Optional[StrategyPipeline] = None,
) -> Dict[str, Any]:
    prev_strict = os.environ.get("PIPELINE_STRICT_IO")
    if strict:
        os.environ["PIPELINE_STRICT_IO"] = "true"

    try:
        pipeline_obj = pipeline or build_pipeline()
        output = pipeline_obj.run(prompt, asset=asset)
    finally:
        if strict:
            if prev_strict is None:
                os.environ.pop("PIPELINE_STRICT_IO", None)
            else:
                os.environ["PIPELINE_STRICT_IO"] = prev_strict

    strict_values, strict_mapping = _summarise_strict_values(output.tool_runs)
    timestamp = datetime.now(timezone.utc).isoformat()
    if strict_values:
        aggregate = sum(value for _, value, _ in strict_values) / len(strict_values)
    else:
        aggregate = output.score

    decision = "BUY" if aggregate > 0 else "SELL" if aggregate < 0 else "HOLD"
    print(f"[strategy] timestamp={timestamp}")
    print(f"[strategy] prompt=\"{prompt}\"")
    print(f"[strategy] aggregate_score={aggregate:.4f} decision={decision}")

    router_info = output.metadata.get("router", {}) if isinstance(output.metadata, dict) else {}
    tool_ranking = router_info.get("tool_ranking")
    executed_tools = router_info.get("executed_tools")
    if tool_ranking:
        print(f"[strategy] router_tool_ranking={tool_ranking}")
    if executed_tools:
        print(f"[strategy] executed_tools={executed_tools}")
    phase_notes = output.metadata.get("context", {}).get("phase_outputs", {})
    if phase_notes:
        notes_preview = {phase: list(payload.keys()) for phase, payload in phase_notes.items()}
        print(f"[strategy] phase_outputs_keys={notes_preview}")
    if strict_mapping:
        print(
            "[strategy] strict_tool_values="
            + ", ".join(f"{name}={value:.4f}" for name, value in strict_mapping.items())
        )

    latest_run_dir = find_latest_run(Path("runs"))
    spec = build_strategy_spec(latest_run_dir)
    spec_path = Path(latest_run_dir) / "strategy_spec.json"
    print(f"[strategy] strategy_spec={spec_path}")
    execute_strategy(spec, iterations=1, interval=0.0)

    return {
        "timestamp": timestamp,
        "output": output,
        "strict_values": strict_values,
        "strict_map": strict_mapping,
        "aggregate": aggregate,
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pipeline in strict I/O mode and simulate a trade.")
    parser.add_argument("--prompt", required=True, help="Natural language strategy prompt.")
    parser.add_argument("--asset", help="Optional explicit asset ticker.")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict I/O mode (defaults to enabled for this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_strategy(prompt=args.prompt, asset=args.asset, strict=not args.no_strict)


if __name__ == "__main__":
    main()
