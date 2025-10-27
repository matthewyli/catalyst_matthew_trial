from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tools import default_tools
from .keyword_config import (
    ASSET_ALIASES,
    BLOCKCHAIN_ALIASES,
    TIMEFRAME_KEYWORDS,
    TOOL_KEYWORDS,
)
from .keyword_detector import KeywordDetector
from .strategy_pipeline import StrategyPipeline
from .tool_registry import ToolRegistry
from .usage_tracker import UsageTracker

DEFAULT_ALPHA = 0.7
DEFAULT_HALF_LIFE_HOURS = 24.0
DEFAULT_BLEND_MODE = "linear"
DEFAULT_LLM_WEIGHT = 1.0

def _build_detector() -> KeywordDetector:
    return KeywordDetector(
        tool_keywords=TOOL_KEYWORDS,
        asset_aliases=ASSET_ALIASES,
        blockchain_aliases=BLOCKCHAIN_ALIASES,
        timeframe_keywords=TIMEFRAME_KEYWORDS,
        llm_weight=_resolve_llm_weight(),
        weights_path=_resolve_keyword_weights_path(),
    )
def _resolve_alpha() -> float:
    raw = os.getenv("PIPELINE_KEYWORD_ALPHA")
    if not raw:
        return DEFAULT_ALPHA
    try:
        val = float(raw)
        return min(max(val, 0.0), 1.0)
    except ValueError:
        return DEFAULT_ALPHA
def _resolve_half_life() -> Optional[float]:
    raw = os.getenv("PIPELINE_USAGE_HALF_LIFE_HOURS")
    if raw is None or raw.strip() == "":
        return DEFAULT_HALF_LIFE_HOURS
def _resolve_blend_mode() -> str:
    mode = os.getenv("PIPELINE_BLEND_MODE", DEFAULT_BLEND_MODE).lower()
    if mode in {"linear", "sqrt", "softmax"}:
        return mode
    return DEFAULT_BLEND_MODE
def _resolve_llm_weight() -> float:
    raw = os.getenv("PIPELINE_LLM_WEIGHT")
    if not raw:
        return DEFAULT_LLM_WEIGHT
    try:
        val = float(raw)
        return max(0.0, val)
    except ValueError:
        return DEFAULT_LLM_WEIGHT
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        return DEFAULT_HALF_LIFE_HOURS
def _resolve_keyword_weights_path() -> Optional[Path]:
    raw = os.getenv("PIPELINE_KEYWORD_WEIGHTS_PATH")
    if raw:
        path = Path(raw)
        return path
    default = Path(__file__).resolve().parent / "keyword_weights.json"
    return default if default.exists() else None
def build_pipeline(*, usage_path: Optional[Path] = None) -> StrategyPipeline:
    detector = _build_detector()
    registry = ToolRegistry()
    registry.bulk_register(default_tools())
    usage_file = usage_path or Path(__file__).resolve().parent.parent / "tool_usage.json"
    tracker = UsageTracker(usage_file, decay_half_life_hours=_resolve_half_life())
    return StrategyPipeline(
        registry=registry,
        detector=detector,
        tracker=tracker,
        keyword_weight_alpha=_resolve_alpha(),
        blend_mode=_resolve_blend_mode(),
    )
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a crypto strategy from a natural-language thesis.")
    parser.add_argument("--prompt", type=str, help="Prompt text describing the strategy thesis.")
    parser.add_argument("--prompt-file", type=Path, help="Path to a file containing the prompt.")
    parser.add_argument("--asset", type=str, help="Optional explicit asset ticker override.")
    parser.add_argument("--json", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)
    prompt: Optional[str] = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        parser.error("Provide --prompt or --prompt-file.")
    pipeline = build_pipeline()
    output = pipeline.run(prompt, asset=args.asset)
    payload = output.as_dict()
    dumps = json.dumps(payload, indent=2 if args.json else None)
    print(dumps)
    return 0
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
