from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseTool, ToolContext, ToolExecutionError, ToolResult

__TOOL_META__ = {
    "name": "asknews_impact",
    "module": "tools.asknews_tool",
    "object": "AskNewsImpactTool",
    "version": "1.0",
    "description": "Scores short-horizon crypto news impact using AskNews + AEI.",
    "author": "auto",
    "keywords": [
        "news",
        "headline",
        "sentiment",
        "impact",
        "event",
        "asknews",
        "narrative",
    ],
    "phases": ["data_gather", "signal_generation"],
    "outputs": ["impact_score", "direction", "confidence"],
}

try:
    from .legacy import asknews_framework  # noqa: F401  # pragma: no cover - ensure dependency is discoverable
    from .legacy.asknews_framework import build_engine_from_env
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError("asknews_framework module is required for AskNewsImpactTool") from exc


class AskNewsImpactTool(BaseTool):
    """Surface short-horizon impact signals from AskNews + AEI stack."""

    name = "asknews_impact"
    description = "Scores recent news flow using the AEI engine."
    keywords = frozenset(
        {
            "news",
            "headline",
            "sentiment",
            "narrative",
            "event",
            "impact",
            "asknews",
        }
    )

    def __init__(self, *, provider_override: Optional[str] = None, default_window_min: int = 90) -> None:
        super().__init__()
        self.provider_override = provider_override
        self.default_window_min = default_window_min

    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        asset = (context.asset or "").upper()
        if not asset:
            raise ToolExecutionError("AskNewsImpactTool requires an asset symbol in the prompt or metadata.")

        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        strict_mode = bool(options.get("strict_io"))

        window_min = int(context.metadata.get("window_min", self.default_window_min))
        min_score = context.metadata.get("min_score")
        max_items = context.metadata.get("max_items")

        try:
            engine = build_engine_from_env(provider_override=self.provider_override)
            results = engine.run(
                assets=[asset],
                window_min=window_min,
                min_score=min_score,
                max_items=max_items,
                market_ref=None,
            )
        except Exception as exc:  # pragma: no cover - runtime path
            raise ToolExecutionError(f"AskNews AEI execution failed: {exc}") from exc

        payload: Dict[str, Any]
        strict_value = 0.0
        if not results:
            summary = f"No high-impact news found for {asset} in the last {window_min} minutes."
            raw_payload = {
                "asset": asset,
                "window_min": window_min,
                "result": None,
            }
        else:
            block = results[0]
            impact = float(block.get("impact_score", 0.0))
            direction = block.get("dir", "flat")
            confidence = float(block.get("confidence", 0.0))
            n_articles = int(block.get("n_articles", 0))
            summary = (
                f"{asset} news impact {impact:.2f} ({direction}, confidence {confidence:.2%}) "
                f"from {n_articles} articles."
            )
            raw_payload = {
                "asset": asset,
                "window_min": window_min,
                "impact_score": impact,
                "direction": direction,
                "confidence": confidence,
                "n_articles": n_articles,
                "components": block.get("components", {}),
                "top_reasons": block.get("top_reasons", []),
                "ops_flags": block.get("ops_flags", {}),
                "research_summary": block.get("research_summary"),
            }
            strict_value = impact

        payload = {"value": strict_value, "raw": raw_payload} if strict_mode else raw_payload
        weight = float(context.weights.get(self.name, 0.0))
        return ToolResult(
            name=self.name,
            weight=weight,
            summary=summary,
            payload=payload,
        )


AskNewsImpactTool.__TOOL_META__ = __TOOL_META__
