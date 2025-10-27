from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import PipelineExecutionError
from .router import KeywordRouter, RouteDecision
from .weights import WeightManager

try:  # pragma: no cover - optional dependency on BaseTool context
    from catalyst_matthew_trial.tools.base import ToolContext as BaseToolContext
    from catalyst_matthew_trial.tools.base import ToolExecutionError as BaseToolExecutionError
except ImportError:  # pragma: no cover
    BaseToolContext = None  # type: ignore[assignment]

    class BaseToolExecutionError(Exception):  # type: ignore[no-redef]
        ...


@dataclass
class PipelineResult:
    prompt: str
    tools_invoked: List[Dict[str, Any]]
    selected_tools: List[str]
    keywords: List[str]
    assets: List[str]
    weights: Dict[str, float]
    weights_updated: Dict[str, float]
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Pipeline:
    """Simple orchestrator pipeline that routes prompts and executes tools."""

    def __init__(self, registry: Dict[str, object], weight_manager: WeightManager) -> None:
        self.registry = registry
        self.weights = weight_manager
        self.router = KeywordRouter(registry)
        self.weights.ensure(self.registry.keys())

    def run(self, prompt: str, *, asset: Optional[str] = None) -> PipelineResult:
        decision = self.router.route(prompt)
        selected = decision.tools or list(self.registry.keys())
        if asset and asset.upper() not in decision.assets:
            decision.assets.insert(0, asset.upper())

        weights_before = self.weights.weights(self.registry.keys())
        usage_counts = self.weights.counts
        tool_runs: List[Dict[str, Any]] = []
        executed_names: List[str] = []
        context_obj = self._build_context(decision, asset, weights_before, usage_counts)

        for name in selected:
            tool = self.registry.get(name)
            if tool is None:
                tool_runs.append(
                    {
                        "name": name,
                        "summary": f"ALERT: need {name} tool",
                        "payload": None,
                    }
                )
                continue
            executor = getattr(tool, "execute", None)
            if not callable(executor):
                tool_runs.append(
                    {
                        "name": name,
                        "summary": f"Tool '{name}' missing execute() method.",
                        "payload": None,
                    }
                )
                continue
            try:
                result = executor(prompt=prompt, context=context_obj)
            except TypeError:
                # Support positional signature (prompt, context)
                try:
                    result = executor(prompt, context_obj)
                except BaseToolExecutionError as exc:
                    result = {
                        "summary": f"{name} failed: {exc}",
                        "payload": {"error": str(exc)},
                    }
                except Exception as exc:  # pragma: no cover - runtime path
                    result = {
                        "summary": f"{name} failed: {exc}",
                        "payload": {"error": str(exc)},
                    }
            except BaseToolExecutionError as exc:
                result = {
                    "summary": f"{name} failed: {exc}",
                    "payload": {"error": str(exc)},
                }

            if isinstance(result, dict):
                summary = result.get("summary") or result.get("message") or ""
                payload = result.get("payload")
            elif hasattr(result, "summary") and hasattr(result, "payload"):
                summary = result.summary
                payload = result.payload
            else:
                summary = str(result)
                payload = None

            tool_runs.append(
                {
                    "name": name,
                    "summary": summary,
                    "payload": payload,
                }
            )
            executed_names.append(name)

        if executed_names:
            self.weights.increment(executed_names)
        weights_after = self.weights.weights(self.registry.keys())

        return PipelineResult(
            prompt=prompt,
            tools_invoked=tool_runs,
            selected_tools=selected,
            keywords=sorted(decision.keywords),
            assets=decision.assets,
            weights=weights_before,
            weights_updated=weights_after,
        )

    def _build_context(
        self,
        decision: RouteDecision,
        asset: Optional[str],
        weights_before: Dict[str, float],
        usage_counts: Dict[str, int],
    ) -> object:
        primary_asset = decision.assets[0] if decision.assets else (asset.upper() if asset else None)
        metadata = {
            "supplied_asset": asset.upper() if asset else None,
            "router_keywords": sorted(decision.keywords),
        }
        if BaseToolContext:
            return BaseToolContext(
                asset=primary_asset,
                assets=tuple(decision.assets),
                detected_keywords=tuple(sorted(decision.keywords)),
                metadata=metadata,
                usage_counts=usage_counts,
                weights=weights_before,
            )
        return {
            "asset": primary_asset,
            "assets": decision.assets,
            "detected_keywords": sorted(decision.keywords),
            "metadata": metadata,
            "usage_counts": usage_counts,
            "weights": weights_before,
        }
