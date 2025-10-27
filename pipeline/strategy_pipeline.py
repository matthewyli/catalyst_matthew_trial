from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tools.backtest_utils import normalize_backtest_output
from tools.base import BaseTool, ToolContext, ToolExecutionError, ToolResult
from .execution_context import ExecutionContext
from .keyword_detector import KeywordDetector
from .prompt_parser import parse_prompt
from cache import force_refresh_from_env
from .data_contracts import validate_phase_output
from .tool_registry import ToolRegistry
from .usage_tracker import UsageTracker


def _strict_io_enabled() -> bool:
    raw = os.getenv("PIPELINE_STRICT_IO", "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PipelineOutput:
    prompt: str
    primary_asset: Optional[str]
    assets: List[str]
    detected_keywords: List[str]
    metadata: Dict[str, Any]
    missing_tools: List[str]
    tool_runs: List[Dict[str, Any]]
    weights_before: Dict[str, float]
    weights_after: Dict[str, float]
    selection_weights: Dict[str, float]
    strategy_notes: List[str] = field(default_factory=list)
    recommendation: str = "neutral"
    score: float = 0.0
    phase_alerts: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["timestamp"] = self.timestamp
        return out


class StrategyPipeline:
    """Orchestrates tool execution across phased adapters."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        detector: KeywordDetector,
        tracker: UsageTracker,
        default_tools_on_empty: bool = True,
        keyword_weight_alpha: float = 0.7,
        blend_mode: str = "linear",
        phases: Optional[Sequence[str]] = None,
    ) -> None:
        self.registry = registry
        self.detector = detector
        self.tracker = tracker
        self.default_tools_on_empty = default_tools_on_empty
        self.keyword_weight_alpha = min(max(keyword_weight_alpha, 0.0), 1.0)
        self.blend_mode = blend_mode.lower()
        if self.blend_mode not in {"linear", "sqrt", "softmax"}:
            self.blend_mode = "linear"
        self.debug = os.getenv("PIPELINE_DEBUG", "true").lower() != "false"
        self.phase_order: List[str] = list(
            phases
            or [
                "data_gather",
                "feature_engineering",
                "signal_generation",
                "risk_sizing",
                "execution",
            ]
        )
        self.tracker.ensure(self.registry.names())

    # ------------------------------------------------------------------ main entry
    def run(self, prompt: str, *, asset: Optional[str] = None) -> PipelineOutput:
        parsed = parse_prompt(prompt, default_asset=asset.upper() if asset else "BTC")
        detection = self.detector.detect(prompt, explicit_asset=parsed.primary_asset)
        assets = parsed.assets or list(detection.assets) or ([asset.upper()] if asset else [])
        primary_asset = parsed.primary_asset or (assets[0] if assets else None)
        strict_io = _strict_io_enabled()

        if self.debug:
            print("[pipeline] --------------------------------------------------")
            print(f"[pipeline] Prompt: {prompt}")
            print(f"[pipeline] Parsed assets={assets}, timeframe={parsed.timeframe_minutes}, "
                  f"indicators={parsed.indicators}, goals={parsed.goals}, force_refresh={parsed.force_refresh}")
            print(f"[pipeline] Detected keywords={sorted(detection.keywords)}")

        router_info: Dict[str, Any] = {
            "matched_terms": detection.matched_terms,
            "matched_patterns": detection.matched_patterns,
            "keyword_scores": detection.scores,
            "keyword_weight_alpha": self.keyword_weight_alpha,
            "metadata": dict(detection.metadata),
            "phase_order": list(self.phase_order),
            "llm_recommendations": detection.llm_recommendations,
            "strict_io": strict_io,
        }

        context_data = ExecutionContext(
            symbols=assets,
            primary_symbol=primary_asset,
            timeframe_minutes=parsed.timeframe_minutes
            or detection.metadata.get("window_min")
            if detection.metadata
            else None,
            params={
                "prompt": prompt,
                "detected_keywords": sorted(detection.keywords),
                "router": router_info,
                "parse": {
                    "assets": parsed.assets,
                    "timeframe_minutes": parsed.timeframe_minutes,
                    "indicators": parsed.indicators,
                    "goals": parsed.goals,
                },
            },
        )
        context_data.update_risk_constraints(self._load_risk_constraints())
        options = context_data.params_section("options")
        options["force_refresh"] = parsed.force_refresh or force_refresh_from_env()
        options["strict_io"] = strict_io

        self.tracker.ensure(self.registry.names())
        weights_before = self.tracker.weights(self.registry.names())
        usage_counts = self.tracker.counts

        combined_scores = self._combine_scores(detection.scores, weights_before)
        router_info["combined_weights"] = combined_scores

        requested_tool_names = [
            name
            for name, score in sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
            if score > 0 or detection.scores.get(name, 0.0) > 0
        ]
        if not requested_tool_names:
            requested_tool_names = [
                name
                for name, score in sorted(detection.scores.items(), key=lambda kv: kv[1], reverse=True)
                if score > 0
            ]
        router_info["tool_ranking"] = requested_tool_names

        tool_runs: List[Dict[str, Any]] = []
        gather_results: List[ToolResult] = []
        phase_outputs = context_data.phase_outputs
        phase_alerts: List[str] = []
        missing_tools: List[str] = []
        executed_records: List[Dict[str, str]] = []
        executed_names: List[str] = []
        phase_combinations: Dict[str, Any] = {}

        telemetry: Dict[str, Any] = {
            "start_time": dt.datetime.utcnow().isoformat(),
            "phases": [],
            "alerts": [],
        }

        runtime_state = context_data.params_section("runtime")

        context = ToolContext(
            asset=primary_asset,
            assets=tuple(assets),
            detected_keywords=tuple(sorted(detection.keywords)),
            metadata=context_data.state,
            usage_counts=usage_counts,
            weights=weights_before,
        )

        executed_set: Set[str] = set()
        pipeline_start = time.perf_counter()

        for phase in self.phase_order:
            phase_start = time.perf_counter()
            raw_candidates = self._tools_for_phase(phase)
            candidate_names = [name for name, _ in raw_candidates]
            selected_names: List[str] = []
            phase_note: Optional[str] = None

            if self.debug:
                print(f"[pipeline] Phase {phase} start candidates={candidate_names}")

            if not raw_candidates:
                if self.default_tools_on_empty:
                    alert = f"ALERT: need {phase} tool"
                    phase_alerts.append(alert)
                    telemetry["alerts"].append(alert)
                    missing_tools.append(f"{phase}:<missing>")
                    context_data.add_phase_note(phase, alert)
                    tool_runs.append(
                        {
                            "phase": phase,
                            "name": f"ALERT:{phase}",
                            "weight": 0.0,
                            "summary": alert,
                            "payload": None,
                        }
                    )
                    phase_note = alert
                telemetry["phases"].append(
                    {
                        "phase": phase,
                        "duration_sec": time.perf_counter() - phase_start,
                        "candidates": candidate_names,
                        "selected": [],
                        "note": phase_note,
                    }
                )
                if self.debug:
                    print(f"[pipeline] Phase {phase} skipped: {phase_note or 'no tools'}")
                continue

            candidates = [(name, tool) for name, tool in raw_candidates if name not in executed_set]
            if not candidates:
                note = "reused prior tool outputs"
                context_data.add_phase_note(phase, note)
                telemetry["phases"].append(
                    {
                        "phase": phase,
                        "duration_sec": time.perf_counter() - phase_start,
                        "candidates": candidate_names,
                        "selected": [],
                        "note": note,
                    }
                )
                if self.debug:
                    print(f"[pipeline] Phase {phase} re-used outputs")
                continue

            selected = self._select_tools_for_phase(candidates, combined_scores)
            if not selected:
                alert = f"ALERT: need {phase} tool"
                phase_alerts.append(alert)
                telemetry["alerts"].append(alert)
                missing_tools.append(f"{phase}:<unranked>")
                context_data.add_phase_note(phase, alert)
                tool_runs.append(
                    {
                        "phase": phase,
                        "name": f"ALERT:{phase}",
                        "weight": 0.0,
                        "summary": alert,
                        "payload": None,
                    }
                )
                telemetry["phases"].append(
                    {
                        "phase": phase,
                        "duration_sec": time.perf_counter() - phase_start,
                        "candidates": candidate_names,
                        "selected": [],
                        "note": alert,
                    }
                )
                if self.debug:
                    print(f"[pipeline] Phase {phase} alert: {alert}")
                continue

            phase_results: List[ToolResult] = []
            for name, tool in selected:
                selected_names.append(name)
                runtime_state["current_phase"] = phase
                runtime_state["active_tool"] = name
                runtime_state["tool_score"] = combined_scores.get(name, 0.0)

                try:
                    result = tool.execute(prompt, context)
                    meta = getattr(tool, "__TOOL_META__", {}) or {}
                    outputs = {str(o).lower() for o in (meta.get("outputs") or [])}
                    if "backtest" in outputs:
                        result.payload = normalize_backtest_output(result.payload)
                    ok, validation_msg = validate_phase_output(phase, result.payload)
                    if not ok:
                        raise ToolExecutionError(f"{name} produced invalid payload: {validation_msg}")
                    gather_results.append(result)
                    executed_names.append(result.name)
                    executed_records.append({"phase": phase, "tool": result.name})
                    context_data.update_phase_output(phase, result.name, result.payload)
                    phase_results.append(result)
                    tool_runs.append(
                        {
                            "phase": phase,
                            "name": result.name,
                            "weight": result.weight,
                            "summary": result.summary,
                            "payload": result.payload,
                        }
                    )
                except ToolExecutionError as exc:
                    tool_runs.append(
                        {
                            "phase": phase,
                            "name": name,
                            "weight": weights_before.get(name, 0.0),
                            "summary": f"{name} failed: {exc}",
                            "payload": {"error": str(exc)},
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    tool_runs.append(
                        {
                            "phase": phase,
                            "name": name,
                            "weight": weights_before.get(name, 0.0),
                            "summary": f"{name} crashed: {exc}",
                            "payload": {"error": str(exc)},
                        }
                    )
                executed_set.add(name)

            if len(phase_results) > 1:
                combined = self._combine_phase_results(phase, phase_results, combined_scores)
                if combined:
                    ok, validation_msg = validate_phase_output(phase, combined["payload"])
                    if not ok:
                        raise ToolExecutionError(f"combined output invalid: {validation_msg}")
                    phase_combinations[phase] = combined["meta"]
                    context_data.update_phase_output(phase, "_combined", combined["payload"])
                    tool_runs.append(combined["log_entry"])
                    phase_note = f"combined {len(phase_results)} tools"

            telemetry["phases"].append(
                {
                    "phase": phase,
                    "duration_sec": time.perf_counter() - phase_start,
                    "candidates": candidate_names,
                    "selected": selected_names,
                    "note": phase_note,
                }
            )
            if self.debug:
                print(
                    f"[pipeline] Phase {phase}: selected={selected_names or '-'} "
                    f"candidates={candidate_names} note={phase_note}"
                )

        router_info["executed_tools"] = executed_records
        router_info["phase_combinations"] = phase_combinations
        telemetry["total_duration_sec"] = time.perf_counter() - pipeline_start

        if executed_names:
            unique_executed = list(dict.fromkeys(executed_names))
            self.tracker.increment(unique_executed)
        else:
            unique_executed = []
        weights_after = self.tracker.weights(self.registry.names())
        selection_weights = self.tracker.weights(unique_executed) if unique_executed else {}

        notes, score = self._compose_strategy_notes(primary_asset, gather_results)
        recommendation = self._decide_recommendation(score)

        output_metadata = {
            "router": router_info,
            "context": context_data.state,
            "telemetry": telemetry,
        }

        output = PipelineOutput(
            prompt=prompt,
            primary_asset=primary_asset,
            assets=list(assets),
            detected_keywords=sorted(detection.keywords),
            metadata=output_metadata,
            missing_tools=missing_tools,
            tool_runs=tool_runs,
            weights_before=weights_before,
            weights_after=weights_after,
            selection_weights=selection_weights,
            strategy_notes=notes,
            recommendation=recommendation,
            score=score,
            phase_alerts=phase_alerts,
        )

        self._persist_run(output)
        if self.debug:
            print(
                f"[pipeline] Completed run: recommendation={output.recommendation} "
                f"score={output.score:.3f} alerts={output.phase_alerts}"
            )
        return output

    # ------------------------------------------------------------------ helpers
    def _tools_for_phase(self, phase: str) -> List[Tuple[str, BaseTool]]:
        phase_lower = phase.lower()
        matched: List[Tuple[str, BaseTool]] = []
        for name in self.registry.names():
            tool = self.registry.get(name)
            if tool is None:
                continue
            meta = getattr(tool, "__TOOL_META__", {}) or {}
            phases = meta.get("phases") or []
            if any(str(p).lower() == phase_lower for p in phases):
                matched.append((name, tool))
        return matched

    def _select_tools_for_phase(
        self,
        candidates: List[Tuple[str, BaseTool]],
        combined_scores: Mapping[str, float],
    ) -> List[Tuple[str, BaseTool]]:
        if not candidates:
            return []
        scored = sorted(
            [(name, tool, combined_scores.get(name, 0.0)) for name, tool in candidates],
            key=lambda item: item[2],
            reverse=True,
        )
        positives = [item for item in scored if item[2] > 0]
        chosen = positives if positives else scored[:1]
        return [(name, tool) for name, tool, _ in chosen]

    def _compose_strategy_notes(self, asset: Optional[str], results: Sequence[ToolResult]) -> tuple[List[str], float]:
        notes: List[str] = []
        score = 0.0
        strict_pairs: List[Tuple[str, float]] = []

        for result in results:
            payload = result.payload or {}
            name = result.name

            if isinstance(payload, dict) and "value" in payload and isinstance(payload["value"], (int, float)):
                strict_pairs.append((name, float(payload["value"])))
                continue

            if name == "asknews_impact" and payload:
                impact = payload.get("impact_score")
                direction = payload.get("direction")
                confidence = payload.get("confidence")
                if impact is not None and direction:
                    direction_sign = 1 if str(direction).lower().startswith("pos") else -1 if str(direction).lower().startswith("neg") else 0
                    score += float(impact) * direction_sign * (confidence or 1.0)
                    conf_text = f"{float(confidence):.0%}" if isinstance(confidence, (int, float)) else "n/a"
                    notes.append(f"News: impact {impact:.2f} {direction} (confidence {conf_text}).")
            elif name == "tvl_growth" and payload:
                changes = payload.get("pct_changes") or {}
                change_7d = changes.get("change_7d")
                if change_7d is not None:
                    score += float(change_7d)
                    notes.append(f"TVL: 7d change {change_7d:.2%}.")
            elif name == "volatility_percentile" and payload:
                percentile = payload.get("vol_percentile")
                if percentile is not None:
                    normalized = (50.0 - float(percentile)) / 100.0
                    score += normalized * 0.5
                    notes.append(f"Volatility: percentile {percentile:.1f} (lower => quieter).")

        if strict_pairs:
            strict_values = [value for _, value in strict_pairs]
            for tool_name, value in strict_pairs:
                notes.append(f"{tool_name}: strict value {value:.4f}")
            score = sum(strict_values) / len(strict_values) if strict_values else 0.0
            return notes, score

        if not notes and asset:
            notes.append(f"No actionable tool output for {asset}; maintain neutral stance.")
        return notes, score

    def _decide_recommendation(self, score: float) -> str:
        if score > 0.75:
            return "strong_long"
        if score > 0.25:
            return "lean_long"
        if score < -0.75:
            return "strong_short"
        if score < -0.25:
            return "lean_short"
        return "neutral"

    def _combine_scores(
        self,
        keyword_scores: Mapping[str, float],
        usage_weights: Mapping[str, float],
    ) -> Dict[str, float]:
        names = set(self.registry.names()) | set(keyword_scores.keys())

        keyword_raw = {name: max(keyword_scores.get(name, 0.0), 0.0) for name in names}
        usage_raw = {name: max(usage_weights.get(name, 0.0), 0.0) for name in names}

        keyword_total = sum(keyword_raw.values())
        if keyword_total > 0:
            keyword_norm = {name: value / keyword_total for name, value in keyword_raw.items()}
        else:
            keyword_norm = {name: 0.0 for name in names}

        usage_total = sum(usage_raw.values())
        if usage_total > 0:
            usage_norm = {name: value / usage_total for name, value in usage_raw.items()}
        else:
            uniform = 1.0 / len(names) if names else 0.0
            usage_norm = {name: uniform for name in names}

        keyword_adj: Dict[str, float] = {}
        usage_adj: Dict[str, float] = {}

        for name in names:
            if self.blend_mode == "sqrt":
                keyword_adj[name] = math.sqrt(keyword_norm.get(name, 0.0))
                usage_adj[name] = math.sqrt(usage_norm.get(name, 0.0))
            elif self.blend_mode == "softmax":
                keyword_adj[name] = math.exp(keyword_norm.get(name, 0.0))
                usage_adj[name] = math.exp(usage_norm.get(name, 0.0))
            else:
                keyword_adj[name] = keyword_norm.get(name, 0.0)
                usage_adj[name] = usage_norm.get(name, 0.0)

        kw_total = sum(keyword_adj.values())
        if kw_total > 0:
            keyword_adj = {name: value / kw_total for name, value in keyword_adj.items()}
        else:
            keyword_adj = {name: 0.0 for name in names}

        usage_total_adj = sum(usage_adj.values())
        if usage_total_adj > 0:
            usage_adj = {name: value / usage_total_adj for name, value in usage_adj.items()}
        else:
            usage_adj = {name: 0.0 for name in names}

        combined = {
            name: self.keyword_weight_alpha * keyword_adj.get(name, 0.0)
            + (1.0 - self.keyword_weight_alpha) * usage_adj.get(name, 0.0)
            for name in names
        }

        total = sum(combined.values())
        if total > 0:
            combined = {name: value / total for name, value in combined.items()}

        return combined

    def _combine_phase_results(
        self,
        phase: str,
        results: Sequence[ToolResult],
        combined_scores: Mapping[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Combine multiple tool results for a phase using weighted averages."""

        tool_weights: Dict[str, float] = {}
        score_sum = sum(max(combined_scores.get(res.name, 0.0), 0.0) for res in results)
        if score_sum > 0:
            for res in results:
                tool_weights[res.name] = max(combined_scores.get(res.name, 0.0), 0.0) / score_sum
        else:
            equal = 1.0 / len(results)
            for res in results:
                tool_weights[res.name] = equal

        weighted_metrics: Dict[str, float] = {}
        source_payloads: Dict[str, Any] = {}
        for res in results:
            weight = tool_weights.get(res.name, 0.0)
            payload = res.payload
            source_payloads[res.name] = payload
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        weighted_metrics[key] = weighted_metrics.get(key, 0.0) + weight * float(value)

        payload = {
            "method": "weighted_mean",
            "source_tools": [
                {"name": res.name, "weight": tool_weights.get(res.name, 0.0)}
                for res in results
            ],
            "weighted_metrics": weighted_metrics,
            "source_payloads": source_payloads,
        }

        weights_desc = ", ".join(
            f"{res.name}({tool_weights.get(res.name, 0.0):.2f})" for res in results
        )

        log_entry = {
            "phase": phase,
            "name": f"{phase}_combined",
            "weight": 0.0,
            "summary": (
                f"Combined {len(results)} tools for phase '{phase}' "
                f"using weighted_mean: {weights_desc}"
            ),
            "payload": payload,
        }

        meta = {
            "method": "weighted_mean",
            "tools": payload["source_tools"],
        }

        return {
            "payload": payload,
            "log_entry": log_entry,
            "meta": meta,
        }

    def _load_risk_constraints(self) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}

        def _float_env(name: str) -> Optional[float]:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        kill = os.getenv("RISK_KILL_SWITCH")
        if kill is not None:
            constraints["kill_switch"] = kill.lower() == "true"

        for env_name, key in (
            ("RISK_MAX_POSITION", "max_position"),
            ("RISK_MAX_LEVERAGE", "max_leverage"),
            ("RISK_MAX_DRAWDOWN", "max_drawdown"),
            ("RISK_CURRENT_DRAWDOWN", "current_drawdown"),
            ("RISK_PORTFOLIO_EQUITY", "portfolio_equity"),
            ("RISK_OPEN_NOTIONAL", "open_notional"),
            ("RISK_DEFAULT_PRICE", "default_price"),
        ):
            value = _float_env(env_name)
            if value is not None:
                constraints[key] = value

        for env_name, key in (("RISK_POSITION_LIMITS", "position_limits"), ("RISK_OPEN_POSITIONS", "open_positions")):
            raw = os.getenv(env_name)
            if not raw:
                continue
            try:
                data = json.loads(raw)
                mapping = {str(sym).upper(): float(val) for sym, val in data.items()}
                constraints[key] = mapping
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return constraints

    def _persist_run(self, output: PipelineOutput) -> None:
        root = Path("runs")
        root.mkdir(parents=True, exist_ok=True)
        ts = output.timestamp.replace(":", "-")
        run_dir = root / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "prompt": output.prompt,
            "primary_asset": output.primary_asset,
            "assets": output.assets,
            "detected_keywords": output.detected_keywords,
            "metadata": output.metadata,
            "missing_tools": output.missing_tools,
            "weights_before": output.weights_before,
            "weights_after": output.weights_after,
            "selection_weights": output.selection_weights,
            "strategy_notes": output.strategy_notes,
            "recommendation": output.recommendation,
            "score": output.score,
            "phase_alerts": output.phase_alerts,
            "timestamp": output.timestamp,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "tool_runs.json").write_text(json.dumps(output.tool_runs, indent=2), encoding="utf-8")

        telemetry = output.metadata.get("telemetry", {})
        router = output.metadata.get("router", {})
        report_lines = [
            f"Prompt: {output.prompt}",
            f"Primary asset: {output.primary_asset}",
            f"Assets: {', '.join(output.assets)}",
            f"Recommendation: {output.recommendation} (score={output.score:.3f})",
            "",
        ]
        if telemetry:
            total = telemetry.get("total_duration_sec")
            if isinstance(total, (int, float)):
                report_lines.append(f"Total runtime: {total:.3f}s")
            report_lines.append("Phase timings:")
            for phase_entry in telemetry.get("phases", []):
                phase = phase_entry.get("phase")
                duration = phase_entry.get("duration_sec", 0.0)
                selected = ", ".join(phase_entry.get("selected", [])) or "-"
                candidates = ", ".join(phase_entry.get("candidates", [])) or "-"
                note = phase_entry.get("note")
                line = f"  - {phase}: {duration:.3f}s | selected: {selected} | candidates: {candidates}"
                if note:
                    line += f" | note: {note}"
                report_lines.append(line)
        if router:
            report_lines.append("Combined weights:")
            weights = router.get("combined_weights", {})
            for name, value in weights.items():
                report_lines.append(f"  - {name}: {value:.3f}")
        if output.phase_alerts:
            report_lines.append("Alerts:")
            for alert in output.phase_alerts:
                report_lines.append(f"  - {alert}")
        (run_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")









