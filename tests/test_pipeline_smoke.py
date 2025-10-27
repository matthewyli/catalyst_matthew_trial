from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalyst_matthew_trial.pipeline.keyword_config import ToolKeywordConfig
from catalyst_matthew_trial.pipeline.keyword_detector import KeywordDetector
from catalyst_matthew_trial.pipeline.strategy_pipeline import StrategyPipeline
from catalyst_matthew_trial.pipeline.tool_registry import ToolRegistry
from catalyst_matthew_trial.pipeline.usage_tracker import UsageTracker
from catalyst_matthew_trial.tools.base import BaseTool, ToolContext, ToolResult


class StubTool(BaseTool):
    def __init__(self, name: str, phases: list[str], score: float = 0.5, outputs: list[str] | None = None) -> None:
        self.name = name
        self.description = f"stub tool {name}"
        self.keywords = frozenset({name})
        self._phases = phases
        self._score = score
        self.__TOOL_META__ = {
            "name": name,
            "module": "tests.test_pipeline_smoke",
            "object": self.__class__.__name__,
            "phases": phases,
            "outputs": outputs or [],
        }
        super().__init__()

    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        runtime = params.get("runtime", {}) if isinstance(params, dict) else {}
        phase = runtime.get("current_phase")
        payload = {
            "phase": phase,
            "tool": self.name,
            "score": self._score,
        }
        if options.get("strict_io"):
            payload["value"] = self._score
        return ToolResult(name=self.name, weight=self._score, summary=f"{self.name} ran", payload=payload)


def build_stub_pipeline(tmp_path: Path) -> StrategyPipeline:
    registry = ToolRegistry()
    tools = [
        StubTool("stub_data", ["data_gather"], score=0.6),
        StubTool("stub_feature_a", ["feature_engineering"], score=0.7),
        StubTool("stub_feature_b", ["feature_engineering"], score=0.3),
        StubTool("stub_signal", ["signal_generation"], score=0.5),
        StubTool("stub_risk", ["risk_sizing"], score=0.4),
        StubTool("stub_execution", ["execution"], score=0.2),
    ]
    registry.bulk_register(tools)

    tool_keywords = {
        "stub_data": ToolKeywordConfig(terms=["data"], patterns=[]),
        "stub_feature_a": ToolKeywordConfig(terms=["feature"], patterns=[]),
        "stub_feature_b": ToolKeywordConfig(terms=["feature"], patterns=[]),
        "stub_signal": ToolKeywordConfig(terms=["signal"], patterns=[]),
        "stub_risk": ToolKeywordConfig(terms=["risk"], patterns=[]),
        "stub_execution": ToolKeywordConfig(terms=["execute"], patterns=[]),
    }
    detector = KeywordDetector(
        tool_keywords=tool_keywords,
        asset_aliases={"sol": "SOL"},
    )

    usage_tracker = UsageTracker(tmp_path / "usage.json", decay_half_life_hours=1.0)

    pipeline = StrategyPipeline(
        registry=registry,
        detector=detector,
        tracker=usage_tracker,
    )
    return pipeline


def test_usage_tracker_decay(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "usage.json", decay_half_life_hours=1.0)
    tracker.ensure(["a", "b"])
    tracker.increment(["a"])
    weights = tracker.weights(["a", "b"])
    assert weights["a"] > weights["b"]


def test_pipeline_run_persists_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    pipeline = build_stub_pipeline(tmp_path)
    output = pipeline.run("Run feature signal execute for SOL", asset="SOL")

    assert output.primary_asset == "SOL"
    assert not output.phase_alerts

    run_dirs = list((tmp_path / "runs").glob("*"))
    assert run_dirs, "Expected a persisted run directory"
    summary_path = run_dirs[0] / "summary.json"
    runs_json = json.loads(summary_path.read_text())
    assert runs_json["prompt"].startswith("Run feature")
    assert "phase_combinations" in runs_json["metadata"]["router"]
    assert any(run["name"] == "feature_engineering_combined" for run in output.tool_runs)


def test_strict_mode_outputs_numeric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_STRICT_IO", "true")
    pipeline = build_stub_pipeline(tmp_path)
    output = pipeline.run("Run feature signal execute for SOL", asset="SOL")

    numeric_runs = [
        run
        for run in output.tool_runs
        if isinstance(run.get("payload"), dict) and isinstance(run["payload"].get("value"), (int, float))
    ]
    assert numeric_runs, "Expected at least one strict value in tool payloads"

    monkeypatch.delenv("PIPELINE_STRICT_IO", raising=False)
