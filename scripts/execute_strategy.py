#!/usr/bin/env python
"""
Convert TextQL primer output into an executable strategy loop.

Usage:
  python scripts/execute_strategy.py [--run-dir runs/<timestamp>] [--interval 900] [--iterations 1]

The script finds the most recent pipeline run (or the one you specify),
extracts the `textql_primer` payload, writes a structured `strategy_spec.json`,
and then iterates through the described pipeline layout, logging the suggested
API calls and validation logic. The actual data fetching/execution is left as
stubs so you can plug in real connectors or order routing when ready.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def find_latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_tool_runs(run_dir: Path) -> List[Mapping[str, Any]]:
    tool_path = run_dir / "tool_runs.json"
    if not tool_path.exists():
        raise FileNotFoundError(f"Missing tool_runs.json in {run_dir}")
    return json.loads(tool_path.read_text())


def parse_textql_answer(primer_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = primer_payload.get("raw_textql_response") or {}
    if not isinstance(raw, Mapping):
        return {}
    answer = raw.get("answer")
    if not answer or not isinstance(answer, str):
        return {}
    text = answer.replace("```json", "").replace("```", "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def build_strategy_spec(run_dir: Path) -> Mapping[str, Any]:
    tool_runs = load_tool_runs(run_dir)
    primer = next((entry for entry in tool_runs if entry.get("name") == "textql_primer"), None)
    if not primer:
        raise ValueError(f"No textql_primer entry found in {run_dir}")

    payload = primer.get("payload", {})
    parsed_answer = parse_textql_answer(payload)

    refined = parsed_answer.get("prompt_refinement") or payload.get("prompt_refinement") or {}
    hypotheses = parsed_answer.get("search_hypotheses") or payload.get("search_hypotheses") or []
    layout = parsed_answer.get("pipeline_layout") or payload.get("pipeline_layout") or []
    dataset_requests = parsed_answer.get("dataset_requests") or payload.get("dataset_requests") or {}

    spec = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_dir": str(run_dir),
        "refined_prompt": refined.get("prompt_text"),
        "prompt_background": refined.get("background"),
        "monitoring_notes": {
            "current_state": refined.get("current_state"),
            "methodology": refined.get("methodology"),
            "data_sources": refined.get("data_sources"),
        },
        "pipeline_layout": layout,
        "hypotheses": hypotheses,
        "dataset_requests": dataset_requests,
        "validation_steps": parsed_answer.get("validation_steps") or payload.get("validation_steps"),
    }

    spec_path = run_dir / "strategy_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def _build_phase_map(hypotheses: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    phase_map: Dict[str, List[Mapping[str, Any]]] = {}
    for hyp in hypotheses:
        phase = "data_gather"
        hooks = hyp.get("onchain_hooks") or []
        if hooks:
            if any("history" in str(h).lower() for h in hooks):
                phase = "feature_engineering"
            if any("funding" in str(h).lower() for h in hooks):
                phase = "signal_generation"
        phase_map.setdefault(phase, []).append(hyp)
    return phase_map


def execute_strategy(spec: Mapping[str, Any], iterations: int, interval: float) -> None:
    run_dir = Path(spec.get("run_dir", "."))
    log_path = run_dir / "strategy_execution.log"
    phase_hypotheses = _build_phase_map(spec.get("hypotheses") or [])

    for iteration in range(1, iterations + 1):
        lines: List[str] = []
        header = f"[strategy-loop] iteration={iteration} timestamp={datetime.utcnow().isoformat()}Z"
        print(header)
        lines.append(header)

        for phase in spec.get("pipeline_layout") or []:
            phase_name = phase.get("phase", "unknown")
            objective = phase.get("objective") or ""
            inputs = ", ".join(phase.get("inputs") or [])
            logic = "; ".join(phase.get("logic") or [])
            outputs = ", ".join(phase.get("outputs") or [])

            phase_header = f"  Phase: {phase_name}"
            print(phase_header)
            lines.append(phase_header)
            details = [
                f"    Objective: {objective}",
                f"    Inputs: {inputs or 'n/a'}",
                f"    Logic: {logic or 'n/a'}",
                f"    Outputs: {outputs or 'n/a'}",
            ]
            for d in details:
                print(d)
                lines.append(d)

            for hyp in phase_hypotheses.get(phase_name, []):
                hyp_line = f"    Hypothesis: {hyp.get('name')} -> {hyp.get('description')}"
                print(hyp_line)
                lines.append(hyp_line)
                for fact in hyp.get("facts") or []:
                    fact_line = f"      Fact: {fact.get('label')} | Source: {fact.get('source')} | Guidance: {fact.get('value')}"
                    print(fact_line)
                    lines.append(fact_line)
                for action in hyp.get("actions") or []:
                    action_line = f"      Action: {action}"
                    print(action_line)
                    lines.append(action_line)
                for validation in hyp.get("validation") or []:
                    validation_line = f"      Validation: {validation}"
                    print(validation_line)
                    lines.append(validation_line)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        if iteration < iterations:
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute TextQL strategy spec.")
    parser.add_argument("--run-dir", type=Path, help="Specific run directory (defaults to latest).")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="Root directory containing runs.")
    parser.add_argument("--interval", type=float, default=900.0, help="Seconds between iterations (default 15m).")
    parser.add_argument("--iterations", type=int, default=1, help="How many iterations to run.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(args.runs_root)
    print(f"[strategy] Using run directory: {run_dir}")
    spec = build_strategy_spec(run_dir)
    print(f"[strategy] Wrote spec to {run_dir / 'strategy_spec.json'}")
    execute_strategy(spec, iterations=args.iterations, interval=args.interval)


if __name__ == "__main__":
    main()
