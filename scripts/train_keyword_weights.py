from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pipeline.keyword_config import (
    ASSET_ALIASES,
    BLOCKCHAIN_ALIASES,
    TIMEFRAME_KEYWORDS,
    TOOL_KEYWORDS,
)
from pipeline.keyword_detector import KeywordDetector


@dataclass
class ToolStats:
    pos_literals: float = 0.0
    pos_patterns: float = 0.0
    pos_scores: float = 0.0
    pos_count: int = 0
    neg_literals: float = 0.0
    neg_patterns: float = 0.0
    neg_scores: float = 0.0
    neg_count: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "pos_literals": self.pos_literals,
            "pos_patterns": self.pos_patterns,
            "pos_scores": self.pos_scores,
            "pos_count": self.pos_count,
            "neg_literals": self.neg_literals,
            "neg_patterns": self.neg_patterns,
            "neg_scores": self.neg_scores,
            "neg_count": self.neg_count,
        }


def load_dataset(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows = payload.get("examples") or payload.get("data") or []
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("Training dataset must be a list or JSONL file.")
    normalized: List[Dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        targets: Sequence[str]
        if "tools" in row and isinstance(row["tools"], (list, tuple)):
            targets = [str(name) for name in row["tools"]]
        elif "tool" in row:
            targets = [str(row["tool"])]
        else:
            continue
        normalized.append({"prompt": prompt.strip(), "tools": list(dict.fromkeys(targets))})
    if not normalized:
        raise ValueError("Training dataset did not yield any usable rows.")
    return normalized


def compute_adjusted_weights(
    stats: Dict[str, ToolStats],
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, Dict[str, float]]:
    eps = 1e-6
    learned: Dict[str, Dict[str, float]] = {}
    for tool, cfg in TOOL_KEYWORDS.items():
        tool_stats = stats.get(tool)
        if tool_stats is None or tool_stats.pos_count == 0:
            continue
        pos_literal_mean = tool_stats.pos_literals / max(tool_stats.pos_count, 1)
        neg_literal_mean = tool_stats.neg_literals / max(tool_stats.neg_count, 1)
        literal_ratio = (pos_literal_mean + eps) / (neg_literal_mean + eps)
        literal_weight = cfg.literal_weight * (literal_ratio ** alpha)

        pos_pattern_mean = tool_stats.pos_patterns / max(tool_stats.pos_count, 1)
        neg_pattern_mean = tool_stats.neg_patterns / max(tool_stats.neg_count, 1)
        pattern_ratio = (pos_pattern_mean + eps) / (neg_pattern_mean + eps)
        pattern_weight = cfg.pattern_weight * (pattern_ratio ** beta)

        pos_score_mean = tool_stats.pos_scores / max(tool_stats.pos_count, 1)
        neg_score_mean = tool_stats.neg_scores / max(tool_stats.neg_count, 1)
        bias = gamma * (pos_score_mean - neg_score_mean)

        learned[tool] = {
            "literal_weight": round(max(literal_weight, 0.05), 6),
            "pattern_weight": round(max(pattern_weight, 0.05), 6),
            "bias": round(bias, 6),
        }
    return learned


def train(
    dataset_path: Path,
    output_path: Path,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, object]:
    rows = load_dataset(dataset_path)
    detector = KeywordDetector(
        tool_keywords=TOOL_KEYWORDS,
        asset_aliases=ASSET_ALIASES,
        blockchain_aliases=BLOCKCHAIN_ALIASES,
        timeframe_keywords=TIMEFRAME_KEYWORDS,
        llm_weight=0.0,
        learned_weights={},
        weights_path=Path("__nonexistent__"),
    )

    stats: Dict[str, ToolStats] = {tool: ToolStats() for tool in TOOL_KEYWORDS}
    for row in rows:
        prompt: str = row["prompt"]  # type: ignore[assignment]
        targets = {tool for tool in row["tools"]}  # type: ignore[arg-type]
        detection = detector.detect(prompt)
        for tool, spec in TOOL_KEYWORDS.items():
            literal_hits = len(detection.matched_terms.get(tool, []))
            pattern_hits = len(detection.matched_patterns.get(tool, []))
            score = float(detection.scores.get(tool, 0.0))
            tool_stat = stats[tool]
            if tool in targets:
                tool_stat.pos_literals += literal_hits
                tool_stat.pos_patterns += pattern_hits
                tool_stat.pos_scores += score
                tool_stat.pos_count += 1
            else:
                tool_stat.neg_literals += literal_hits
                tool_stat.neg_patterns += pattern_hits
                tool_stat.neg_scores += score
                tool_stat.neg_count += 1

    learned_weights = compute_adjusted_weights(stats, alpha, beta, gamma)
    payload = {
        "metadata": {
            "dataset": str(dataset_path),
            "samples": len(rows),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
        "weights": learned_weights,
        "stats": {tool: tool_stats.as_dict() for tool, tool_stats in stats.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train keyword weight overrides from labeled prompt-to-tool data.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("prompts/router_training.jsonl"),
        help="Path to labeled training data (JSONL or JSON list).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pipeline" / "keyword_weights.json",
        help="Where to write the learned weights JSON.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Exponent for literal hit ratio.")
    parser.add_argument("--beta", type=float, default=0.5, help="Exponent for pattern hit ratio.")
    parser.add_argument("--gamma", type=float, default=0.25, help="Scaling factor for score bias.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train(args.dataset, args.output, args.alpha, args.beta, args.gamma)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
