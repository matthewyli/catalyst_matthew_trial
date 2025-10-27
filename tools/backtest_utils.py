from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _ensure_equity_curve(raw_curve: Any) -> List[Dict[str, float]]:
    if isinstance(raw_curve, list):
        if raw_curve and isinstance(raw_curve[0], dict):
            return [
                {"timestamp": float(point.get("timestamp", idx)), "equity": float(point.get("equity", point.get("value", 0.0)))}
                for idx, point in enumerate(raw_curve)
            ]
        if raw_curve and isinstance(raw_curve[0], (int, float)):
            return [{"timestamp": float(idx), "equity": float(value)} for idx, value in enumerate(raw_curve)]
    return []


def _ensure_metrics(raw_metrics: Any) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(raw_metrics, dict):
        for key, value in raw_metrics.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def _ensure_trades(raw_trades: Any) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    if isinstance(raw_trades, list):
        for entry in raw_trades:
            if isinstance(entry, dict):
                trade = dict(entry)
                trades.append(trade)
    return trades


def normalize_backtest_output(raw: Any) -> Dict[str, Any]:
    """Normalize a backtest payload to {equity_curve, metrics, trades}."""

    equity_curve = []
    metrics = {}
    trades = []

    if isinstance(raw, dict):
        equity_curve = _ensure_equity_curve(
            raw.get("equity_curve") or raw.get("curve") or raw.get("equity_curve_points")
        )
        metrics = _ensure_metrics(raw.get("metrics") or raw.get("stats") or raw.get("performance"))
        trades = _ensure_trades(raw.get("trades") or raw.get("positions") or raw.get("executions"))
        payload = dict(raw)
    elif isinstance(raw, list):
        equity_curve = _ensure_equity_curve(raw)
        payload = {}
    else:
        payload = {}

    payload["equity_curve"] = equity_curve
    payload["metrics"] = metrics
    payload["trades"] = trades

    return payload
