from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass
class ExecutionRequest:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResponse:
    accepted: bool
    dry_run: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ExecutionAdapter:
    """Common execution interface shared by paper and live adapters."""

    mode: str = "base"

    def submit(self, *, orders: Sequence[ExecutionRequest], context: Mapping[str, Any]) -> ExecutionResponse:
        raise NotImplementedError


class PaperExecutionAdapter(ExecutionAdapter):
    mode = "paper"

    def submit(self, *, orders: Sequence[ExecutionRequest], context: Mapping[str, Any]) -> ExecutionResponse:
        details = {
            "orders": [order.__dict__ for order in orders],
            "context_snapshot": dict(context),
        }
        return ExecutionResponse(
            accepted=True,
            dry_run=True,
            message="Paper execution accepted; no live orders sent.",
            details=details,
        )


class LiveExecutionAdapter(ExecutionAdapter):
    mode = "live"

    def __init__(
        self,
        *,
        dry_run: bool = True,
        enabled: bool = False,
        max_notional: float = 0.0,
        allowed_symbols: Iterable[str] | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.enabled = enabled
        self.max_notional = max_notional
        self.allowed_symbols = {sym.upper() for sym in (allowed_symbols or [])}

    def submit(self, *, orders: Sequence[ExecutionRequest], context: Mapping[str, Any]) -> ExecutionResponse:
        violations: list[str] = []
        normalized_orders: list[Dict[str, Any]] = []

        for order in orders:
            order_dict = order.__dict__.copy()
            notional = (order.price or 0.0) * order.quantity
            order_dict["notional"] = notional

            if self.allowed_symbols and order.symbol.upper() not in self.allowed_symbols:
                violations.append(f"symbol {order.symbol} not allowed")
            if self.max_notional and notional > self.max_notional:
                violations.append(f"order notional {notional:.2f} exceeds max {self.max_notional:.2f}")

            normalized_orders.append(order_dict)

        details = {
            "orders": normalized_orders,
            "violations": violations,
            "context_snapshot": dict(context),
            "dry_run": self.dry_run or not self.enabled,
        }

        if not self.enabled:
            return ExecutionResponse(
                accepted=False,
                dry_run=True,
                message="Live execution disabled; no orders sent.",
                details=details,
            )

        if violations:
            return ExecutionResponse(
                accepted=False,
                dry_run=True,
                message="Live execution blocked due to policy violations.",
                details=details,
            )

        if self.dry_run:
            return ExecutionResponse(
                accepted=True,
                dry_run=True,
                message="Live adapter in dry-run mode; orders logged only.",
                details=details,
            )

        # Placeholder for real exchange integration.
        details["note"] = "Live execution would be dispatched here."
        return ExecutionResponse(
            accepted=True,
            dry_run=False,
            message="Live orders accepted.",
            details=details,
        )


def get_execution_adapter_from_env() -> ExecutionAdapter:
    mode = os.getenv("EXECUTION_MODE", "paper").lower()
    if mode == "live":
        dry_run = os.getenv("EXECUTION_LIVE_DRY_RUN", "true").lower() != "false"
        enabled = os.getenv("EXECUTION_LIVE_ENABLED", "false").lower() == "true"
        max_notional_raw = os.getenv("EXECUTION_LIVE_MAX_NOTIONAL", "0")
        try:
            max_notional = float(max_notional_raw)
        except ValueError:
            max_notional = 0.0
        allowed = os.getenv("EXECUTION_LIVE_ALLOWED_SYMBOLS")
        allowed_symbols = [sym.strip().upper() for sym in allowed.split(",")] if allowed else None
        return LiveExecutionAdapter(
            dry_run=dry_run,
            enabled=enabled,
            max_notional=max_notional,
            allowed_symbols=allowed_symbols,
        )
    return PaperExecutionAdapter()

