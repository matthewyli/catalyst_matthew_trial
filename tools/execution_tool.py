from __future__ import annotations

import json
from typing import Any, Dict, List

from execution import ExecutionRequest, ExecutionResponse, get_execution_adapter_from_env
from .base import BaseTool, ToolContext, ToolExecutionError, ToolResult


class ExecutionAdapterTool(BaseTool):
    """Adapter that sends orders via paper or live execution backends."""

    name = "execution_adapter"
    description = "Routes generated orders to paper or live execution adapters."
    keywords = frozenset({"execute", "order", "trade", "fill"})

    __TOOL_META__ = {
        "name": name,
        "module": "tools.execution_tool",
        "object": "ExecutionAdapterTool",
        "phases": ["execution"],
        "outputs": ["execution_log"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.adapter = get_execution_adapter_from_env()

    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        strict_mode = bool(options.get("strict_io"))

        runtime = params.get("runtime", {}) if isinstance(params, dict) else {}
        phase = runtime.get("current_phase", "execution")
        phase_outputs = context.metadata.get("phase_outputs", {})
        orders_payload = phase_outputs.get("signal_generation", {}).get("orders")

        if not orders_payload:
            payload = {
                "phase": phase,
                "mode": self.adapter.mode,
                "executed": False,
                "reason": "empty_orders",
            }
            if strict_mode:
                payload = {"value": 0.0, "raw": payload}
            return ToolResult(
                name=self.name,
                weight=context.weights.get(self.name, 0.0),
                summary="No orders to execute.",
                payload=payload,
            )

        requests = self._normalize_orders(orders_payload)
        risk_constraints = context.metadata.get("risk_constraints", {}) or {}
        safe, reason, risk_details = self._check_risk(requests, risk_constraints)
        if not safe:
            raw_payload = {
                "phase": phase,
                "mode": self.adapter.mode,
                "executed": False,
                "dry_run": True,
                "message": reason,
                "details": {
                    "risk": risk_details,
                },
            }
            payload = {"value": 0.0, "raw": raw_payload} if strict_mode else raw_payload
            return ToolResult(
                name=self.name,
                weight=context.weights.get(self.name, 0.0),
                summary=reason,
                payload=payload,
            )

        response = self.adapter.submit(orders=requests, context=context.metadata)

        raw_payload = {
            "phase": phase,
            "mode": self.adapter.mode,
            "executed": response.accepted and not response.dry_run,
            "dry_run": response.dry_run,
            "message": response.message,
            "details": response.details,
            "risk": risk_details,
        }
        if strict_mode:
            value = 1.0 if raw_payload.get("executed") else 0.0
            payload = {"value": value, "raw": raw_payload}
        else:
            payload = raw_payload
        return ToolResult(
            name=self.name,
            weight=context.weights.get(self.name, 0.0),
            summary=response.message,
            payload=payload,
        )

    def _normalize_orders(self, raw: Any) -> List[ExecutionRequest]:
        requests: List[ExecutionRequest] = []
        if isinstance(raw, dict):
            raw = raw.get("orders") or []
        if not isinstance(raw, list):
            raise ToolExecutionError("orders payload must be a list or dict with 'orders'.")
        for entry in raw:
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except json.JSONDecodeError as exc:
                    raise ToolExecutionError(f"invalid order json: {entry}") from exc
            if not isinstance(entry, dict):
                raise ToolExecutionError(f"order entry must be a dict, got {type(entry)}")
            try:
                requests.append(
                    ExecutionRequest(
                        symbol=str(entry["symbol"]).upper(),
                        side=str(entry["side"]).lower(),
                        quantity=float(entry["quantity"]),
                        price=float(entry["price"]) if entry.get("price") is not None else None,
                        meta={k: v for k, v in entry.items() if k not in {"symbol", "side", "quantity", "price"}},
                    )
                )
            except KeyError as exc:
                raise ToolExecutionError(f"missing order field: {exc}") from exc
        return requests

    def _check_risk(
        self,
        requests: List[ExecutionRequest],
        constraints: Dict[str, Any],
    ) -> tuple[bool, str, Dict[str, Any]]:
        constraints = dict(constraints)
        kill_switch = constraints.get("kill_switch")
        if kill_switch:
            return False, "Kill switch active; execution halted.", {"reason": "kill_switch"}

        max_drawdown = constraints.get("max_drawdown")
        current_drawdown = constraints.get("current_drawdown")
        if (
            isinstance(max_drawdown, (int, float))
            and isinstance(current_drawdown, (int, float))
            and current_drawdown >= max_drawdown
        ):
            return (
                False,
                "Max drawdown reached; execution blocked.",
                {"reason": "drawdown_limit", "current_drawdown": current_drawdown, "max_drawdown": max_drawdown},
            )

        default_price = constraints.get("default_price") or 0.0
        position_limits = {str(k).upper(): float(v) for k, v in (constraints.get("position_limits") or {}).items()}
        open_positions = {str(k).upper(): float(v) for k, v in (constraints.get("open_positions") or {}).items()}
        global_position = float(constraints.get("max_position", 0.0) or 0.0)
        open_notional = float(constraints.get("open_notional", 0.0) or 0.0)

        portfolio_equity = constraints.get("portfolio_equity")
        max_leverage = constraints.get("max_leverage")

        violations: List[str] = []
        total_new_notional = 0.0

        for req in requests:
            price = req.price if req.price is not None else default_price
            notional = abs(price * req.quantity) if price else abs(req.quantity)
            total_new_notional += notional

            symbol = req.symbol.upper()
            symbol_limit = position_limits.get(symbol)
            open_symbol = open_positions.get(symbol, 0.0)
            if symbol_limit is not None and open_symbol + notional > symbol_limit:
                violations.append(
                    f"symbol_limit:{symbol} current={open_symbol:.2f} + new={notional:.2f} > limit={symbol_limit:.2f}"
                )
            if global_position and notional > global_position:
                violations.append(f"global_position notional {notional:.2f} exceeds limit {global_position:.2f}")

        if isinstance(max_leverage, (int, float)) and isinstance(portfolio_equity, (int, float)) and portfolio_equity > 0:
            total_notional = open_notional + total_new_notional
            leverage = total_notional / portfolio_equity
            if leverage > max_leverage:
                violations.append(
                    f"leverage {leverage:.2f} exceeds cap {max_leverage:.2f} (equity={portfolio_equity:.2f})"
                )

        if violations:
            return False, "Risk limits violated; execution blocked.", {"violations": violations}

        return True, "Risk checks passed.", {
            "total_new_notional": total_new_notional,
            "open_notional": open_notional,
        }
