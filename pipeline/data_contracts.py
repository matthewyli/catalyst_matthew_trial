from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


Validator = Callable[[Any], Tuple[bool, str]]


def _require_mapping(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, f"expected mapping, got {type(payload).__name__}"
    return True, ""


def _validate_orders(payload: Any) -> Tuple[bool, str]:
    ok, msg = _require_mapping(payload)
    if not ok:
        return ok, msg
    orders = payload.get("orders")
    if orders is None:
        return True, ""
    if not isinstance(orders, Sequence):
        return False, "orders must be a sequence"
    required_fields = {"symbol", "side", "quantity"}
    for idx, order in enumerate(orders):
        if isinstance(order, str):
            # allow serialized orders; assume execution tool will parse
            continue
        if not isinstance(order, Mapping):
            return False, f"orders[{idx}] must be a mapping"
        missing = required_fields - order.keys()
        if missing:
            return False, f"orders[{idx}] missing required fields: {', '.join(sorted(missing))}"
        try:
            float(order["quantity"])
        except (TypeError, ValueError):
            return False, f"orders[{idx}].quantity must be numeric"
    return True, ""


def _validate_risk(payload: Any) -> Tuple[bool, str]:
    ok, msg = _require_mapping(payload)
    if not ok:
        return ok, msg
    for key in ("position_limits", "open_positions"):
        section = payload.get(key)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            return False, f"{key} must be a mapping"
        for sym, value in section.items():
            try:
                float(value)
            except (TypeError, ValueError):
                return False, f"{key}[{sym}] must be numeric"
    return True, ""


def _validate_execution(payload: Any) -> Tuple[bool, str]:
    ok, msg = _require_mapping(payload)
    if not ok:
        return ok, msg
    for field in ("executed", "dry_run"):
        if field in payload and not isinstance(payload[field], bool):
            return False, f"{field} must be boolean"
    return True, ""


PHASE_CONTRACTS: Dict[str, Validator] = {
    "data_gather": _require_mapping,
    "feature_engineering": _require_mapping,
    "signal_generation": _validate_orders,
    "risk_sizing": _validate_risk,
    "execution": _validate_execution,
}


def validate_phase_output(phase: str, payload: Any) -> Tuple[bool, str]:
    validator = PHASE_CONTRACTS.get(phase.lower(), _require_mapping)
    return validator(payload)
