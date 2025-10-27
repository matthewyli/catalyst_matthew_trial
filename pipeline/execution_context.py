from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass
class ExecutionContext:
    """Shared execution context flowing through all pipeline phases."""

    symbols: Sequence[str]
    primary_symbol: Optional[str]
    timeframe_minutes: Optional[int] = None
    data_sources: Dict[str, Any] = field(default_factory=dict)
    risk_constraints: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbols = list(self.symbols)
        self._state: Dict[str, Any] = {
            "inputs": {
                "symbols": self.symbols,
                "primary_symbol": self.primary_symbol,
                "timeframe_minutes": self.timeframe_minutes,
            },
            "data_sources": self.data_sources,
            "risk_constraints": self.risk_constraints or {},
            "params": self.params,
            "phase_outputs": self.phase_outputs,
        }

    # ------------------------------------------------------------------ state access
    @property
    def state(self) -> Dict[str, Any]:
        """Return live shared state (mutations are reflected everywhere)."""
        return self._state

    # ------------------------------------------------------------------ inputs / params
    def set_timeframe(self, minutes: Optional[int]) -> None:
        self.timeframe_minutes = minutes
        self._state["inputs"]["timeframe_minutes"] = minutes

    def set_symbols(self, symbols: Sequence[str], *, primary: Optional[str] = None) -> None:
        self.symbols = list(symbols)
        self._state["inputs"]["symbols"] = self.symbols
        if primary is not None:
            self.primary_symbol = primary
            self._state["inputs"]["primary_symbol"] = primary

    def update_data_source(self, name: str, info: Any) -> None:
        self.data_sources[name] = info

    def update_risk_constraints(self, constraints: Mapping[str, Any]) -> None:
        self.risk_constraints.update(constraints)
        self._state["risk_constraints"] = self.risk_constraints

    def risk_constraints_section(self) -> Dict[str, Any]:
        section = self._state.setdefault("risk_constraints", {})
        if not isinstance(section, dict):
            section = {}
            self._state["risk_constraints"] = section
        return section

    def add_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def params_section(self, key: str) -> MutableMapping[str, Any]:
        section = self.params.setdefault(key, {})
        if not isinstance(section, dict):
            section = {}
            self.params[key] = section
        return section

    # ------------------------------------------------------------------ phase outputs
    def update_phase_output(self, phase: str, tool_name: str, payload: Any) -> None:
        self.phase_outputs.setdefault(phase, {})[tool_name] = payload

    def add_phase_note(self, phase: str, note: str) -> None:
        notes = self.phase_outputs.setdefault(phase, {}).setdefault("_notes", [])
        if isinstance(notes, list):
            notes.append(note)
        else:  # defensive: replace non-list notes
            self.phase_outputs[phase]["_notes"] = [note]
