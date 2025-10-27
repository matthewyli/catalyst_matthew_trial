from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


class UsageTracker:
    """Persist tool usage counts and derive normalized weights with optional time decay."""

    def __init__(self, storage_path: Path, *, decay_half_life_hours: Optional[float] = None) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.decay_half_life_hours = decay_half_life_hours if decay_half_life_hours and decay_half_life_hours > 0 else None
        self._counts: Dict[str, Dict[str, Optional[str] | float]] = {}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.storage_path.exists():
            self._counts = {}
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        parsed: Dict[str, Dict[str, Optional[str] | float]] = {}
        for key, value in data.items():
            if isinstance(value, dict) and "count" in value:
                parsed[str(key)] = {
                    "count": float(value.get("count", 0.0) or 0.0),
                    "last_ts": value.get("last_ts"),
                }
            else:
                parsed[str(key)] = {"count": float(value or 0.0), "last_ts": None}
        self._counts = parsed

    def _save(self) -> None:
        payload = {
            name: {
                "count": float(record.get("count", 0.0) or 0.0),
                "last_ts": record.get("last_ts"),
            }
            for name, record in self._counts.items()
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ public API
    def ensure(self, names: Iterable[str]) -> None:
        changed = False
        for name in names:
            if name not in self._counts:
                self._counts[name] = {"count": 0.0, "last_ts": None}
                changed = True
        if changed:
            self._save()

    def increment(self, names: Sequence[str]) -> None:
        if not names:
            return
        now = datetime.now(timezone.utc)
        for name in names:
            record = self._counts.setdefault(name, {"count": 0.0, "last_ts": None})
            self._decayed_count(record, now, mutate=True)
            record["count"] = float(record.get("count", 0.0) or 0.0) + 1.0
            record["last_ts"] = now.isoformat()
        self._save()

    @property
    def counts(self) -> Mapping[str, float]:
        now = datetime.now(timezone.utc)
        return {
            name: self._decayed_count(dict(record), now, mutate=False)
            for name, record in self._counts.items()
        }

    def weights(self, scope: Optional[Iterable[str]] = None) -> Dict[str, float]:
        now = datetime.now(timezone.utc)
        if scope is None:
            names = list(self._counts.keys())
        else:
            names = list(scope)
            self.ensure(names)

        effective_counts: Dict[str, float] = {}
        for name in names:
            record = self._counts.setdefault(name, {"count": 0.0, "last_ts": None})
            effective_counts[name] = self._decayed_count(record, now, mutate=True)
        self._save()

        total = sum(effective_counts.values())
        if total <= 0:
            n = len(effective_counts)
            return {name: (1.0 / n if n else 0.0) for name in effective_counts}

        return {name: count / total for name, count in effective_counts.items()}

    # ------------------------------------------------------------------ helpers
    def _decayed_count(
        self,
        record: Dict[str, Optional[str] | float],
        now: datetime,
        *,
        mutate: bool,
    ) -> float:
        count = float(record.get("count", 0.0) or 0.0)
        if not self.decay_half_life_hours:
            if mutate and record.get("last_ts") is None:
                record["last_ts"] = now.isoformat()
            return count

        last_ts = record.get("last_ts")
        if not last_ts:
            if mutate:
                record["last_ts"] = now.isoformat()
            return count

        try:
            last_dt = datetime.fromisoformat(str(last_ts))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            last_dt = now

        delta_hours = max(0.0, (now - last_dt).total_seconds() / 3600.0)
        if delta_hours == 0 or not self.decay_half_life_hours:
            factor = 1.0
        else:
            factor = math.pow(0.5, delta_hours / self.decay_half_life_hours)

        decayed = count * factor
        if mutate:
            record["count"] = decayed
            record["last_ts"] = now.isoformat()
        return decayed
