from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


class WeightManager:
    """Persist and compute usage-derived weights for tools."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._counts: Dict[str, int] = {}
        self._load()

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
        self._counts = {str(k): int(v) for k, v in data.items()}

    def _save(self) -> None:
        payload = {name: int(count) for name, count in self._counts.items()}
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ensure(self, names: Iterable[str]) -> None:
        changed = False
        for name in names:
            if name not in self._counts:
                self._counts[name] = 0
                changed = True
        if changed:
            self._save()

    def increment(self, names: Sequence[str]) -> None:
        if not names:
            return
        for name in names:
            self._counts[name] = self._counts.get(name, 0) + 1
        self._save()

    def weights(self, scope: Optional[Iterable[str]] = None) -> Dict[str, float]:
        if scope is None:
            items = list(self._counts.items())
        else:
            items = [(name, self._counts.get(name, 0)) for name in scope]
        total = sum(count for _, count in items)
        if total == 0:
            n = len(items)
            return {name: (1.0 / n if n else 0.0) for name, _ in items}
        return {name: count / total for name, count in items}

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)
