from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Sequence


def _default_cache_dir() -> Path:
    root = Path(os.getenv("CACHE_DIR", "cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_cache_key(*parts: object) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def force_refresh_from_env() -> bool:
    return os.getenv("CACHE_FORCE_REFRESH", "false").lower() == "true"


class CacheManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _default_cache_dir()

    def _path_for(self, key: str) -> Path:
        return self.base_dir / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def is_fresh(self, key: str, ttl_seconds: float | None) -> bool:
        if ttl_seconds is None:
            return self.exists(key)
        path = self._path_for(key)
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age <= ttl_seconds

    def load(self, key: str) -> Any:
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)

    def save(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def get_or_set(
        self,
        key_parts: Sequence[object],
        loader: Callable[[], Any],
        *,
        ttl_seconds: float | None = None,
        force_refresh: bool = False,
    ) -> Any:
        key = make_cache_key(*key_parts)
        if not force_refresh and self.is_fresh(key, ttl_seconds):
            cached = self.load(key)
            if cached is not None:
                return cached
        value = loader()
        self.save(key, value)
        return value

