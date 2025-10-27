from __future__ import annotations

import importlib
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from .exceptions import ToolLoadError


@dataclass
class ToolSpec:
    name: str
    module: str
    object: str
    enabled: bool = True
    phases: List[str] | None = None


class ToolLoader:
    """Load and validate tools declared in YAML configuration."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def load(self) -> Dict[str, object]:
        specs = self._read_specs()
        registry: Dict[str, object] = {}
        for spec in specs:
            if not spec.enabled:
                continue
            instance = self._instantiate(spec)
            registry[spec.name] = instance
        return registry

    def _read_specs(self) -> Iterable[ToolSpec]:
        if not self.config_path.exists():
            raise ToolLoadError(f"Registry file missing: {self.config_path}")
        data = self._load_yaml()
        tools = data.get("tools", [])
        specs: List[ToolSpec] = []
        for entry in tools:
            try:
                module = entry["module"]
                obj = entry["object"]
                name = entry.get("name") or entry.get("id") or obj
                enabled = bool(entry.get("enabled", True))
                phases = entry.get("phases") or None
            except KeyError as exc:
                raise ToolLoadError(f"Invalid tool entry: {entry}") from exc
            specs.append(ToolSpec(name=name, module=module, object=obj, enabled=enabled, phases=phases))
        return specs

    def _load_yaml(self) -> Dict[str, object]:
        text = self.config_path.read_text(encoding="utf-8")
        if yaml:
            return yaml.safe_load(text) or {}
        return self._parse_minimal_yaml(text)

    def _parse_minimal_yaml(self, text: str) -> Dict[str, object]:
        tools: List[Dict[str, object]] = []
        current: Dict[str, object] | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("tools:"):
                if current:
                    tools.append(current)
                    current = None
                continue
            if line.startswith("- "):
                if current:
                    tools.append(current)
                current = {}
                line = line[2:].strip()
                if not line:
                    continue
            if ":" not in line or current is None:
                continue
            key, value = line.split(":", 1)
            parsed_value = self._parse_value(value.strip())
            current[key.strip()] = parsed_value
        if current:
            tools.append(current)
        return {"tools": tools}

    def _parse_value(self, value: str) -> object:
        if value == "":
            return ""
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [item.strip().strip("'\"") for item in inner.split(",")]
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1]
        try:
            return literal_eval(value)
        except Exception:
            return value

    def _instantiate(self, spec: ToolSpec) -> object:
        try:
            module = importlib.import_module(spec.module)
        except ImportError as exc:
            raise ToolLoadError(f"Failed to import module '{spec.module}' for tool '{spec.name}'") from exc
        try:
            obj = getattr(module, spec.object)
        except AttributeError as exc:
            raise ToolLoadError(f"Module '{spec.module}' missing attribute '{spec.object}'") from exc
        if hasattr(obj, "__call__") and not isinstance(obj, type):
            # Callable factory
            instance = obj()
        elif isinstance(obj, type):
            instance = obj()
        else:
            instance = obj
        meta = getattr(obj, "__TOOL_META__", None) or getattr(instance, "__TOOL_META__", None)
        if not isinstance(meta, dict):
            raise ToolLoadError(f"Tool '{spec.name}' missing __TOOL_META__ dictionary.")
        if meta.get("name") and meta["name"] != spec.name:
            raise ToolLoadError(
                f"Name mismatch for tool '{spec.name}': meta name {meta['name']}"  # type: ignore[index]
            )
        return instance
