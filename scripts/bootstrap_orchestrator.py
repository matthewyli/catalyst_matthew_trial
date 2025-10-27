from __future__ import annotations

"""
Bootstrap script for generating the orchestrator package.

Run once (idempotent) to lay down the scaffolding under ./pipeline/orchestrator/.
Safe to re-run; existing files are only overwritten when content changes.
"""

import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = ROOT / "pipeline" / "orchestrator"


DEFAULT_FILES: Dict[Path, str] = {}


def register(relative_path: str, content: str) -> None:
    DEFAULT_FILES[ORCHESTRATOR_DIR / relative_path] = content.lstrip("\n")


register(
    "__init__.py",
    """
from __future__ import annotations

\"\"\"Auto-generated orchestrator package.\"\"\"
""",
)

register(
    "config/__init__.py",
    """
from __future__ import annotations
""",
)

register(
    "config/tools_registry.yaml",
    """
# Tool registry configuration.
# Each entry declares an importable tool module and runtime metadata.
tools: []
""",
)

register(
    "core/__init__.py",
    """
from __future__ import annotations
""",
)

register(
    "core/exceptions.py",
    """
from __future__ import annotations


class OrchestratorError(Exception):
    \"\"\"Base orchestrator exception.\"\"\"


class ToolLoadError(OrchestratorError):
    \"\"\"Raised when a tool cannot be imported or validated.\"\"\"


class PipelineExecutionError(OrchestratorError):
    \"\"\"Raised when pipeline execution fails.\"\"\"
""",
)

register(
    "core/weights.py",
    """
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


class WeightManager:
    \"\"\"Persist and compute usage-derived weights for tools.\"\"\"

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
            data = json.loads(self.storage_path.read_text(encoding=\"utf-8\"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        self._counts = {str(k): int(v) for k, v in data.items()}

    def _save(self) -> None:
        payload = {name: int(count) for name, count in self._counts.items()}
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")

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
""",
)

register(
    "core/loader.py",
    """
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from .exceptions import ToolLoadError


@dataclass
class ToolSpec:
    name: str
    module: str
    object: str
    enabled: bool = True
    phases: List[str] | None = None


class ToolLoader:
    \"\"\"Load and validate tools declared in YAML configuration.\"\"\"

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
            raise ToolLoadError(f\"Registry file missing: {self.config_path}\")
        data = yaml.safe_load(self.config_path.read_text(encoding=\"utf-8\")) or {}
        tools = data.get(\"tools\", [])
        specs: List[ToolSpec] = []
        for entry in tools:
            try:
                module = entry[\"module\"]
                obj = entry[\"object\"]
                name = entry.get(\"name\") or entry.get(\"id\") or obj
                enabled = bool(entry.get(\"enabled\", True))
                phases = entry.get(\"phases\") or None
            except KeyError as exc:
                raise ToolLoadError(f\"Invalid tool entry: {entry}\") from exc
            specs.append(ToolSpec(name=name, module=module, object=obj, enabled=enabled, phases=phases))
        return specs

    def _instantiate(self, spec: ToolSpec) -> object:
        try:
            module = importlib.import_module(spec.module)
        except ImportError as exc:
            raise ToolLoadError(f\"Failed to import module '{spec.module}' for tool '{spec.name}'\") from exc
        try:
            obj = getattr(module, spec.object)
        except AttributeError as exc:
            raise ToolLoadError(f\"Module '{spec.module}' missing attribute '{spec.object}'\") from exc
        if hasattr(obj, \"__call__\") and not isinstance(obj, type):
            # Callable factory
            instance = obj()
        elif isinstance(obj, type):
            instance = obj()
        else:
            instance = obj
        meta = getattr(obj, \"__TOOL_META__\", None) or getattr(instance, \"__TOOL_META__\", None)
        if not isinstance(meta, dict):
            raise ToolLoadError(f\"Tool '{spec.name}' missing __TOOL_META__ dictionary.\")
        if meta.get(\"name\") and meta[\"name\"] != spec.name:
            raise ToolLoadError(
                f\"Name mismatch for tool '{spec.name}': meta name {meta['name']}\"  # type: ignore[index]
            )
        return instance
""",
)

register(
    "core/router.py",
    """
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


@dataclass
class RouteDecision:
    tools: List[str]
    keywords: Set[str]
    assets: List[str]


class KeywordRouter:
    \"\"\"Keyword-driven router leveraging tool metadata declarations.\"\"\"

    UPPER_TICKER = re.compile(r\"\\b[A-Z]{2,6}\\b\")

    def __init__(self, registry: Dict[str, object]) -> None:
        self.registry = registry
        self.keyword_map: Dict[str, Set[str]] = {}
        for name, obj in registry.items():
            meta = getattr(obj, \"__TOOL_META__\", {})
            keywords = set(k.lower() for k in meta.get(\"keywords\", []))
            for kw in keywords:
                self.keyword_map.setdefault(kw, set()).add(name)

    def route(self, prompt: str) -> RouteDecision:
        lowered = prompt.lower()
        hit_keywords: Set[str] = set()
        tool_hits: Set[str] = set()
        for keyword, tools in self.keyword_map.items():
            if keyword in lowered:
                hit_keywords.add(keyword)
                tool_hits.update(tools)
        assets = self._detect_assets(prompt)
        return RouteDecision(tools=sorted(tool_hits), keywords=hit_keywords, assets=assets)

    def _detect_assets(self, prompt: str) -> List[str]:
        seen: Set[str] = set()
        assets: List[str] = []
        for match in self.UPPER_TICKER.findall(prompt):
            if match not in seen:
                assets.append(match)
                seen.add(match)
        return assets
""",
)

register(
    "core/pipeline.py",
    """
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import PipelineExecutionError
from .router import KeywordRouter, RouteDecision
from .weights import WeightManager


@dataclass
class PipelineResult:
    prompt: str
    tools_invoked: List[Dict[str, Any]]
    selected_tools: List[str]
    keywords: List[str]
    assets: List[str]
    weights: Dict[str, float]
    weights_updated: Dict[str, float]
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Pipeline:
    \"\"\"Simple orchestrator pipeline that routes prompts and executes tools.\"\"\"

    def __init__(self, registry: Dict[str, object], weight_manager: WeightManager) -> None:
        self.registry = registry
        self.weights = weight_manager
        self.router = KeywordRouter(registry)
        self.weights.ensure(self.registry.keys())

    def run(self, prompt: str, *, asset: Optional[str] = None) -> PipelineResult:
        decision = self.router.route(prompt)
        selected = decision.tools or list(self.registry.keys())
        if asset and asset.upper() not in decision.assets:
            decision.assets.insert(0, asset.upper())

        weights_before = self.weights.weights(self.registry.keys())
        tool_runs: List[Dict[str, Any]] = []
        executed_names: List[str] = []
        context = {
            \"prompt\": prompt,
            \"assets\": decision.assets,
            \"weights\": weights_before,
        }

        for name in selected:
            tool = self.registry.get(name)
            if tool is None:
                tool_runs.append(
                    {
                        \"name\": name,
                        \"summary\": f\"ALERT: need {name} tool\",
                        \"payload\": None,
                    }
                )
                continue
            executor = getattr(tool, \"execute\", None)
            if not callable(executor):
                tool_runs.append(
                    {
                        \"name\": name,
                        \"summary\": f\"Tool '{name}' missing execute() method.\",
                        \"payload\": None,
                    }
                )
                continue
            try:
                result = executor(prompt=prompt, context=context)
            except TypeError:
                # Support our BaseTool signature (prompt, context)
                try:
                    result = executor(prompt, context)
                except Exception as exc:  # pragma: no cover - runtime path
                    result = {
                        \"summary\": f\"{name} failed: {exc}\",
                        \"payload\": {\"error\": str(exc)},
                    }
            except Exception as exc:  # pragma: no cover - runtime path
                result = {
                    \"summary\": f\"{name} failed: {exc}\",
                    \"payload\": {\"error\": str(exc)},
                }

            if isinstance(result, dict):
                summary = result.get(\"summary\") or result.get(\"message\") or \"\"
                payload = result.get(\"payload\")
            elif hasattr(result, \"summary\") and hasattr(result, \"payload\"):
                summary = result.summary
                payload = result.payload
            else:
                summary = str(result)
                payload = None

            tool_runs.append(
                {
                    \"name\": name,
                    \"summary\": summary,
                    \"payload\": payload,
                }
            )
            executed_names.append(name)

        if executed_names:
            self.weights.increment(executed_names)
        weights_after = self.weights.weights(self.registry.keys())

        return PipelineResult(
            prompt=prompt,
            tools_invoked=tool_runs,
            selected_tools=selected,
            keywords=sorted(decision.keywords),
            assets=decision.assets,
            weights=weights_before,
            weights_updated=weights_after,
        )
""",
)

register(
    "main.py",
    """
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core.loader import ToolLoader
from .core.pipeline import Pipeline
from .core.weights import WeightManager

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / \"config\" / \"tools_registry.yaml\"
WEIGHTS_PATH = PACKAGE_ROOT / \"state\" / \"weights.json\"


def build_pipeline() -> Pipeline:
    loader = ToolLoader(CONFIG_PATH)
    registry = loader.load()
    manager = WeightManager(WEIGHTS_PATH)
    return Pipeline(registry, manager)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=\"Run orchestrator pipeline on a natural language prompt.\")
    parser.add_argument(\"prompt\", type=str, help=\"Strategy prompt to analyze.\")
    parser.add_argument(\"--asset\", type=str, default=None, help=\"Explicit asset symbol override.\")
    parser.add_argument(\"--json\", action=\"store_true\", help=\"Emit JSON output.\")
    args = parser.parse_args(argv)

    pipeline = build_pipeline()
    result = pipeline.run(args.prompt, asset=args.asset)
    payload = result.as_dict()
    output = json.dumps(payload, indent=2 if args.json else None)
    print(output)
    return 0


if __name__ == \"__main__\":  # pragma: no cover
    sys.exit(main())
""",
)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
    path.write_text(content, encoding="utf-8")


def bootstrap() -> None:
    for path, content in DEFAULT_FILES.items():
        write_file(path, content)
    state_dir = ORCHESTRATOR_DIR / "state"
    state_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    bootstrap()
    print(f"Orchestrator scaffold written to {ORCHESTRATOR_DIR}")
