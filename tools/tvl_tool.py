from __future__ import annotations

import importlib
import os
from typing import Dict, Optional

from cache import CacheManager, force_refresh_from_env
from .base import BaseTool, ToolContext, ToolExecutionError, ToolResult

__TOOL_META__ = {
    "name": "tvl_growth",
    "module": "tools.tvl_tool",
    "object": "TVLGrowthTool",
    "version": "1.0",
    "description": "Analyses DeFiLlama TVL momentum to gauge liquidity trends.",
    "author": "auto",
    "keywords": [
        "tvl",
        "defi",
        "liquidity",
        "inflow",
        "outflow",
        "staking",
        "momentum",
    ],
    "phases": ["data_gather", "feature_engineering"],
    "outputs": ["latest_tvl", "pct_changes", "trailing_ratio"],
}


def _resolve_ttl(default_seconds: float) -> float:
    raw = os.getenv("CACHE_TTL_SECONDS_TVL") or os.getenv("CACHE_TTL_SECONDS")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default_seconds


_CACHE = CacheManager()
_TTL_SECONDS = _resolve_ttl(3600.0)


CHAIN_METRIC_MAP: Dict[str, str] = {
    "ETH": "chain:Ethereum",
    "SOL": "chain:Solana",
    "MATIC": "chain:Polygon",
    "ARB": "chain:Arbitrum",
    "OP": "chain:Optimism",
    "AVAX": "chain:Avalanche",
    "BNB": "chain:BNB Chain",
    "BASE": "chain:Base",
}


class TVLGrowthTool(BaseTool):
    """Fetch recent DeFiLlama TVL trends for popular chains."""

    name = "tvl_growth"
    description = "Analyses DeFiLlama TVL momentum for the selected asset/chain."
    keywords = frozenset(
        {
            "tvl",
            "defi",
            "liquidity",
            "inflow",
            "outflow",
            "staking",
        }
    )

    def __init__(self, *, default_metric: str = "defi_total") -> None:
        super().__init__()
        self.default_metric = default_metric

    def _resolve_metric(self, asset: Optional[str]) -> str:
        if asset:
            symbol = asset.upper()
            if symbol in CHAIN_METRIC_MAP:
                return CHAIN_METRIC_MAP[symbol]
        return self.default_metric

    def _pct_change(self, series, days: int) -> Optional[float]:
        if series.empty or len(series) <= days:
            return None
        recent = float(series.iloc[-1])
        past = float(series.iloc[-(days + 1)])
        if past == 0:
            return None
        return (recent - past) / past

    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        try:
            import pandas as pd  # noqa: F401  # pragma: no cover
        except ImportError:
            raise ToolExecutionError("TVLGrowthTool requires pandas (install via 'pip install pandas').")

        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        strict_mode = bool(options.get("strict_io"))

        try:
            module = importlib.import_module("tools.legacy.tvl_sync_growth")
            fetch_llama_series = getattr(module, "fetch_llama_series")
        except Exception as exc:  # pragma: no cover - runtime import failure
            raise ToolExecutionError(f"Unable to import tvl_sync_growth: {exc}") from exc

        metric = self._resolve_metric(context.asset)

        options = context.metadata.get("params", {}).get("options", {})
        force_refresh = bool(options.get("force_refresh")) or force_refresh_from_env()

        def loader():
            return fetch_llama_series(metric)

        try:
            series = _CACHE.get_or_set(
                ("tvl_series", metric),
                loader,
                ttl_seconds=_TTL_SECONDS,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            raise ToolExecutionError(f"TVL fetch failed for metric {metric}: {exc}") from exc

        series = series.dropna().sort_index()
        if series.empty:
            raise ToolExecutionError(f"No TVL data available for metric {metric}.")

        changes = {f"change_{h}d": self._pct_change(series, h) for h in (1, 7, 30, 90)}
        recent = float(series.iloc[-1])
        trailing_mean = series.iloc[-30:].mean() if len(series) >= 30 else series.mean()
        trailing_rate = None
        if trailing_mean and trailing_mean != 0:
            trailing_rate = float(recent / trailing_mean - 1.0)

        raw_payload = {
            "metric": metric,
            "latest_tvl": recent,
            "pct_changes": {k: round(v, 4) if v is not None else None for k, v in changes.items()},
            "trailing_ratio": round(trailing_rate, 4) if trailing_rate is not None else None,
        }

        growth_components = [v for v in changes.values() if v is not None]
        growth_score = float(sum(growth_components) / len(growth_components)) if growth_components else 0.0
        summary = (
            f"{metric} TVL {recent:,.0f} USD | "
            f"avg momentum {growth_score:.2%} "
            f"(1d/7d/30d change: "
            + ", ".join(
                f"{h.replace('change_', '').upper()}={changes[h]:.2%}" if changes[h] is not None else f"{h.replace('change_', '').upper()}=n/a"
                for h in ("change_1d", "change_7d", "change_30d")
            )
            + ")"
        )

        payload = {"value": growth_score, "raw": raw_payload} if strict_mode else raw_payload
        weight = float(context.weights.get(self.name, 0.0))
        return ToolResult(
            name=self.name,
            weight=weight,
            summary=summary,
            payload=payload,
        )


TVLGrowthTool.__TOOL_META__ = __TOOL_META__
