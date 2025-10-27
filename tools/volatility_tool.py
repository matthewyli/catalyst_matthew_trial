from __future__ import annotations

import importlib
import os
from typing import Dict, Optional, Sequence

from cache import CacheManager, force_refresh_from_env
from .base import BaseTool, ToolContext, ToolExecutionError, ToolResult

__TOOL_META__ = {
    "name": "volatility_percentile",
    "module": "tools.volatility_tool",
    "object": "VolatilityPercentileTool",
    "version": "1.0",
    "description": "Computes realized volatility percentiles from Mobula price history.",
    "author": "auto",
    "keywords": [
        "vol",
        "volatility",
        "variance",
        "risk",
        "sigma",
        "percentile",
    ],
    "phases": ["feature_engineering", "risk_sizing"],
    "outputs": ["vol_percentile", "vol_percentile_conf"],
}


def _resolve_ttl(default_seconds: float) -> float:
    raw = os.getenv("CACHE_TTL_SECONDS_VOL") or os.getenv("CACHE_TTL_SECONDS")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default_seconds


_CACHE = CacheManager()
_TTL_SECONDS = _resolve_ttl(1800.0)


class VolatilityPercentileTool(BaseTool):
    """Measure realized volatility regime relative to historical distribution."""

    name = "volatility_percentile"
    description = "Computes realized volatility percentiles from Mobula price history."
    keywords = frozenset(
        {
            "vol",
            "volatility",
            "variance",
            "risk",
            "sigma",
            "volatility percentile",
        }
    )

    def __init__(
        self,
        *,
        freq: str = "h",
        windows: Sequence[int] = (7, 14, 30),
        lookback_days: int = 365,
    ) -> None:
        super().__init__()
        self.freq = freq
        self.windows = tuple(int(w) for w in windows)
        self.lookback_days = int(lookback_days)

    def _load_module(self):
        try:
            module = importlib.import_module("tools.legacy.vol_pctile")
        except ImportError as exc:
            raise ToolExecutionError(
                "vol_pctile (and dependencies pandas, numpy, requests) is required for volatility analysis."
            ) from exc
        required = (
            "blend_percentiles",
            "compute_confidence",
            "fetch_prices_mobula",
            "realized_vol_percentile",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ToolExecutionError(f"vol_pctile module missing expected exports: {missing}")
        return module

    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        asset = (context.asset or "").upper()
        if not asset:
            raise ToolExecutionError("VolatilityPercentileTool requires an asset symbol in the prompt or metadata.")

        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        strict_mode = bool(options.get("strict_io"))

        module = self._load_module()
        fetch_prices_mobula = module.fetch_prices_mobula
        realized_vol_percentile = module.realized_vol_percentile
        blend_percentiles = module.blend_percentiles
        compute_confidence = module.compute_confidence

        router_meta = context.metadata.get("params", {}).get("router", {}).get("metadata", {})
        blockchain = router_meta.get("blockchain")
        mobula_key = context.metadata.get("mobula_api_key", os.environ.get("MOBULA_API_KEY", ""))
        fetch_days = self.lookback_days + max(self.windows)

        options = context.metadata.get("params", {}).get("options", {})
        force_refresh = bool(options.get("force_refresh")) or force_refresh_from_env()

        symbol_query = asset
        blockchain_param = None
        if isinstance(blockchain, str):
            if asset.startswith("0X"):
                blockchain_param = blockchain
            else:
                symbol_query = blockchain

        def loader():
            return fetch_prices_mobula(symbol_query, self.freq, fetch_days, mobula_key, blockchain_param)

        try:
            df = _CACHE.get_or_set(
                ("mobula_prices", symbol_query, self.freq, fetch_days, blockchain_param or ""),
                loader,
                ttl_seconds=_TTL_SECONDS,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Mobula price fetch failed: {exc}") from exc

        if df is None or df.empty or "close" not in df:
            raise ToolExecutionError("Mobula response did not contain pricing data.")

        results = []
        for window in self.windows:
            try:
                rv, pctl, cov = realized_vol_percentile(
                    df["close"],
                    w=window,
                    lb_days=self.lookback_days,
                    freq=self.freq,
                )
                results.append(
                    {
                        "w": int(window),
                        "realized_vol": rv,
                        "pctile": pctl,
                        "confidence": cov,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "w": int(window),
                        "error": str(exc),
                    }
                )

        score, score_conf = blend_percentiles(results)
        conf_scalar, conf_parts = compute_confidence(
            df,
            results,
            self.freq,
            self.lookback_days,
            list(self.windows),
        )

        details: Dict[str, Optional[float]] = {
            f"pctile_{int(r['w'])}d": r.get("pctile") for r in results if "pctile" in r
        }

        if score is None:
            summary = f"{asset} volatility percentile unavailable"
        else:
            conf_text = f"{score_conf:.0%}" if score_conf is not None else "n/a"
            summary = f"{asset} volatility ~{score:.1f}th percentile (confidence {conf_text})"

        raw_payload = {
            "asset": asset,
            "windows": list(self.windows),
            "results": results,
            "vol_percentile": score,
            "vol_percentile_conf": score_conf,
            "confidence_scalar": conf_scalar,
            "confidence_components": conf_parts,
            "mobula_key_present": bool(mobula_key),
            **details,
        }

        if score is None:
            strict_value = 0.0
        else:
            strict_value = (50.0 - float(score)) / 100.0

        payload = {"value": strict_value, "raw": raw_payload} if strict_mode else raw_payload
        weight = float(context.weights.get(self.name, 0.0))
        return ToolResult(
            name=self.name,
            weight=weight,
            summary=summary,
            payload=payload,
        )


VolatilityPercentileTool.__TOOL_META__ = __TOOL_META__
