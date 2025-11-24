from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from integrations.textql_client import create_chat, TextQLClientError

from .base import BaseTool, ToolContext, ToolExecutionError, ToolResult

__PRIMER_META__ = {
    "name": "textql_primer",
    "module": "tools.textql_context_tool",
    "object": "TextQLPrimerTool",
    "version": "1.0",
    "description": "Bootstraps the pipeline with a TextQL-powered web + on-chain research plan.",
    "author": "auto",
    "keywords": [
        "textql",
        "search",
        "exa",
        "context",
        "plan",
        "macro",
        "hypothesis",
    ],
    "phases": ["data_gather"],
    "outputs": [
        "mission_summary",
        "macro_backdrop",
        "search_hypotheses",
        "dataset_requests",
        "web_sources",
    ],
}


def _as_list(value: Optional[Sequence[str]]) -> List[str]:
    return [str(item) for item in value or [] if item]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_textql_answer_blob(answer: str) -> Optional[Mapping[str, Any]]:
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    candidate = match.group(1).strip() if match else text
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except ValueError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


class TextQLPrimerTool(BaseTool):
    """Prime the agent with TextQL search + hypothesis scaffolding."""

    name = "textql_primer"
    description = "Calls TextQL (with Exa + data runners) to draft a search plan and context packet."
    keywords = frozenset(
        {
            "textql",
            "search",
            "exa",
            "macro",
            "hypothesis",
            "context",
            "research",
            "plan",
        }
    )

    def __init__(
        self,
        *,
        max_hypotheses: int = 4,
        timeout_seconds: float = 120.0,
        include_prompt_in_payload: bool = False,
    ) -> None:
        super().__init__()
        self.max_hypotheses = max(1, int(max_hypotheses))
        env_timeout = os.getenv("TEXTQL_TIMEOUT_SEC")
        if env_timeout:
            try:
                timeout_seconds = float(env_timeout)
            except ValueError:
                pass
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.include_prompt_in_payload = include_prompt_in_payload
        self.api_key = os.getenv("TEXTQL_API_KEY", "").strip()
        self.model = os.getenv("TEXTQL_MODEL", "MODEL_SONNET_4").strip() or "MODEL_SONNET_4"
        self.paradigm_type = os.getenv("TEXTQL_PARADIGM_TYPE", "TYPE_UNIVERSAL").strip() or "TYPE_UNIVERSAL"
        self.paradigm_version = _env_int("TEXTQL_PARADIGM_VERSION") or 1
        self.research_mode = _env_bool("TEXTQL_RESEARCH_MODE", False)
        self.sql_connector_id = _env_int("TEXTQL_SQL_CONNECTOR_ID")
        self.max_retries = max(0, _env_int("TEXTQL_MAX_RETRIES") or 1)
        retry_delay = os.getenv("TEXTQL_RETRY_DELAY_SEC")
        try:
            self.retry_delay_seconds = max(0.0, float(retry_delay)) if retry_delay else 5.0
        except ValueError:
            self.retry_delay_seconds = 5.0
        cache_path_env = os.getenv("TEXTQL_CACHE_PATH")
        if cache_path_env:
            self.cache_path = Path(cache_path_env).expanduser()
        else:
            self.cache_path = Path(os.getenv("TEXTQL_CACHE_DIR", "runs")) / "textql_direct_sample.json"
        try:
            self.cache_ttl_seconds = float(os.getenv("TEXTQL_CACHE_TTL_SEC", "3600"))
        except ValueError:
            self.cache_ttl_seconds = 3600.0
        self.log_dir = Path(os.getenv("TEXTQL_LOG_DIR", "runs")) / "textql_logs"
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._last_request_payload: Dict[str, Any] = {}
        self.connector_ids: List[int] = []
        if self.sql_connector_id:
            self.connector_ids.append(self.sql_connector_id)
        self.api_mode = os.getenv("TEXTQL_API_MODE", "RPC").strip().upper() or "RPC"
        self.rpc_url = os.getenv(
            "TEXTQL_RPC_URL",
            "https://app.textql.com/rpc/public/textql.rpc.public.chat.ChatService/QueryOneShot",
        ).strip()
        self.answer_url = os.getenv(
            "TEXTQL_ANSWER_URL",
            "https://app.textql.com/rpc/public/textql.rpc.public.chat.ChatService/GetAPIChatAnswer",
        ).strip()
        self.poll_interval_seconds = max(0.5, float(os.getenv("TEXTQL_POLL_INTERVAL_SEC", "3")))
        self.poll_max_attempts = max(1, int(os.getenv("TEXTQL_POLL_MAX_ATTEMPTS", "40")))
        poll_duration = os.getenv("TEXTQL_POLL_MAX_DURATION_SEC")
        try:
            self.poll_max_duration_seconds = max(0.0, float(poll_duration)) if poll_duration else 0.0
        except ValueError:
            self.poll_max_duration_seconds = 0.0
        try:
            backoff = float(os.getenv("TEXTQL_POLL_BACKOFF_MULTIPLIER", "1.25"))
        except ValueError:
            backoff = 1.25
        self.poll_backoff_multiplier = max(1.0, backoff)
        self.universal_flags = {
            "webSearchEnabled": _env_bool("TEXTQL_ENABLE_WEB_SEARCH", True),
            "sqlEnabled": _env_bool("TEXTQL_ENABLE_SQL", False),
            "ontologyEnabled": _env_bool("TEXTQL_ENABLE_ONTOLOGY", False),
            "experimentalEnabled": _env_bool("TEXTQL_ENABLE_EXPERIMENTAL", False),
            "tableauEnabled": _env_bool("TEXTQL_ENABLE_TABLEAU", False),
            "autoApproveEnabled": _env_bool("TEXTQL_ENABLE_AUTO_APPROVE", False),
            "pythonEnabled": _env_bool("TEXTQL_ENABLE_PYTHON", True),
            "streamlitEnabled": _env_bool("TEXTQL_ENABLE_STREAMLIT", False),
            "googleDriveEnabled": _env_bool("TEXTQL_ENABLE_GOOGLE_DRIVE", False),
            "powerbiEnabled": _env_bool("TEXTQL_ENABLE_POWERBI", False),
        }

    # ------------------------------------------------------------------ public entry
    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        params = context.metadata.get("params", {}) if isinstance(context.metadata, dict) else {}
        options = params.get("options", {}) if isinstance(params, dict) else {}
        strict_mode = bool(options.get("strict_io"))
        force_refresh = bool(options.get("force_refresh"))

        mission_prompt = params.get("prompt", prompt)
        plan_prompt = self._build_textql_prompt(mission_prompt, context)
        warnings: List[str] = []
        raw_response: Optional[Any] = None

        plan_payload: Dict[str, Any]
        from_cache = False
        source = "live"
        if self.api_key:
            try:
                raw_response = self._call_textql(plan_prompt)
                plan_payload = self._extract_plan(raw_response)
                plan_payload = self._normalize_plan_payload(plan_payload, mission_prompt, context)
                self._persist_live_payload(
                    plan_payload,
                    raw_response,
                    context.asset,
                    user_prompt=mission_prompt,
                )
            except ToolExecutionError as exc:
                warnings.append(str(exc))
                plan_payload, from_cache = self._cached_or_fallback(mission_prompt, context, "api_error", force_refresh)
                source = "cache" if from_cache else "fallback"
            except Exception as exc:  # pragma: no cover - defensive parsing guard
                warnings.append(f"TextQL unexpected error: {exc}")
                plan_payload, from_cache = self._cached_or_fallback(mission_prompt, context, "api_error", force_refresh)
                source = "cache" if from_cache else "fallback"
        else:
            warnings.append("TEXTQL_API_KEY missing; generated offline search plan.")
            plan_payload, from_cache = self._cached_or_fallback(mission_prompt, context, "no_api_key", force_refresh)
            source = "cache" if from_cache else "fallback"

        if not from_cache and "mission_summary" not in plan_payload:
            plan_payload = self._normalize_plan_payload(plan_payload, mission_prompt, context)

        if not from_cache and raw_response is not None:
            request_payload = getattr(self, "_last_request_payload", {"question": plan_prompt})
            self._log_interaction(context, request_payload, plan_payload)

        payload: Dict[str, Any] = {
            "mission_summary": plan_payload.get("mission_summary"),
            "macro_backdrop": plan_payload.get("macro_backdrop"),
            "search_hypotheses": plan_payload.get("search_hypotheses"),
            "dataset_requests": plan_payload.get("dataset_requests"),
            "web_sources": plan_payload.get("web_sources"),
            "validation_steps": plan_payload.get("validation_steps"),
            "warnings": warnings or None,
            "raw_textql_response": raw_response,
            "source": source,
            "from_cache": bool(from_cache),
        }
        if self.include_prompt_in_payload:
            payload["textql_prompt"] = plan_prompt
        if payload.get("warnings") is None:
            payload.pop("warnings")

        summary = self._summarize(context.asset, payload, warnings)
        weight = float(context.weights.get(self.name, 0.0))
        return ToolResult(name=self.name, weight=weight, summary=summary, payload=payload)

    def _cached_or_fallback(
        self,
        prompt: str,
        context: ToolContext,
        reason: str,
        force_refresh: bool,
    ) -> tuple[Dict[str, Any], bool]:
        cached = self._load_cached(context.asset, prompt, force_refresh=force_refresh)
        if cached:
            return cached, True
        return self._fallback_plan(prompt, context, reason), False

    def _persist_live_payload(
        self,
        payload: Mapping[str, Any],
        raw_response: Any,
        asset: Optional[str],
        *,
        user_prompt: str,
    ) -> None:
        try:
            struct = dict(payload)
            struct.pop("raw_textql_response", None)
            struct["_cache_meta"] = {
                "asset": (asset or "").upper(),
                "prompt_hash": self._hash_prompt(user_prompt),
                "saved_at": time.time(),
            }
            # write to a per-run file under runs/textql_logs/<asset>/<ts>.json
            ts = datetime.utcnow().isoformat().replace(":", "-")
            cache_root = Path(os.getenv("TEXTQL_CACHE_DIR", "runs")) / "textql_logs" / (asset or "global")
            cache_root.mkdir(parents=True, exist_ok=True)
            (cache_root / f"{ts}.json").write_text(json.dumps(struct, indent=2), encoding="utf-8")
            # also keep the legacy cache_path for backward compatibility without raw response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(struct, indent=2), encoding="utf-8")
        except Exception:
            pass  # best effort caching

    def _load_cached(self, asset: Optional[str], prompt: str, *, force_refresh: bool) -> Optional[Dict[str, Any]]:
        if force_refresh:
            return None
        candidates: list[Path] = []
        if self.cache_path.exists():
            candidates.append(self.cache_path)
        legacy_dir = Path(os.getenv("TEXTQL_CACHE_DIR", "runs")) / "textql_logs" / (asset or "global")
        if legacy_dir.exists():
            candidates.extend(sorted(legacy_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, Mapping) and self._cache_matches_asset(data, asset, prompt):
                meta = data.get("_cache_meta", {})
                saved_at = meta.get("saved_at")
                if isinstance(saved_at, (int, float)) and self.cache_ttl_seconds:
                    age = time.time() - float(saved_at)
                    if age > self.cache_ttl_seconds:
                        continue
                cleaned = dict(data)
                cleaned.pop("_cache_meta", None)
                return cleaned
        return None

    def _cache_matches_asset(self, data: Mapping[str, Any], asset: Optional[str], prompt: str) -> bool:
        meta = data.get("_cache_meta")
        if not isinstance(meta, Mapping):
            return False
        cached_asset = (meta.get("asset") or "").upper()
        requested = (asset or "").upper()
        if not cached_asset:
            return False
        if requested and cached_asset != requested:
            return False
        cached_prompt_hash = meta.get("prompt_hash")
        if cached_prompt_hash and cached_prompt_hash != self._hash_prompt(prompt):
            return False
        return True

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ textql helpers
    def _call_textql(self, request_prompt: str) -> Mapping[str, Any]:
        if self.api_mode == "REST":
            tools_overrides = dict(self.universal_flags)
            self._last_request_payload = {
                "question": request_prompt,
                "tools": tools_overrides,
                "connectorIds": self.connector_ids,
            }
            try:
                data = create_chat(
                    question=request_prompt,
                    chat_id=None,
                    connector_ids=self.connector_ids or None,
                    tools_overrides=tools_overrides,
                    timeout=self.timeout_seconds,
                api_key=self.api_key or os.getenv("TEXTQL_API_KEY"),
                base_url=os.getenv("TEXTQL_BASE_URL"),
            )
            except TextQLClientError as exc:
                raise ToolExecutionError(f"TextQL request failed: {exc}") from exc

            response_text = data.get("response", "")
            if not isinstance(response_text, str) or not response_text.strip():
                raise ToolExecutionError("TextQL response missing `response` text.")
            return {"answer": response_text, "chatId": data.get("chatId"), "raw": data}

        return self._call_textql_rpc(request_prompt)

    def _call_textql_rpc(self, request_prompt: str) -> Mapping[str, Any]:
        import requests  # local import to avoid hard dependency at module import time

        if not self.rpc_url:
            raise ToolExecutionError("TEXTQL_RPC_URL is not configured.")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        body = self._build_payload(request_prompt)
        self._last_request_payload = dict(body)

        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(self.rpc_url, json=body, headers=headers, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code >= 400:
                    last_error = ToolExecutionError(
                        f"TextQL RPC returned {response.status_code}: {response.text[:200]}"
                    )
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        last_error = exc
                    else:
                        resolved = self._resolve_chat_answer(payload)
                        if resolved is not None:
                            return resolved
                        return payload

            if attempt < attempts and self.retry_delay_seconds > 0:
                time.sleep(self.retry_delay_seconds * attempt)

        raise ToolExecutionError(f"TextQL RPC request failed after {attempts} attempt(s): {last_error}")

    def _resolve_chat_answer(self, payload: Any) -> Optional[Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            return None
        if payload.get("answer"):
            return payload
        chat_id = payload.get("chatId")
        if chat_id and self.answer_url:
            return self._poll_chat_answer(str(chat_id))
        return None

    def _poll_chat_answer(self, chat_id: str) -> Mapping[str, Any]:
        import requests  # local import to avoid hard dependency at module import time

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        payload = {"chatId": chat_id}
        last_error: Optional[Exception] = None
        attempt = 0
        start_time = time.time()
        deadline = (
            start_time + self.poll_max_duration_seconds if self.poll_max_duration_seconds > 0 else None
        )
        interval = self.poll_interval_seconds

        while True:
            attempt += 1
            try:
                response = requests.post(
                    self.answer_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                raw_text = response.text or ""
                data: Optional[Mapping[str, Any]] = None
                if raw_text:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        data = None
                        if response.status_code < 400:
                            last_error = exc
                if response.status_code >= 400:
                    response_code = ""
                    if data and isinstance(data, Mapping):
                        response_code = str(data.get("code", "")).lower()
                    if response_code == "not_found":
                        last_error = ToolExecutionError("TextQL chat answer not ready.")
                    else:
                        last_error = ToolExecutionError(
                            f"TextQL GetAPIChatAnswer returned {response.status_code}: {raw_text[:200]}"
                        )
                elif not data or not isinstance(data, Mapping):
                    last_error = ToolExecutionError("TextQL chat answer missing body.")
                else:
                    answer = data.get("answer")
                    status = data.get("status")
                    if answer:
                        data.setdefault("chatId", chat_id)
                        data.setdefault("pollAttempts", attempt)
                        data.setdefault("latencySeconds", time.time() - start_time)
                        return data
                    if status and str(status).lower() == "failed":
                        raise ToolExecutionError("TextQL chat reported failure.")
                    last_error = ToolExecutionError("TextQL chat answer not ready.")

            more_attempts_available = attempt < self.poll_max_attempts
            more_time_available = deadline is not None and time.time() < deadline
            if not more_attempts_available and not more_time_available:
                break

            sleep_time = interval
            if deadline is not None:
                time_left = deadline - time.time()
                if time_left <= 0:
                    break
                sleep_time = min(sleep_time, max(0.0, time_left))
            if sleep_time > 0:
                time.sleep(sleep_time)
            interval = min(interval * self.poll_backoff_multiplier, self.timeout_seconds)

        waited = time.time() - start_time
        if last_error is None:
            last_error = ToolExecutionError("TextQL chat answer not ready.")
        raise ToolExecutionError(
            f"TextQL chat answer not ready after {attempt} poll(s) / {waited:.1f}s: {last_error}"
        )


    def _build_payload(self, question: str) -> Dict[str, Any]:
        paradigm: Dict[str, Any] = {
            "type": self.paradigm_type,
            "version": self.paradigm_version,
        }
        options = self._paradigm_options()
        if options:
            paradigm["options"] = options

        payload: Dict[str, Any] = {
            "question": question,
            "paradigm": paradigm,
            "model": self.model,
        }
        if self.research_mode:
            payload["research"] = True
        return payload

    def _paradigm_options(self) -> Dict[str, Any]:
        if self.paradigm_type == "TYPE_UNIVERSAL":
            return {"universal": dict(self.universal_flags)}
        if self.paradigm_type == "TYPE_SQL":
            sql_options: Dict[str, Any] = {}
            if self.sql_connector_id is not None:
                sql_options["connectorId"] = self.sql_connector_id
            return {"sql": sql_options} if sql_options else {}
        return {}

    def _extract_plan(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, Mapping):
            answer = response.get("answer")
            if isinstance(answer, str):
                return self._coerce_plan(answer)
            for key in ("plan", "result", "data"):
                if key in response:
                    return self._coerce_plan(response[key])
            if "messages" in response and isinstance(response["messages"], Sequence):
                text_blocks = [msg.get("content") for msg in response["messages"] if isinstance(msg, Mapping)]
                text = "\n\n".join([str(block) for block in text_blocks if block])
                return self._coerce_plan(text)
        if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
            return self._coerce_plan(response[-1])
        return self._coerce_plan(response)

    def _coerce_plan(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            if "```" in text:
                text = text.split("```")[-2] if len(text.split("```")) >= 2 else text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if "{" in text and "}" in text:
                    snippet = text[text.index("{") : text.rindex("}") + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        pass
            return {"mission_summary": text}
        return {"mission_summary": str(value)}

    # ------------------------------------------------------------------ prompt + fallback builders
    def _build_textql_prompt(self, user_prompt: str, context: ToolContext) -> str:
        inputs = context.metadata.get("inputs", {}) if isinstance(context.metadata, dict) else {}
        parse = context.metadata.get("params", {}).get("parse", {}) if isinstance(context.metadata, dict) else {}
        timeframe = inputs.get("timeframe_minutes")
        assets = [asset for asset in context.assets if asset]
        goals = _as_list(parse.get("goals"))
        indicators = _as_list(parse.get("indicators"))
        timeframe_text = f"{timeframe} minutes" if timeframe else "unspecified"

        priority_goals = goals or indicators
        template = textwrap.dedent(
            """
            You are the Pipeline Research Primer. Before any downstream tool runs, gather real, current data for the thesis below using TextQL capabilities (Exa web search, Mobula metrics, Enso datasets, Python runners). Produce facts that are ready to plug into execution logic and rewrite the downstream prompt with those facts.

            Examples (keep structure and numeric specificity):
              - Low-vol swing: vol <1.5% hourly, volume >$350M, price change 2-6%, hold 240m, exit vol >3% or volume < $250M.
              - Breakout confirm: price above $140, buy/sell ratio >1.1, TVL 7d change >5%, volume >$500M, stop if vol >4%.
              - Mean reversion short: price +8% 24h, vol >5%, volume surge 2x baseline, fade back to VWAP; exit if buy/sell >1.0 or vol <2%.
              - Whale squeeze setup: funding rate < -0.01%, price flat (+/-1%), whale net inflow >$20M 24h, long bias for short squeeze; exit if funding turns positive or inflows reverse.
              - Stablecoin farmer: pick pool APY > 8% with TVL >= $50M, stable asset pairs only, rebalance weekly; drop pools if TVL < $40M or APY < 5%.
              - TVL/price divergence: TVL 7d change > 15% while token price < 2% 7d; buy for catch-up; exit if price rallies >10% or TVL momentum fades (<5% 7d).

            Strategy context:
              - Raw thesis: "{user_prompt}"
              - Primary asset: {primary_asset}
              - Symbols of interest: {symbols}
              - Time horizon: {timeframe_text}
              - Prominent indicators/goals: {goal_text}

            Your response MUST be valid JSON matching this schema (no markdown, no comments). Avoid embedding fragile point-in-time prices or volumes; instead describe ranges, thresholds, and how the pipeline should monitor them, citing which datasets/endpoints provide the evidence:
            {{
              "prompt_refinement": {{
                "background": "brief narrative summarizing the thesis with cited catalysts",
                "current_state": "data-backed snapshot with metrics + timestamps from TextQL calls",
                "methodology": "explain how the current state was derived and which signals/APIs proved most reliable",
                "prompt_text": "rewritten downstream prompt (<=120 words) referencing the validated datasets, triggers, and numeric thresholds"
              }},
              "mission_summary": "short sentence describing the actionable focus",
              "macro_backdrop": {{
                "regime": "string",
                "risk_drivers": ["string"],
                "funding_or_liquidity": ["string"],
                "key_metrics": [{{"label": "string", "value": "string or number", "source": "string"}}]
              }},
              "search_hypotheses": [
                {{
                  "name": "short handle",
                  "description": "what is being validated",
                  "onchain_hooks": ["mobula or enso endpoints already executed"],
                  "facts": [
                    {{
                      "label": "what was observed",
                      "value": "numeric/string value with units",
                      "source": "dataset or article",
                      "as_of": "ISO timestamp",
                      "confidence": "qualitative"
                    }}
                  ],
                  "actions": ["next concrete steps for the pipeline"],
                  "validation": ["conditions that confirm or reject this hypothesis"]
                }}
              ],
              "pipeline_layout": [
                {{
                  "phase": "name of phase (e.g., data_gather)",
                  "objective": "what this phase accomplishes",
                  "inputs": ["datasets or hooks used in this phase"],
                  "logic": ["rules/thresholds checked"],
                  "outputs": ["signals produced or actions triggered"]
                }}
              ],
              "dataset_requests": {{
                "mobula": ["endpoint already hit or to refresh"],
                "enso": ["filter or screener used"],
                "other": ["additional APIs (e.g., fear_greed_index)"]
              }},
              "web_sources": [
                {{"url": "https://example.com", "summary": "one-sentence insight", "as_of": "ISO timestamp"}}
              ],
              "validation_steps": ["ordered checklist the pipeline should follow next"]
            }}

            Requirements:
              - Use at most {max_hypotheses} hypotheses.
              - Ground prompt_refinement.* in live data so the rewritten prompt is better than the original.
              - Focus on durable trigger logic (ranges, trend directions, monitoring cadence) and spell out which datasets/endpoints the pipeline should call; do not hardcode today's price/volume unless it illustrates a threshold example.
              - Describe at least three pipeline_layout entries so downstream automation knows which phase calls which API and why.
              - Each fact should cite the dataset/API used plus a timestamp or update cadence.
              - Prefer concise sentences; total response under 400 words.
              - Do not return instructions to run searches -- return the search RESULTS.
            """
        ).strip()
        return template.format(
            user_prompt=user_prompt,
            primary_asset=context.asset or (assets[0] if assets else "unspecified"),
            symbols=", ".join(assets) if assets else "n/a",
            timeframe_text=timeframe_text,
            goal_text=", ".join(priority_goals) if priority_goals else "n/a",
            max_hypotheses=self.max_hypotheses,
        )

    def _fallback_plan(self, user_prompt: str, context: ToolContext, reason: str) -> Dict[str, Any]:
        assets = [asset for asset in context.assets if asset] or ["broad-market"]
        asset = context.asset or assets[0]
        hypotheses: List[Dict[str, Any]] = []
        skeletons = [
            (
                "Liquidity Divergence",
                "Check whether inflows across Mobula/Enso flows confirm the thesis.",
                [
                    f"Search Enso token flows for {asset} and peers over the last 7d.",
                    f"Pull Mobula TVL / liquidity and funding metrics for {asset}.",
                    f"Run Exa web search for institutional commentary on {asset} positioning.",
                ],
            ),
            (
                "Narrative Heat",
                "Gauge how strongly narratives around the asset are resonating across media and dev updates.",
                [
                    f"Use Exa search for '{asset} roadmap catalyst funding' filtered to past week.",
                    f"Scan Mobula news endpoints for sentiment around {asset}.",
                    "Look for Enso strategy vault deployments overlapping with the thesis goals.",
                ],
            ),
            (
                "On-chain Stress Test",
                "Confirm that volatility, gas usage, and liquidity depth support the proposed execution approach.",
                [
                    f"Mobula realized volatility percentile for {asset}.",
                    f"Exa search: \"{asset} volatility regime orderbook depth\" past 14d.",
                    "Mobula or Enso data on whale flows / large swaps.",
                ],
            ),
        ]
        for name, description, queries in skeletons[: self.max_hypotheses]:
            hypotheses.append(
                {
                    "name": name,
                    "description": description,
                    "onchain_hooks": [
                        "/1/market/data",
                        "tokens?chainId=1&type=defi",
                    ],
                    "facts": [
                        {
                            "label": "Dataset guidance",
                            "value": f"Use {query} to capture multi-day trends rather than intraday prints.",
                            "source": "offline_template",
                            "as_of": "",
                            "confidence": "unknown",
                        }
                        for query in queries
                    ],
                    "validation": [
                        "Confirm Mobula metrics and Exa sources align within 24h window.",
                        "Highlight any conflicting signals across datasets.",
                    ],
                    "followups": [
                        "Prioritize downstream tools that can quantify the strongest hypothesis.",
                    ],
                }
            )

        return {
            "mission_summary": self._default_summary(user_prompt, context),
            "macro_backdrop": {
                "regime": "Awaiting live TextQL data",
                "risk_drivers": ["Generated via offline template"],
                "funding_or_liquidity": ["Use Mobula + Enso once available"],
            },
            "search_hypotheses": hypotheses,
            "dataset_requests": {
                "mobula": ["prices.historical", "metrics.volatility_percentile"],
                "enso": ["tokens.filter", "vaults.performance"],
                "other": ["Exa web search"],
            },
            "web_sources": [],
            "pipeline_layout": [
                {
                    "phase": "data_gather",
                    "objective": "Collect news, TVL, and TextQL context before trading logic.",
                    "inputs": ["asknews.latest", "mobula.tvl", "textql.primer"],
                    "logic": ["score news impact", "measure TVL momentum", "refine prompt"],
                    "outputs": ["news_impact_score", "tvl_trend", "refined_prompt"],
                },
                {
                    "phase": "feature_engineering",
                    "objective": "Convert raw metrics into actionable indicators.",
                    "inputs": ["mobula.volatility", "textql.runtime"],
                    "logic": ["compute volatility percentile", "update live hypotheses"],
                    "outputs": ["volatility_signal", "runtime_context"],
                },
                {
                    "phase": "execution",
                    "objective": "Decide whether to stage trades or stay neutral.",
                    "inputs": ["volatility_signal", "news_impact_score", "refined_prompt"],
                    "logic": ["compare signals to thresholds", "respect refined entry/exit rules"],
                    "outputs": ["paper_trade_orders", "strategy_notes"],
                },
            ],
            "validation_steps": [
                "Upgrade to live TextQL API for richer search context.",
                f"Focus early research on {asset} liquidity, flows, and active catalysts.",
            ],
            "fallback_reason": reason,
            "prompt_refinement": {
                "background": f"Original thesis: {user_prompt}",
                "current_state": "Awaiting live TextQL data",
                "methodology": "Offline template response; upgrade to live TextQL for factual rewrite.",
                "data_sources": ["offline_template"],
                "prompt_text": self._default_refined_prompt(user_prompt, context),
            },
            "refined_prompt": self._default_refined_prompt(user_prompt, context),
        }

    def _normalize_plan_payload(
        self,
        plan_payload: Mapping[str, Any],
        user_prompt: str,
        context: ToolContext,
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = dict(plan_payload or {})
        if not isinstance(normalized.get("mission_summary"), str) or not normalized["mission_summary"].strip():
            normalized["mission_summary"] = self._default_summary(user_prompt, context)

        macro = normalized.get("macro_backdrop")
        if not isinstance(macro, Mapping):
            macro = {}
        normalized["macro_backdrop"] = {
            "regime": str(macro.get("regime", "")).strip() or "unspecified",
            "risk_drivers": [str(item) for item in macro.get("risk_drivers", []) or []],
            "funding_or_liquidity": [str(item) for item in macro.get("funding_or_liquidity", []) or []],
            "key_metrics": [
                item
                for item in macro.get("key_metrics", [])
                if isinstance(item, Mapping)
            ],
        }

        hypotheses = normalized.get("search_hypotheses")
        if not isinstance(hypotheses, Sequence):
            hypotheses = []
        trimmed: List[Dict[str, Any]] = []
        for raw in hypotheses:
            if not isinstance(raw, Mapping):
                continue
            entry = dict(raw)
            entry.setdefault("name", "untitled")
            entry.setdefault("description", "")
            if not isinstance(entry.get("onchain_hooks"), Sequence):
                entry["onchain_hooks"] = []
            facts = entry.get("facts")
            if not isinstance(facts, Sequence):
                facts = []
            entry["facts"] = [
                fact for fact in facts if isinstance(fact, Mapping)
            ]
            if not isinstance(entry.get("actions"), Sequence):
                entry["actions"] = []
            if not isinstance(entry.get("validation"), Sequence):
                entry["validation"] = []
            trimmed.append(entry)
            if len(trimmed) >= self.max_hypotheses:
                break
        normalized["search_hypotheses"] = trimmed

        if not normalized["search_hypotheses"]:
            overlay = self._derive_runtime_overlay(normalized, context, user_prompt)
            if overlay:
                normalized["search_hypotheses"] = overlay.get("hypotheses", [])
                macro = normalized.get("macro_backdrop") or {}
                current_risk = list(macro.get("risk_drivers") or [])
                for fact in overlay.get("risk_drivers", []):
                    if fact and fact not in current_risk:
                        current_risk.append(fact)
                macro["risk_drivers"] = current_risk
                normalized["macro_backdrop"] = macro
                existing_steps = list(normalized.get("validation_steps") or [])
                for step in overlay.get("validation_steps", []):
                    if step and step not in existing_steps:
                        existing_steps.append(step)
                if existing_steps:
                    normalized["validation_steps"] = existing_steps

        refinement = self._compose_prompt_refinement(normalized, user_prompt, context)
        normalized["prompt_refinement"] = refinement
        normalized["refined_prompt"] = refinement.get("prompt_text", "")

        layout = normalized.get("pipeline_layout")
        if not isinstance(layout, Sequence):
            layout = []
        normalized["pipeline_layout"] = [
            {
                "phase": str(item.get("phase", "")).strip() or "unspecified",
                "objective": str(item.get("objective", "")).strip(),
                "inputs": [str(entry).strip() for entry in (item.get("inputs") or []) if str(entry).strip()],
                "logic": [str(entry).strip() for entry in (item.get("logic") or []) if str(entry).strip()],
                "outputs": [str(entry).strip() for entry in (item.get("outputs") or []) if str(entry).strip()],
            }
            for item in layout
            if isinstance(item, Mapping)
        ]

        return normalized

    def _derive_runtime_overlay(
        self,
        normalized: Mapping[str, Any],
        context: ToolContext,
        user_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        raw = normalized.get("raw_textql_response")
        if not isinstance(raw, Mapping):
            return None
        parsed = _parse_textql_answer_blob(raw.get("answer"))
        if not parsed:
            return None

        raw_timestamp = str(
            raw.get("timestamp")
            or raw.get("createdAt")
            or raw.get("updatedAt")
            or raw.get("generationTimestamp")
            or ""
        )

        facts = parsed.get("facts") or parsed.get("insights")
        if not isinstance(facts, Sequence):
            facts = []
        fact_entries: List[Dict[str, Any]] = []
        risk_lines: List[str] = []
        for index, fact in enumerate(facts):
            if isinstance(fact, Mapping):
                label = str(fact.get("label") or fact.get("name") or f"fact_{index + 1}")
                value = fact.get("value") or fact.get("text") or fact.get("detail") or ""
                source = fact.get("source") or "textql_runtime"
                as_of = fact.get("as_of") or fact.get("timestamp") or raw_timestamp
                confidence = str(fact.get("confidence") or "")
                fact_entries.append(
                    {
                        "label": label,
                        "value": value,
                        "source": source,
                        "as_of": as_of,
                        "confidence": confidence,
                    }
                )
                risk_line = f"{label}: {value}".strip()
                if risk_line:
                    risk_lines.append(risk_line)
            else:
                value = str(fact).strip()
                if not value:
                    continue
                label = f"fact_{index + 1}"
                fact_entries.append(
                    {
                        "label": label,
                        "value": value,
                        "source": "textql_runtime",
                        "as_of": raw_timestamp,
                        "confidence": "",
                    }
                )
                risk_lines.append(value)

        actions_src = parsed.get("actions") or parsed.get("next_steps") or parsed.get("recommendations")
        if isinstance(actions_src, Sequence) and not isinstance(actions_src, (str, bytes)):
            actions_iter = actions_src
        elif actions_src:
            actions_iter = [actions_src]
        else:
            actions_iter = []
        actions = [str(item).strip() for item in actions_iter if str(item).strip()]

        validation_src = parsed.get("validation") or parsed.get("checks") or parsed.get("conditions")
        if isinstance(validation_src, Sequence) and not isinstance(validation_src, (str, bytes)):
            validation_iter = validation_src
        elif validation_src:
            validation_iter = [validation_src]
        else:
            validation_iter = []
        validation = [str(item).strip() for item in validation_iter if str(item).strip()]

        hypothesis_name = str(parsed.get("name") or parsed.get("title") or "runtime_signal")
        description = str(
            parsed.get("summary")
            or parsed.get("mission_summary")
            or parsed.get("context")
            or normalized.get("mission_summary")
            or self._default_summary(user_prompt, context)
        )
        onchain_hooks_src = parsed.get("onchain_hooks")
        if isinstance(onchain_hooks_src, Sequence) and not isinstance(onchain_hooks_src, (str, bytes)):
            onchain_hooks = [str(item).strip() for item in onchain_hooks_src if str(item).strip()]
        elif isinstance(onchain_hooks_src, str):
            onchain_hooks = [onchain_hooks_src.strip()] if onchain_hooks_src.strip() else []
        else:
            onchain_hooks = []

        if not (fact_entries or actions or validation):
            return None

        hypothesis = {
            "name": hypothesis_name,
            "description": description,
            "onchain_hooks": onchain_hooks,
            "facts": fact_entries,
            "actions": actions,
            "validation": validation,
        }

        return {
            "hypotheses": [hypothesis],
            "risk_drivers": risk_lines or actions,
            "validation_steps": validation,
        }

    def _default_summary(self, user_prompt: str, context: ToolContext) -> str:
        asset = context.asset or (context.assets[0] if context.assets else "target asset")
        return f"Frame actionable search context for {asset} given: {user_prompt[:160]}"

    def _default_refined_prompt(self, user_prompt: str, context: ToolContext) -> str:
        asset = context.asset or (context.assets[0] if context.assets else "target asset")
        base = user_prompt.strip() or "Monitor high-impact catalysts and liquidity checks."
        return f"{asset}: {base}"

    def _compose_prompt_refinement(
        self,
        normalized: Mapping[str, Any],
        user_prompt: str,
        context: ToolContext,
    ) -> Dict[str, str]:
        asset = context.asset or (context.assets[0] if context.assets else "target asset")
        raw_refinement = normalized.get("prompt_refinement")
        if not isinstance(raw_refinement, Mapping):
            raw_refinement = {}

        mission = str(normalized.get("mission_summary") or "").strip()
        background = str(raw_refinement.get("background") or mission or user_prompt).strip()
        if not background:
            background = f"Original thesis: {user_prompt}"

        macro = normalized.get("macro_backdrop") or {}
        key_metrics = macro.get("key_metrics") or []
        metric_bits: List[str] = []
        if isinstance(key_metrics, Sequence):
            for metric in key_metrics:
                if not isinstance(metric, Mapping):
                    continue
                label = str(metric.get("label") or "").strip()
                value = metric.get("value")
                value_text = str(value) if value not in (None, "") else ""
                parts = [part for part in [label, value_text] if part]
                if not parts:
                    continue
                metric_str = " ".join(parts)
                source = str(metric.get("source") or "").strip()
                if source:
                    metric_str = f"{metric_str} ({source})"
                metric_bits.append(metric_str)

        fact_bits: List[str] = []
        fact_sources: List[str] = []
        for hypothesis in normalized.get("search_hypotheses") or []:
            if not isinstance(hypothesis, Mapping):
                continue
            for fact in hypothesis.get("facts") or []:
                if not isinstance(fact, Mapping):
                    continue
                label = str(fact.get("label") or "").strip()
                value = str(fact.get("value") or "").strip()
                snippet = " ".join([item for item in [label, value] if item]).strip()
                if snippet:
                    fact_bits.append(snippet)
                source = str(fact.get("source") or "").strip()
                if source:
                    fact_sources.append(source)

        raw_current_state = str(raw_refinement.get("current_state") or "").strip()
        current_state = (
            raw_current_state
            or "; ".join(metric_bits[:3])
            or "; ".join(fact_bits[:3])
            or "Awaiting live TextQL data."
        )

        dataset_requests = normalized.get("dataset_requests") or {}
        dataset_bits: List[str] = []
        if isinstance(dataset_requests, Mapping):
            for name, entries in dataset_requests.items():
                if not entries:
                    continue
                for entry in entries:
                    dataset_bits.append(f"{name}:{entry}")

        unique_sources: List[str] = []
        for source in fact_sources + dataset_bits:
            source = str(source).strip()
            if source and source not in unique_sources:
                unique_sources.append(source)

        raw_methodology = str(raw_refinement.get("methodology") or "").strip()
        if raw_methodology:
            methodology = raw_methodology
        elif unique_sources:
            methodology = f"Derived from {', '.join(unique_sources[:5])}."
        else:
            methodology = "Used cached priors; awaiting confirmed datasets."

        raw_prompt_text = str(raw_refinement.get("prompt_text") or "").strip()
        prompt_text = raw_prompt_text
        source_clause = ", ".join(unique_sources[:4]) if unique_sources else "TextQL feeds"
        if not prompt_text:
            condition_text = "; ".join((fact_bits or metric_bits)[:2])
            mission_clause = mission or "update positioning rules"
            if condition_text:
                prompt_text = (
                    f"{asset}: {mission_clause}. Act when {condition_text}. "
                    f"Refresh inputs via {source_clause}."
                )
            else:
                prompt_text = self._default_refined_prompt(user_prompt, context)
        elif unique_sources:
            prompt_text = f"{prompt_text.rstrip('.')} (Reference {source_clause})."

        words = prompt_text.split()
        if len(words) > 120:
            prompt_text = " ".join(words[:120])

        return {
            "background": background,
            "current_state": current_state,
            "methodology": methodology,
            "data_sources": unique_sources[:8],
            "prompt_text": prompt_text.strip(),
        }

    def _log_interaction(
        self,
        context: ToolContext,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
    ) -> None:
        try:
            timestamp = datetime.utcnow().isoformat()
            log_path = self.log_dir / f"{self.name}_{timestamp.replace(':', '-')}.json"
            record = {
                "tool": self.name,
                "timestamp": timestamp,
                "asset": context.asset,
                "assets": list(context.assets),
                "question": request_payload.get("question"),
                "response": response_payload,
            }
            log_path.write_text(json.dumps(record, indent=2))
        except Exception:
            pass

    def _summarize(self, asset: Optional[str], payload: Mapping[str, Any], warnings: Sequence[str]) -> str:
        hyp_count = len(payload.get("search_hypotheses") or [])
        warning_text = f" ({len(warnings)} warning)" if warnings else ""
        asset_text = asset or "global"
        refined = ""
        refinement = payload.get("prompt_refinement")
        if isinstance(refinement, Mapping):
            refined = str(refinement.get("prompt_text") or "").strip()
        if not refined:
            refined = str(payload.get("refined_prompt") or "").strip()
        snippet = ""
        if refined:
            compact = refined.replace("\n", " ").strip()
            if len(compact) > 140:
                compact = compact[:137] + "..."
            snippet = f" | refined: {compact}"
        return f"{asset_text}: TextQL refined prompt ({hyp_count} tracks){warning_text}{snippet}"


TextQLPrimerTool.__TOOL_META__ = __PRIMER_META__


__RUNTIME_META__ = {
    "name": "textql_runtime",
    "module": "tools.textql_context_tool",
    "object": "TextQLRuntimeTool",
    "version": "1.0",
    "description": "Executes follow-up TextQL research during later pipeline phases.",
    "author": "auto",
    "keywords": [
        "textql",
        "followup",
        "runtime",
        "search",
        "context",
        "analysis",
    ],
    "phases": ["feature_engineering", "signal_generation", "risk_sizing"],
    "outputs": [
        "mission_summary",
        "macro_backdrop",
        "search_hypotheses",
        "dataset_requests",
        "web_sources",
    ],
}


class TextQLRuntimeTool(TextQLPrimerTool):
    """Runtime TextQL queries for mid-pipeline follow-ups."""

    name = "textql_runtime"
    description = "Executes TextQL follow-up research using current phase context."
    keywords = frozenset({"textql", "followup", "runtime", "search", "analysis"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cache_dir = Path(os.getenv("TEXTQL_CACHE_DIR", "runs"))
        self.cache_path = cache_dir / "textql_runtime_cache.json"

    def _build_textql_prompt(self, user_prompt: str, context: ToolContext) -> str:
        inputs = context.metadata.get("inputs", {}) if isinstance(context.metadata, dict) else {}
        parse = context.metadata.get("params", {}).get("parse", {}) if isinstance(context.metadata, dict) else {}
        runtime = context.metadata.get("params", {}).get("runtime", {}) if isinstance(context.metadata, dict) else {}
        phase = runtime.get("current_phase")
        assets = [asset for asset in context.assets if asset]
        goals = _as_list(parse.get("goals"))
        indicators = _as_list(parse.get("indicators"))
        timeframe = inputs.get("timeframe_minutes")
        timeframe_text = f"{timeframe} minutes" if timeframe else "unspecified"

        prompt_block = textwrap.dedent(
            f"""
            You are the Runtime TextQL Analyst. The pipeline is currently in phase "{phase or 'unknown'}". Provide immediate data-backed answers that help execution decisions mid-run.

            Examples (concise, executable):
              - Confirm long: vol <2%, volume >$400M, price +3-6% 24h, buy/sell >1.0; exit if vol >4% or volume < $250M.
              - Block execution: conditions_met=false when vol >5% or volume dries up; recommend HOLD/exit.

            Context:
              - Original thesis: "{user_prompt}"
              - Phase: {phase or 'unknown'}
              - Symbols: {', '.join(assets) if assets else 'n/a'}
              - Time horizon: {timeframe_text}
              - Goals/indicators: {', '.join(goals or indicators) if goals or indicators else 'n/a'}

            Return STRICT JSON with the same schema as the primer (prompt_refinement plus facts/actions/validation). Keep it under 250 words and focus on updates that may change decisions right now.
            """
        ).strip()
        return prompt_block

    def _cached_or_fallback(self, prompt: str, context: ToolContext, reason: str) -> tuple[Dict[str, Any], bool]:
        # Runtime tool prefers fresh data; only fall back to primer cache if runtime cache empty.
        cached = self._load_cached(context.asset)
        if cached:
            return cached, True
        # fallback to primer cache if available
        primer_cache = Path(os.getenv("TEXTQL_CACHE_DIR", "runs")) / "textql_direct_sample.json"
        if primer_cache.exists():
            try:
                data = json.loads(primer_cache.read_text(encoding="utf-8"))
                if isinstance(data, Mapping) and self._cache_matches_asset(data, context.asset):
                    cleaned = dict(data)
                    cleaned.pop("_cache_meta", None)
                    return cleaned, True
            except Exception:
                pass
        return self._fallback_plan(prompt, context, reason), False


TextQLRuntimeTool.__TOOL_META__ = __RUNTIME_META__
