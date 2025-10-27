from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


class LLMRouter:
    """Thin wrapper around OpenAI responses for tool routing suggestions."""

    def __init__(self, *, model: str | None = None, max_tools: int = 3) -> None:
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_API_BASE")
        model = model or os.getenv("PIPELINE_LLM_MODEL", "gpt-4o-mini")
        self.max_tools = max_tools

        if key and OpenAI is not None:
            client_kwargs = {"api_key": key}
            if base:
                client_kwargs["base_url"] = base
            self.client = OpenAI(**client_kwargs)
            self.model = model
            self.enabled = True
        else:  # pragma: no cover - disabled path
            self.client = None
            self.model = model
            self.enabled = False

    def recommend(self, prompt: str, tool_descriptions: Dict[str, str]) -> List[Tuple[str, float]]:
        if not self.enabled or not self.client:
            return []

        tools_blob = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())
        user_prompt = f"""You are a router that selects which analysis tools should run for a crypto trading prompt.

Available tools:
{tools_blob}

Instructions:
1. Read the user's prompt.
2. Pick up to {self.max_tools} tools (if none are relevant, return an empty list).
3. Return strict JSON in the format:
[
  {{"tool": "tool_name", "confidence": 0.0_to_1.0, "reason": "brief note"}}
]
Only include the JSON. Confidence should reflect how essential the tool is.

User prompt:
\"\"\"{prompt}\"\"\"
"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that outputs concise JSON without additional text.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
        except Exception:
            return []

        try:
            text = response.output_text  # type: ignore[attr-defined]
        except AttributeError:
            # fallback: concatenate outputs
            chunks = []
            for item in getattr(response, "output", []) or []:
                part = getattr(item, "content", [])
                for block in part:
                    if block.type == "text":
                        chunks.append(block.text)
            text = "\n".join(chunks)

        try:
            data = json.loads(text)
        except Exception:
            return []

        recommendations: List[Tuple[str, float]] = []
        if isinstance(data, list):
            for item in data[: self.max_tools]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("tool") or "").strip()
                if not name:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                recommendations.append((name, confidence))
        return recommendations

