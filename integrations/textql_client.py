from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


TEXTQL_BASE_URL = os.getenv("TEXTQL_BASE_URL", "https://app.textql.com/v1")
TEXTQL_API_KEY = os.getenv("TEXTQL_API_KEY")


class TextQLClientError(RuntimeError):
    """Raised when the TextQL API returns an error or the call fails."""


def _auth_headers() -> Dict[str, str]:
    if not TEXTQL_API_KEY:
        raise TextQLClientError("TEXTQL_API_KEY environment variable is missing.")
    return {"Authorization": f"Bearer {TEXTQL_API_KEY}"}


def create_chat(
    *,
    question: str,
    chat_id: Optional[str] = None,
    connector_ids: Optional[List[int]] = None,
    tools_overrides: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Call POST /v1/chat and return the parsed JSON response."""
    if len(question or "") < 3:
        raise ValueError("TextQL question must be at least 3 characters.")

    url = f"{TEXTQL_BASE_URL.rstrip('/')}/chat"
    payload: Dict[str, Any] = {"question": question}
    if chat_id:
        payload["chatId"] = chat_id

    tools: Dict[str, Any] = {}
    if connector_ids:
        tools["connectorIds"] = connector_ids
    if tools_overrides:
        tools.update(tools_overrides)
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json", **_auth_headers()}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise TextQLClientError(f"Network error calling TextQL: {exc}") from exc

    if response.status_code != 200:
        raise TextQLClientError(
            f"TextQL returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise TextQLClientError("TextQL response was not valid JSON.") from exc

    if data.get("error"):
        raise TextQLClientError(f"TextQL API error: {data['error']}")
    return data


def create_chat_and_extract_answer(
    *,
    question: str,
    chat_id: Optional[str] = None,
    connector_ids: Optional[List[int]] = None,
    tools_overrides: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> str:
    """Call /chat and return only the `response` field."""
    data = create_chat(
        question=question,
        chat_id=chat_id,
        connector_ids=connector_ids,
        tools_overrides=tools_overrides,
        timeout=timeout,
    )
    answer = data.get("response")
    if not isinstance(answer, str):
        raise TextQLClientError("TextQL response missing `response` field.")
    return answer
