"""LiteLLM-backed client helpers used by the engine and GUI."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlparse, urlunparse

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import completion

from constitutional_ai.config import (
    ModelSettings,
    ProviderCredentials,
    PROVIDER_ENV_VARS,
    build_litellm_model,
    credential_field_for_provider,
)
from constitutional_ai.models import UsageStats

try:
    from litellm import get_valid_models
except ImportError:  # pragma: no cover
    get_valid_models = None  # type: ignore[assignment]


@dataclass(slots=True)
class CompletionResult:
    """Normalized result from a single LiteLLM completion call."""

    content: str
    usage: UsageStats
    raw: dict[str, Any]


@dataclass(slots=True)
class ModelListResult:
    """Model listing result for a provider."""

    models: list[dict[str, Any]]
    supports_listing: bool


class LiteLLMAPIError(RuntimeError):
    """Raised when a LiteLLM request fails."""


_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[float, ModelListResult]] = {}
_MODEL_CACHE_TTL_SECONDS = 60.0
_ENDPOINT_LISTING_PROVIDERS = {"openai", "anthropic", "gemini", "xai", "litellm_proxy", "fireworks_ai"}


def _debug_enabled() -> bool:
    """Return True when debug logging is enabled by environment."""
    return os.getenv("CONSTITUTIONAL_AI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert LiteLLM/OpenAI response objects into a plain dictionary."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _content_to_text(content: Any) -> str:
    """Normalize completion content to a user-facing string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                    continue
        return "".join(chunks).strip()
    return str(content or "").strip()


def _usage_from_response(raw: dict[str, Any]) -> UsageStats:
    """Extract normalized usage information from a LiteLLM response."""
    return UsageStats.from_mapping(raw.get("usage") if isinstance(raw, dict) else None)


def _normalize_runtime_api_base(provider: str, api_base: str) -> str:
    """Adjust provider base URLs to what LiteLLM expects at request time."""
    raw = str(api_base or "").strip()
    if not raw:
        return ""
    if provider != "openai":
        return raw

    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif path != "/v1":
        return raw
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _build_completion_kwargs(
    *,
    endpoint: ModelSettings,
    credentials: ProviderCredentials,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_ms: int,
) -> dict[str, Any]:
    """Build one LiteLLM completion call payload."""
    model_name = build_litellm_model(endpoint.provider, endpoint.model)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": max(1.0, timeout_ms / 1000.0),
    }

    api_key = credentials.get_for_provider(endpoint.provider)
    if api_key:
        kwargs["api_key"] = api_key
    if endpoint.api_base:
        kwargs["api_base"] = _normalize_runtime_api_base(endpoint.provider, endpoint.api_base)
    if endpoint.api_version:
        kwargs["api_version"] = endpoint.api_version
    return kwargs


@contextmanager
def _temporary_provider_environment(endpoint: ModelSettings, credentials: ProviderCredentials) -> Iterator[None]:
    """Temporarily set env vars expected by LiteLLM model-discovery helpers."""
    previous: dict[str, str | None] = {}
    updates: dict[str, str] = {}
    field_name = credential_field_for_provider(endpoint.provider)
    if field_name:
        env_var = PROVIDER_ENV_VARS[field_name]
        credential = credentials.get_for_provider(endpoint.provider)
        if credential:
            updates[env_var] = credential
    if endpoint.provider == "openai" and endpoint.api_base:
        updates["OPENAI_BASE_URL"] = _normalize_runtime_api_base(endpoint.provider, endpoint.api_base)
    if endpoint.provider == "azure":
        if endpoint.api_base:
            updates["AZURE_API_BASE"] = endpoint.api_base
        if endpoint.api_version:
            updates["AZURE_API_VERSION"] = endpoint.api_version

    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in updates.items():
            old = previous.get(key)
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def chat_completion(
    *,
    endpoint: ModelSettings,
    credentials: ProviderCredentials,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_ms: int,
) -> CompletionResult:
    """Call LiteLLM `completion()` and return normalized content plus usage."""
    kwargs = _build_completion_kwargs(
        endpoint=endpoint,
        credentials=credentials,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_ms=timeout_ms,
    )
    if _debug_enabled():
        print(
            f"[constitutional_ai.debug] completion provider={endpoint.provider} model={kwargs['model']}",
            file=sys.stderr,
        )

    try:
        response = completion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise LiteLLMAPIError(str(exc)) from exc

    raw = _to_plain_dict(response)
    try:
        choice = raw["choices"][0]
        message = choice["message"]
        content = _content_to_text(message.get("content"))
    except (KeyError, IndexError, TypeError) as exc:
        raise LiteLLMAPIError("LiteLLM response did not include choices[0].message.content.") from exc

    if not content:
        raise LiteLLMAPIError("LiteLLM response content was empty.")

    usage = _usage_from_response(raw)
    return CompletionResult(content=content, usage=usage, raw=raw)


def list_models(*, endpoint: ModelSettings, credentials: ProviderCredentials, timeout_ms: int) -> ModelListResult:
    """Return available models for a provider when LiteLLM supports endpoint discovery."""
    provider = endpoint.provider
    cache_key = (
        provider,
        credentials.get_for_provider(provider),
        endpoint.api_base,
        endpoint.api_version,
    )
    now = time.time()
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None and now - cached[0] <= _MODEL_CACHE_TTL_SECONDS:
        return cached[1]

    supports_listing = provider in _ENDPOINT_LISTING_PROVIDERS and get_valid_models is not None
    if not supports_listing:
        result = ModelListResult(models=[], supports_listing=False)
        _MODEL_CACHE[cache_key] = (now, result)
        return result

    try:
        with _temporary_provider_environment(endpoint, credentials):
            discovered = get_valid_models(check_provider_endpoint=True, custom_llm_provider=provider)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        raise LiteLLMAPIError(f"Model listing failed for provider '{provider}': {exc}") from exc

    raw_models = discovered if isinstance(discovered, list) else []
    models = [{"id": str(name).strip()} for name in raw_models if str(name).strip()]
    models.sort(key=lambda item: item["id"])
    result = ModelListResult(models=models, supports_listing=True)
    _MODEL_CACHE[cache_key] = (now, result)
    return result
