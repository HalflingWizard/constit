"""Configuration loading and persistence for constitutional-ai."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


DEFAULT_CONFIG_DIR = Path.home() / ".constitutional_ai"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_RULES = [
    "Be helpful, clear, and accurate.",
    "Do not provide illegal wrongdoing instructions or facilitation.",
    "Do not reveal secrets or private data (including API keys).",
    "If uncertain, state uncertainty and ask clarifying questions.",
    "Keep responses concise unless the user asks for depth.",
]

PROVIDER_ENV_VARS = {
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "xai_api_key": "XAI_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "togetherai_api_key": "TOGETHERAI_API_KEY",
    "huggingface_api_key": "HUGGINGFACE_API_KEY",
    "azure_api_key": "AZURE_API_KEY",
}

PROVIDER_CREDENTIAL_FIELDS = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "xai": "xai_api_key",
    "openrouter": "openrouter_api_key",
    "groq": "groq_api_key",
    "togetherai": "togetherai_api_key",
    "huggingface": "huggingface_api_key",
    "azure": "azure_api_key",
}

PROVIDERS_WITH_OPTIONAL_KEYS = {"ollama", "lm_studio"}
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def normalize_api_base(value: str) -> str:
    """Normalize optional API base URLs and reject malformed values."""
    raw = str(value or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid api_base: must start with http:// or https://.")
    if not parsed.netloc:
        raise ValueError("Invalid api_base: missing hostname.")

    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def normalize_provider(value: Any) -> str:
    """Return a normalized LiteLLM provider id."""
    provider = str(value or DEFAULT_PROVIDER).strip().lower().replace("-", "_")
    if provider == "together":
        return "togetherai"
    if provider == "google":
        return "gemini"
    return provider or DEFAULT_PROVIDER


def normalize_model_name(provider: str, value: Any) -> str:
    """Normalize a stored model name while allowing advanced full model ids."""
    provider = normalize_provider(provider)
    model = str(value or "").strip()
    if not model:
        return DEFAULT_OPENAI_MODEL if provider == "openai" else ""
    prefix = f"{provider}/"
    if model.startswith(prefix):
        return model[len(prefix) :]
    return model


def build_litellm_model(provider: str, model: str) -> str:
    """Return the actual LiteLLM model string for a provider + model pair."""
    normalized_provider = normalize_provider(provider)
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("Model is required.")
    if "/" in normalized_model and normalized_model.split("/", 1)[0] == normalized_provider:
        return normalized_model
    return f"{normalized_provider}/{normalized_model}"


def provider_requires_api_key(provider: str) -> bool:
    """Return True when the provider typically requires a credential."""
    return normalize_provider(provider) not in PROVIDERS_WITH_OPTIONAL_KEYS


def credential_field_for_provider(provider: str) -> str | None:
    """Return the credentials field name used by a provider, if any."""
    return PROVIDER_CREDENTIAL_FIELDS.get(normalize_provider(provider))


@dataclass(slots=True)
class ProviderCredentials:
    """Saved provider-specific credentials for LiteLLM."""

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    xai_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    togetherai_api_key: str = ""
    huggingface_api_key: str = ""
    azure_api_key: str = ""

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "ProviderCredentials":
        value = value or {}
        return ProviderCredentials(
            openai_api_key=str(value.get("openai_api_key", "") or ""),
            anthropic_api_key=str(value.get("anthropic_api_key", "") or ""),
            gemini_api_key=str(value.get("gemini_api_key", "") or ""),
            xai_api_key=str(value.get("xai_api_key", "") or ""),
            openrouter_api_key=str(value.get("openrouter_api_key", "") or ""),
            groq_api_key=str(value.get("groq_api_key", "") or ""),
            togetherai_api_key=str(value.get("togetherai_api_key", "") or ""),
            huggingface_api_key=str(value.get("huggingface_api_key", "") or ""),
            azure_api_key=str(value.get("azure_api_key", "") or ""),
        )

    def get_for_provider(self, provider: str) -> str:
        field_name = credential_field_for_provider(provider)
        if not field_name:
            return ""
        return str(getattr(self, field_name, "") or "").strip()


@dataclass(slots=True)
class ModelSettings:
    """One writer or judge endpoint configuration."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_OPENAI_MODEL
    api_base: str = ""
    api_version: str = ""

    @staticmethod
    def from_mapping(value: dict[str, Any] | None, *, default_provider: str = DEFAULT_PROVIDER) -> "ModelSettings":
        value = value or {}
        provider = normalize_provider(value.get("provider", default_provider))
        return ModelSettings(
            provider=provider,
            model=normalize_model_name(provider, value.get("model", DEFAULT_OPENAI_MODEL if provider == "openai" else "")),
            api_base=normalize_api_base(str(value.get("api_base", "") or "")),
            api_version=str(value.get("api_version", "") or "").strip(),
        )

    def litellm_model(self) -> str:
        return build_litellm_model(self.provider, self.model)


def _migrate_legacy_role_model(raw_model: Any, provider: str) -> str:
    """Convert prior OpenAI model strings into the new provider-aware model name."""
    text = str(raw_model or "").strip()
    if not text:
        return DEFAULT_OPENAI_MODEL if normalize_provider(provider) == "openai" else ""
    prefix = f"{normalize_provider(provider)}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _migrate_legacy_role_settings(settings: dict[str, Any], role_name: str) -> ModelSettings:
    """Build a role config from either new nested settings or old flat keys."""
    nested = settings.get(role_name)
    if isinstance(nested, dict):
        return ModelSettings.from_mapping(nested)

    provider = DEFAULT_PROVIDER
    api_base = normalize_api_base(str(settings.get("base_url", "") or ""))
    model_key = f"{role_name}_model"
    model = _migrate_legacy_role_model(settings.get(model_key, DEFAULT_OPENAI_MODEL), provider)
    return ModelSettings(provider=provider, model=model, api_base=api_base, api_version="")


@dataclass(slots=True)
class RuntimeSettings:
    """Runtime knobs used by the constitutional loop and LiteLLM client."""

    credentials: ProviderCredentials = field(default_factory=ProviderCredentials)
    writer: ModelSettings = field(default_factory=ModelSettings)
    judge: ModelSettings = field(default_factory=ModelSettings)
    temperature: float = 0.4
    max_tokens: int = 650
    max_revisions_per_rule: int = 1
    execution_mode: str = "sequential"
    parallel_max_iterations: int = 0
    max_iteration_ms: int = 0
    timeout_ms: int = 45_000

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "RuntimeSettings":
        """Create settings from untrusted mapping input with defaults."""
        value = value or {}
        execution_mode = str(value.get("execution_mode", "sequential") or "sequential").strip().lower()
        if execution_mode not in {"sequential", "parallel"}:
            execution_mode = "sequential"

        legacy_credentials = {}
        legacy_api_key = str(value.get("api_key", "") or "").strip()
        if legacy_api_key:
            legacy_credentials["openai_api_key"] = legacy_api_key

        return RuntimeSettings(
            credentials=ProviderCredentials.from_mapping(
                {**legacy_credentials, **(value.get("credentials") if isinstance(value.get("credentials"), dict) else {})}
            ),
            writer=_migrate_legacy_role_settings(value, "writer"),
            judge=_migrate_legacy_role_settings(value, "judge"),
            temperature=float(value.get("temperature", 0.4) or 0.4),
            max_tokens=int(value.get("max_tokens", 650) or 650),
            max_revisions_per_rule=int(value.get("max_revisions_per_rule", 1) or 1),
            execution_mode=execution_mode,
            parallel_max_iterations=max(0, int(value.get("parallel_max_iterations", 0) or 0)),
            max_iteration_ms=max(0, int(value.get("max_iteration_ms", 0) or 0)),
            timeout_ms=int(value.get("timeout_ms", 45_000) or 45_000),
        )


@dataclass(slots=True)
class PromptTemplates:
    """Prompt templates used by writer and judge roles."""

    writer_system: str = (
        "You are the writer agent. Revise the existing response with minimal changes. "
        "Preserve the original wording, structure, and tone as much as possible. "
        "Only modify the specific parts needed to address the judge's critique and follow the provided rule. "
        "Do not rewrite or rephrase unaffected sections. "
        "Return ONLY the final user-facing answer, with no meta-commentary."
    )
    judge_pass_system: str = (
        "You are the judge agent. Evaluate the writer agent's answer against the given rule ONLY. "
        "Do not use any other criteria. "
        "Return JSON ONLY (no markdown, no extra text). First decide whether the rule applies to this user prompt "
        "and answer. If it does not apply, mark it as not applicable. If it applies, decide whether the answer "
        "follows the rule. Schema: {\"applies\": boolean, \"pass\": boolean}. "
        "Constraints: if applies is false, pass MUST be true."
    )
    judge_critique_system: str = (
        "You are the judge agent. Evaluate the writer agent's answer against the given rule ONLY. "
        "The answer has already failed this rule. Provide a concise critique and explicit, actionable required fixes. "
        "Base your judgment only on the given rule, not on any other criteria. "
        "The required fixes must clearly identify what part of the answer is problematic and how it must be changed "
        "so the revised answer no longer violates the rule. "
        "Return JSON ONLY (no markdown, no extra text). Schema: {\"critique\": string, \"required_fixes\": string}."
    )

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "PromptTemplates":
        """Create prompts from untrusted mapping input with defaults."""
        value = value or {}
        defaults = PromptTemplates()
        return PromptTemplates(
            writer_system=str(value.get("writer_system", defaults.writer_system) or defaults.writer_system),
            judge_pass_system=str(value.get("judge_pass_system", defaults.judge_pass_system) or defaults.judge_pass_system),
            judge_critique_system=str(
                value.get("judge_critique_system", defaults.judge_critique_system) or defaults.judge_critique_system
            ),
        )


@dataclass(slots=True)
class AppConfig:
    """Top-level persisted configuration shared by CLI, GUI, and notebooks."""

    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    rules: list[str] = field(default_factory=lambda: list(DEFAULT_RULES))
    prompts: PromptTemplates = field(default_factory=PromptTemplates)

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "AppConfig":
        """Create a complete config object from a JSON mapping."""
        value = value or {}
        rules_raw = value.get("rules", list(DEFAULT_RULES))
        rules = [str(line).strip() for line in rules_raw if str(line).strip()] if isinstance(rules_raw, list) else []
        return AppConfig(
            settings=RuntimeSettings.from_mapping(value.get("settings") if isinstance(value, dict) else None),
            rules=rules,
            prompts=PromptTemplates.from_mapping(value.get("prompts") if isinstance(value, dict) else None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a JSON-compatible dictionary."""
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON from disk, returning an empty mapping for missing or invalid files."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_credential_sources(path: str | Path | None = None) -> dict[str, str]:
    """Return where each effective provider credential comes from: env, config, or none."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = _read_json(cfg_path)
    config_credentials = {}
    if isinstance(payload.get("settings"), dict) and isinstance(payload["settings"].get("credentials"), dict):
        config_credentials = payload["settings"]["credentials"]
    elif isinstance(payload.get("settings"), dict) and str(payload["settings"].get("api_key", "") or "").strip():
        config_credentials = {"openai_api_key": str(payload["settings"].get("api_key", "") or "").strip()}

    sources: dict[str, str] = {}
    for field_name, env_var in PROVIDER_ENV_VARS.items():
        env_value = os.getenv(env_var, "").strip()
        if env_value:
            sources[field_name] = "environment"
        elif str(config_credentials.get(field_name, "") or "").strip():
            sources[field_name] = "config"
        else:
            sources[field_name] = "none"
    return sources


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from disk and overlay environment-sourced provider credentials."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    base = AppConfig.from_mapping(_read_json(cfg_path))
    for field_name, env_var in PROVIDER_ENV_VARS.items():
        env_value = os.getenv(env_var, "").strip()
        if env_value:
            setattr(base.settings.credentials, field_name, env_value)
    return base


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    """Persist config to disk and return the final path used."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return cfg_path


def merge_config(base: AppConfig, payload: dict[str, Any] | None) -> AppConfig:
    """Merge untrusted override payload into an existing config object."""
    payload = payload or {}
    merged = AppConfig.from_mapping(base.to_dict())

    if isinstance(payload.get("settings"), dict):
        current = merged.settings
        incoming = payload["settings"]
        current_dict = asdict(current)
        merged_settings = {**current_dict, **incoming}

        if isinstance(incoming.get("credentials"), dict):
            merged_settings["credentials"] = {**current_dict["credentials"], **incoming["credentials"]}
        if isinstance(incoming.get("writer"), dict):
            merged_settings["writer"] = {**current_dict["writer"], **incoming["writer"]}
        if isinstance(incoming.get("judge"), dict):
            merged_settings["judge"] = {**current_dict["judge"], **incoming["judge"]}

        merged.settings = RuntimeSettings.from_mapping(merged_settings)

    if isinstance(payload.get("rules"), list):
        merged.rules = [str(line).strip() for line in payload["rules"] if str(line).strip()]

    if isinstance(payload.get("rules_text"), str):
        merged.rules = [line.strip() for line in payload["rules_text"].splitlines() if line.strip()]

    if isinstance(payload.get("prompts"), dict):
        merged.prompts = PromptTemplates.from_mapping({**asdict(merged.prompts), **payload["prompts"]})

    return merged


def set_config_value(base: AppConfig, key_path: str, value: Any) -> AppConfig:
    """Set one dotted key path in config and return the updated config object."""
    dotted = str(key_path or "").strip()
    if not dotted:
        raise ValueError("key_path is required.")
    parts = dotted.split(".")
    payload = base.to_dict()
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor or not isinstance(cursor[part], dict):
            raise ValueError(f"Invalid key path: {dotted}")
        cursor = cursor[part]
    leaf = parts[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise ValueError(f"Invalid key path: {dotted}")
    cursor[leaf] = value
    return AppConfig.from_mapping(payload)


def update_config_value(path: str | Path | None, key_path: str, value: Any) -> Path:
    """Load config, set one dotted key path, save it, and return saved path."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = load_config(cfg_path)
    updated = set_config_value(config, key_path, value)
    return save_config(updated, cfg_path)
