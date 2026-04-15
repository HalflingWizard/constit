"""Datamodels used by the constitutional engine and API layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Role = Literal["user", "assistant"]


def now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatMessage:
    """A user/assistant message in the conversation thread."""

    role: Role
    content: str
    at: str = field(default_factory=now_iso)

    @staticmethod
    def from_mapping(value: dict[str, Any]) -> "ChatMessage | None":
        """Create a message from untrusted input, or return None if invalid."""
        role = value.get("role")
        if role not in ("user", "assistant"):
            return None
        content = value.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        at = value.get("at")
        if not isinstance(at, str):
            at = now_iso()
        return ChatMessage(role=role, content=content, at=at)

    def to_openai(self) -> dict[str, str]:
        """Convert this message to OpenAI chat-completions format."""
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class UsageStats:
    """Token usage metadata returned by a completion call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "UsageStats":
        """Parse usage fields from an API response payload."""
        if not isinstance(value, dict):
            return UsageStats()
        return UsageStats(
            prompt_tokens=int(value.get("prompt_tokens", 0) or 0),
            completion_tokens=int(value.get("completion_tokens", 0) or 0),
            total_tokens=int(value.get("total_tokens", 0) or 0),
        )

    def add(self, other: "UsageStats") -> None:
        """Accumulate token usage in-place."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass(slots=True)
class WriterDraft:
    """A writer output generated during a turn."""

    at: str
    kind: str
    content: str
    usage: UsageStats
    iteration: int | None = None
    rule_index: int | None = None
    rule: str | None = None
    based_on_critique: str | None = None


@dataclass(slots=True)
class JudgeCheck:
    """A judge pass-check and optional critique for one rule evaluation."""

    at: str
    rule_index: int
    rule: str
    applies: bool
    passed: bool
    pass_raw: str
    pass_usage: UsageStats
    critique: str = ""
    required_fixes: str = ""
    critique_raw: str = ""
    critique_usage: UsageStats = field(default_factory=UsageStats)
    iteration: int | None = None


@dataclass(slots=True)
class TurnEvent:
    """A timeline event emitted during a constitutional turn."""

    at: str
    stage: str
    message: str
    mode: str
    rule_index: int | None = None
    rule: str | None = None
    iteration: int | None = None


@dataclass(slots=True)
class TurnTranscript:
    """Full structured transcript of one constitutional turn."""

    user: str
    thread: list[dict[str, str]]
    rules: list[str]
    id: str = field(default_factory=lambda: str(uuid4()))
    at: str = field(default_factory=now_iso)
    writer_drafts: list[WriterDraft] = field(default_factory=list)
    judge_checks: list[JudgeCheck] = field(default_factory=list)
    events: list[TurnEvent] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    duration_ms: int = 0
    final: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize transcript to a JSON-compatible dictionary."""
        raw = asdict(self)
        raw["writer"] = {"drafts": raw.pop("writer_drafts")}
        raw["judge"] = {"checks": raw.pop("judge_checks")}
        raw["run"] = {"events": raw.pop("events")}
        # Preserve backward-compatible key names used by the original UI.
        for check in raw["judge"]["checks"]:
            check["pass"] = check.pop("passed")
        return raw
