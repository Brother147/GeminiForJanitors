"""Data models.

The name of some fields in here may not match with the source they are from."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from json import loads
from math import isfinite
from typing import Any

from .commands import Command, parse_message, strip_message
from .utils import comma_split

################################################################################


@dataclass(kw_only=True, slots=True)
class JaiMessage:
    """JanitorAI message, retaining provider-specific OpenAI fields."""

    commands: list[Command] = field(default_factory=list)
    content: Any = "."
    role: str = "user"
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(data: dict | str):
        if isinstance(data, str):
            data = loads(data)
        if not isinstance(data, dict):
            raise TypeError("Invalid data")

        jai_msg = JaiMessage()

        role = data.get("role")
        if role is not None:
            if not isinstance(role, str):
                raise TypeError("Message role must be a string")
            if role:
                if len(role) > 64:
                    raise ValueError("Message role is too long")
                jai_msg.role = role

        content = data.get("content", ".")
        if content is not None and not isinstance(content, (str, list)):
            raise TypeError("Message content must be a string, array, or null")
        if isinstance(content, str) and len(content) > 8 * 1024 * 1024:
            raise ValueError("Message content is too large")
        if isinstance(content, list) and len(content) > 4096:
            raise ValueError("Message content contains too many parts")

        # Only plain user text is parsed for this proxy's //commands. Structured
        # content (images/audio/etc.) and non-user messages must pass through
        # untouched so provider-specific message shapes remain valid.
        if isinstance(content, str) and content:
            if jai_msg.role == "user":
                jai_msg.commands, jai_msg.content = parse_message(content)
            else:
                jai_msg.content = strip_message(content)
        else:
            jai_msg.content = content

        jai_msg.extra_fields = {
            key: value for key, value in data.items() if key not in {"role", "content"}
        }
        return jai_msg

    def to_payload(self) -> dict[str, Any]:
        """Recreate the original chat message without proxy-only command state."""
        payload = dict(self.extra_fields)
        payload["role"] = self.role
        payload["content"] = self.content
        return payload

    def text_content(self) -> str:
        """Return text suitable for local command detection and logging."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: list[str] = []
            for part in self.content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        return ""


@dataclass(kw_only=True, slots=True)
class JaiRequest:
    """JanitorAI request with unknown top-level fields retained."""

    # Authorization header
    api_key: str = ""
    api_key_index: int = 1
    api_key_count: int = 1

    # Request body
    max_tokens: int = 0
    messages: list[JaiMessage] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    temperature: float = 0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Request URL
    quiet: bool = False
    quiet_commands: bool = False  # Only for testing

    # Commands
    use_fixturns: bool = False

    def append_message(self, role: str, content: str):
        self.messages.append(
            JaiMessage(
                content=content,
                role=role,
            )
        )

    @staticmethod
    def parse(data: dict | str):
        if isinstance(data, str):
            data = loads(data)
        if not isinstance(data, dict):
            raise TypeError("Invalid data")

        jai_req = JaiRequest()

        max_tokens = data.get("max_tokens")
        if max_tokens is not None:
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
                raise TypeError("Request max_tokens must be an integer")
            if not 0 <= max_tokens <= 1_000_000:
                raise ValueError("Request max_tokens is outside the supported range")
            jai_req.max_tokens = max_tokens

        messages = data.get("messages")
        if messages is not None:
            if not isinstance(messages, list):
                raise TypeError("Request messages must be a list")
            if len(messages) > 8192:
                raise ValueError("Request contains too many messages")
            jai_req.messages = [JaiMessage.parse(jai_msg) for jai_msg in messages]

        models = data.get("model")
        if models is not None:
            if not isinstance(models, str):
                raise TypeError("Request model must be a string")
            if len(models) > 2048:
                raise ValueError("Request model field is too long")
            if models:
                for model in comma_split(models):
                    normalized_model = model.lower()
                    if "/" in model:
                        provider, model_name = model.split("/", maxsplit=1)
                        provider = provider.lower()
                        # Radeon model IDs are case-sensitive; preserve the exact
                        # spelling JanitorAI supplied. Keep existing normalization
                        # for every other provider.
                        if provider != "radeon":
                            model_name = model_name.lower()
                        jai_req.models[provider] = model_name
                    elif normalized_model.startswith(("gemini-", "gemma-")):
                        jai_req.models["google"] = normalized_model
                    elif normalized_model.startswith("deepseek-"):
                        jai_req.models["deepseek"] = normalized_model
                    else:
                        if unknown := jai_req.models.get("unknown"):
                            jai_req.models["unknown"] = f"{unknown}, {normalized_model}"
                        else:
                            jai_req.models["unknown"] = normalized_model

        stream = data.get("stream")
        if stream is not None:
            if not isinstance(stream, bool):
                raise TypeError("Request stream must be a boolean")
            jai_req.stream = stream

        numeric_fields = (
            "temperature",
            "frequency_penalty",
            "repetition_penalty",
            "top_p",
        )
        for field_name in numeric_fields:
            value = data.get(field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"Request {field_name} must be a number")
                if abs(value) > 1_000_000 or not isfinite(value):
                    raise ValueError(
                        f"Request {field_name} is outside the supported range"
                    )
                setattr(jai_req, field_name, value)

        top_k = data.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("Request top_k must be an integer")
            if not 0 <= top_k <= 1_000_000:
                raise ValueError("Request top_k is outside the supported range")
            jai_req.top_k = top_k

        # Keep every other JSON field. Provider adapters decide which options
        # their upstream API accepts; this layer must not discard new JanitorAI
        # controls such as reasoning_effort/reasoning/thinking.
        known_fields = {
            "model", "messages", "stream", "max_tokens", "temperature",
            "frequency_penalty", "repetition_penalty", "top_k", "top_p",
        }
        jai_req.extra_params = {
            key: value for key, value in data.items() if key not in known_fields
        }

        return jai_req

################################################################################


@dataclass(kw_only=True, slots=True)
class JaiResultTokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(kw_only=True, slots=True)
class JaiResultMetadata:
    "JanitorAI Generate Content Result Metadata"

    api_key_valid: bool = True
    rejection_feedback: str = ""
    token_usage: JaiResultTokenUsage | None = None


@dataclass(init=False, kw_only=True, slots=True)
class JaiResult:
    "JanitorAI Generate Content Result"

    status: int
    text: str
    error: str
    extras: str
    metadata: JaiResultMetadata
    stream: Iterator[str] | None

    def __init__(
        self,
        status: int,
        message: str,
        *,
        extras: str = "",
        metadata: JaiResultMetadata | None = None,
        stream: Iterator[str] | None = None,
    ):
        self.status = status

        if status == 200:
            self.text = message
            self.error = ""
        else:
            self.text = ""
            self.error = message

        self.extras = extras
        self.metadata = metadata or JaiResultMetadata()
        self.stream = stream

    def __bool__(self) -> bool:
        return self.status == 200


################################################################################
