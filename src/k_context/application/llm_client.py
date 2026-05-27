"""LLM client 契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


TOKEN_USAGE_API = "api_usage"
TOKEN_USAGE_ESTIMATED = "estimated"
TOKEN_USAGE_UNAVAILABLE = "unavailable"
SUPPORTED_TOKEN_USAGE_SOURCES = {
    TOKEN_USAGE_API,
    TOKEN_USAGE_ESTIMATED,
    TOKEN_USAGE_UNAVAILABLE,
}

LLM_STATUS_SUCCESS = "success"
LLM_STATUS_FAILURE = "failure"
LLM_ROLE_SYSTEM = "system"
LLM_ROLE_USER = "user"
LLM_ROLE_ASSISTANT = "assistant"
LLM_ROLE_TOOL = "tool"
SUPPORTED_LLM_MESSAGE_ROLES = {
    LLM_ROLE_SYSTEM,
    LLM_ROLE_USER,
    LLM_ROLE_ASSISTANT,
    LLM_ROLE_TOOL,
}



class LLMClientError(RuntimeError):
    """当 LLM client 无法完成生成请求时抛出。"""


class LLMClientUnavailableError(LLMClientError):
    """当配置的 LLM provider 不可用时抛出。"""


@dataclass(frozen=True)
class ToolCall:
    """一次 OpenAI-compatible assistant 工具调用。"""

    tool_call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.tool_call_id).strip():
            raise ValueError("tool_call_id must not be empty.")
        if not str(self.name).strip():
            raise ValueError("tool call name must not be empty.")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool call arguments must be a mapping.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class LLMMessage:
    """一条 OpenAI-compatible chat 消息。"""

    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in SUPPORTED_LLM_MESSAGE_ROLES:
            raise ValueError(
                "Unsupported LLM message role: "
                f"{self.role}. Supported values: {', '.join(sorted(SUPPORTED_LLM_MESSAGE_ROLES))}."
            )
        if self.role == LLM_ROLE_TOOL and not str(self.tool_call_id or "").strip():
            raise ValueError("tool role messages must include tool_call_id.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _json_dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return payload


@dataclass(frozen=True)
class LLMRequest:
    """由问题和已检索来源构建的有依据 LLM 生成请求。"""

    question: str
    messages: tuple[LLMMessage, ...]
    sources: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: Mapping[str, Any] | str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """符合文档中 TokenUsage 契约的 token 用量记录。"""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    source: str

    def __post_init__(self) -> None:
        if self.source not in SUPPORTED_TOKEN_USAGE_SOURCES:
            raise ValueError(
                "Unsupported token usage source: "
                f"{self.source}. Supported values: {', '.join(sorted(SUPPORTED_TOKEN_USAGE_SOURCES))}."
            )
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided.")

    @classmethod
    def unavailable(cls) -> "TokenUsage":
        return cls(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            source=TOKEN_USAGE_UNAVAILABLE,
        )

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


@dataclass(frozen=True)
class LLMResponse:
    """LLM 回答以及用量、延迟、状态和脱敏错误字段。"""

    answer: str
    token_usage: TokenUsage
    token_usage_source: str
    latency_ms: float
    status: str
    error_message: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {LLM_STATUS_SUCCESS, LLM_STATUS_FAILURE}:
            raise ValueError("LLMResponse status must be success or failure.")
        if self.token_usage_source not in SUPPORTED_TOKEN_USAGE_SOURCES:
            raise ValueError(
                "Unsupported token_usage_source: "
                f"{self.token_usage_source}. Supported values: "
                f"{', '.join(sorted(SUPPORTED_TOKEN_USAGE_SOURCES))}."
            )
        if self.token_usage.source != self.token_usage_source:
            raise ValueError("token_usage.source must match token_usage_source.")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "token_usage": self.token_usage.to_dict(),
            "token_usage_source": self.token_usage_source,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error_message": self.error_message,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


class LLMClient(Protocol):
    """有依据 LLM 回答生成的应用层边界。"""

    model: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        """基于已检索上下文生成回答。"""


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False)
