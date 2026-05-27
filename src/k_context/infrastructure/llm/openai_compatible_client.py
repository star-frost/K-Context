"""OpenAI 兼容 Chat Completions LLM 客户端。"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from k_context.application.llm_client import (
    LLMClientError,
    LLMClientUnavailableError,
    LLMRequest,
    LLMResponse,
    LLM_STATUS_SUCCESS,
    TOKEN_USAGE_API,
    TokenUsage,
    ToolCall,
)
from k_context.domain.models import KContextConfig


KCONTEXT_LLM_API_KEY = "KCONTEXT_LLM_API_KEY"
KCONTEXT_LLM_BASE_URL = "KCONTEXT_LLM_BASE_URL"
KCONTEXT_LLM_MODEL = "KCONTEXT_LLM_MODEL"
KCONTEXT_LLM_THINKING = "KCONTEXT_LLM_THINKING"
KCONTEXT_LLM_REASONING_EFFORT = "KCONTEXT_LLM_REASONING_EFFORT"

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
THINKING_ENABLED = "enabled"
THINKING_DISABLED = "disabled"
CHAT_COMPLETIONS_PATH = "/chat/completions"
_SECRETISH_PATTERN = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|token)([\s:=\"]+)[A-Za-z0-9._~+/=-]{6,}"
)


class OpenAICompatibleLLMClient:
    """OpenAI-compatible chat completions API 的基础设施客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        environ: Mapping[str, str] | None = None,
        opener: Any | None = None,
        timeout_seconds: float | None = None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        self.base_url = _require_non_empty(base_url, "llm_base_url")
        self.model = _require_non_empty(model, "llm_model")
        self._api_key = _api_key_from_env(env)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.timeout_seconds = timeout_seconds
        self._opener = opener or build_opener()
        self.endpoint_url = _chat_completions_url(self.base_url)
        self.thinking = _normalize_thinking(
            thinking if thinking is not None else env.get(KCONTEXT_LLM_THINKING),
            base_url=self.base_url,
        )
        self.reasoning_effort = _optional_non_empty(
            reasoning_effort if reasoning_effort is not None else env.get(KCONTEXT_LLM_REASONING_EFFORT)
        )

    @classmethod
    def from_config(
        cls,
        config: KContextConfig,
        *,
        base_url: str | None = None,
        model: str | None = None,
        environ: Mapping[str, str] | None = None,
        opener: Any | None = None,
        timeout_seconds: float | None = None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> "OpenAICompatibleLLMClient":
        """根据运行时覆盖、环境变量、配置和默认值构建客户端。"""

        env = os.environ if environ is None else environ
        return cls(
            base_url=_first_non_empty(
                base_url,
                env.get(KCONTEXT_LLM_BASE_URL),
                config.llm_base_url,
                DEFAULT_DEEPSEEK_BASE_URL,
            ),
            model=_first_non_empty(
                model,
                env.get(KCONTEXT_LLM_MODEL),
                config.llm_model,
                DEFAULT_DEEPSEEK_MODEL,
            ),
            environ=env,
            opener=opener,
            timeout_seconds=timeout_seconds,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """向配置的 Chat Completions endpoint 发送有依据请求。"""

        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in request.messages],
            "stream": False,
        }
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if request.tools:
            payload["tools"] = [dict(tool) for tool in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        encoded_payload = json.dumps(payload).encode("utf-8")
        http_request = Request(
            self.endpoint_url,
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = perf_counter()
        try:
            with self._opener.open(http_request, timeout=self.timeout_seconds) as response:
                raw_body = response.read()
        except HTTPError as exc:
            raise LLMClientError(
                "LLM API HTTP error: "
                f"status={exc.code}; "
                f"provider_error={_http_error_body_summary(exc)}; "
                f"request_summary={_request_summary(payload)}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMClientUnavailableError(
                f"LLM API request failed: {_safe_exception_name(exc)}."
            ) from exc

        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        data = _decode_json_response(raw_body)
        tool_calls = _extract_tool_calls(data)
        answer = _extract_answer(data, allow_empty=bool(tool_calls))
        token_usage = _extract_token_usage(data)
        return LLMResponse(
            answer=answer,
            tool_calls=tool_calls,
            token_usage=token_usage,
            token_usage_source=token_usage.source,
            latency_ms=latency_ms,
            status=LLM_STATUS_SUCCESS,
            error_message=None,
        )


def _api_key_from_env(environ: Mapping[str, str]) -> str:
    try:
        api_key = environ[KCONTEXT_LLM_API_KEY]
    except KeyError as exc:
        raise LLMClientUnavailableError(f"{KCONTEXT_LLM_API_KEY} is not set.") from exc
    return _require_non_empty(api_key, KCONTEXT_LLM_API_KEY)


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and value.strip():
            return value
    return ""


def _require_non_empty(value: str, label: str) -> str:
    checked = value.strip()
    if not checked:
        raise LLMClientUnavailableError(f"{label} is not configured.")
    return checked


def _optional_non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    checked = value.strip()
    return checked or None


def _normalize_thinking(value: str | None, *, base_url: str) -> str | None:
    """返回 provider 安全的 DeepSeek thinking 模式。

    DeepSeek 当前 chat-completions API 对某些模型默认启用 thinking。
    本项目的工具调用客户端尚未跨轮次保留 reasoning_content，
    因此 DeepSeek 请求默认显式关闭 thinking，除非调用方通过
    ``KCONTEXT_LLM_THINKING`` 覆盖。
    非 DeepSeek 的 OpenAI-compatible provider 默认不会收到额外字段。
    """

    if value is None:
        return THINKING_DISABLED if "deepseek" in base_url.casefold() else None
    checked = value.strip().casefold()
    if checked in {"", "omit", "none"}:
        return None
    if checked in {"off", "false", "0", THINKING_DISABLED}:
        return THINKING_DISABLED
    if checked in {"on", "true", "1", THINKING_ENABLED}:
        return THINKING_ENABLED
    raise LLMClientUnavailableError(
        f"{KCONTEXT_LLM_THINKING} must be one of enabled, disabled, or omit."
    )


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith(CHAT_COMPLETIONS_PATH):
        return cleaned
    return f"{cleaned}{CHAT_COMPLETIONS_PATH}"


def _decode_json_response(raw_body: bytes) -> Mapping[str, Any]:
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMClientError("LLM API response is not valid JSON.") from exc
    if not isinstance(data, Mapping):
        raise LLMClientError("LLM API response must be a JSON object.")
    return data


def _http_error_body_summary(exc: HTTPError) -> str:
    try:
        raw_body = exc.read()
    except Exception:  # noqa: BLE001
        return "unavailable"
    if not raw_body:
        return "empty"
    try:
        decoded = raw_body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return "undecodable"
    return _sanitize_for_error(decoded[:1200])


def _request_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    message_roles = []
    tool_message_count = 0
    assistant_tool_call_message_count = 0
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            message_roles.append(role)
            if role == "tool":
                tool_message_count += 1
            if role == "assistant" and message.get("tool_calls"):
                assistant_tool_call_message_count += 1
    tool_choice = payload.get("tool_choice")
    thinking = payload.get("thinking")
    return {
        "model": payload.get("model"),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "message_roles": message_roles,
        "tools_count": len(payload.get("tools", ())) if isinstance(payload.get("tools"), list) else 0,
        "tool_choice": _tool_choice_summary(tool_choice),
        "thinking": thinking.get("type") if isinstance(thinking, Mapping) else None,
        "reasoning_effort_present": "reasoning_effort" in payload,
        "stream": payload.get("stream"),
        "tool_message_count": tool_message_count,
        "assistant_tool_call_message_count": assistant_tool_call_message_count,
    }


def _tool_choice_summary(tool_choice: Any) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            return f"function:{function.get('name')}"
        return str(tool_choice.get("type") or "object")
    return type(tool_choice).__name__


def _sanitize_for_error(text: str) -> str:
    sanitized = str(text)
    for value in (os.environ.get(KCONTEXT_LLM_API_KEY),):
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    sanitized = _SECRETISH_PATTERN.sub(r"\1\2[REDACTED]", sanitized)
    return sanitized.replace("Authorization", "[REDACTED]").replace("Bearer", "[REDACTED]")


def _first_message(data: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        choices = data["choices"]
        first_choice = choices[0]
        message = first_choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("LLM API response is missing choices[0].message.") from exc
    if not isinstance(message, Mapping):
        raise LLMClientError("LLM API response choices[0].message must be an object.")
    return message


def _extract_answer(data: Mapping[str, Any], *, allow_empty: bool = False) -> str:
    message = _first_message(data)
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise LLMClientError("LLM API response choices[0].message.content must be a string.")
    if not allow_empty and not content.strip():
        raise LLMClientError("LLM API response answer is empty.")
    return content


def _extract_tool_calls(data: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    message = _first_message(data)
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls is None:
        return ()
    if not isinstance(raw_tool_calls, list):
        raise LLMClientError("LLM API response message.tool_calls must be a list.")
    return tuple(_tool_call(item) for item in raw_tool_calls)


def _tool_call(item: Any) -> ToolCall:
    if not isinstance(item, Mapping):
        raise LLMClientError("LLM API tool call must be an object.")
    function = item.get("function")
    if not isinstance(function, Mapping):
        raise LLMClientError("LLM API tool call is missing function.")
    tool_call_id = item.get("id")
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        raise LLMClientError("LLM API tool call is missing id.")
    if not isinstance(name, str) or not name.strip():
        raise LLMClientError("LLM API tool call is missing function.name.")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM API tool call arguments are not valid JSON.") from exc
    elif isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    else:
        raise LLMClientError("LLM API tool call arguments must be JSON object text.")
    if not isinstance(arguments, Mapping):
        raise LLMClientError("LLM API tool call arguments must decode to an object.")
    return ToolCall(tool_call_id=tool_call_id, name=name, arguments=arguments)


def _extract_token_usage(data: Mapping[str, Any]) -> TokenUsage:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return TokenUsage.unavailable()
    return TokenUsage(
        prompt_tokens=_optional_non_negative_int(usage.get("prompt_tokens"), "prompt_tokens"),
        completion_tokens=_optional_non_negative_int(
            usage.get("completion_tokens"), "completion_tokens"
        ),
        total_tokens=_optional_non_negative_int(usage.get("total_tokens"), "total_tokens"),
        source=TOKEN_USAGE_API,
    )


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMClientError(f"LLM API usage.{field_name} must be an integer when provided.")
    if value < 0:
        raise LLMClientError(f"LLM API usage.{field_name} must be non-negative.")
    return value


def _safe_exception_name(exc: BaseException) -> str:
    return exc.__class__.__name__
