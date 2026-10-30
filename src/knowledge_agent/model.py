from typing import Protocol

import httpx
from pydantic import Field

from .config import Settings
from .schemas import StrictModel


class FunctionCall(StrictModel):
    name: str = Field(min_length=1)
    arguments: str


class ToolCall(StrictModel):
    id: str = Field(min_length=1)
    type: str = "function"
    function: FunctionCall


class ModelMessage(StrictModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=12)


class ChatModel(Protocol):
    mode: str

    def respond(self, messages: list[dict], tools: list[dict]) -> ModelMessage: ...


class ModelError(Exception):
    """Safe-to-display provider failure; never contains headers or response bodies."""


class CompatibleChatModel:
    mode = "live"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        settings.require_live_model()
        self.settings = settings
        self.client = httpx.Client(
            timeout=httpx.Timeout(settings.model_timeout, connect=min(10, settings.model_timeout)),
            transport=transport,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def respond(self, messages: list[dict], tools: list[dict]) -> ModelMessage:
        try:
            response = self.client.post(
                self.settings.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.api_key.get_secret_value()},
                json={
                    "model": self.settings.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]
            # Ignore optional provider metadata such as reasoning_content; do not persist it.
            message = ModelMessage.model_validate(
                {k: raw[k] for k in ("role", "content", "tool_calls") if k in raw and raw[k] is not None}
            )
            if message.role != "assistant" or any(c.type != "function" for c in message.tool_calls):
                raise ValueError("invalid model role/type")
            ids = [c.id for c in message.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate tool call ids")
            return message
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"model_http_{exc.response.status_code}") from None
        except httpx.TimeoutException:
            raise ModelError("model_timeout") from None
        except httpx.RequestError:
            raise ModelError("model_connection_failed") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelError("model_invalid_response") from None
