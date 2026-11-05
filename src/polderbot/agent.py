"""Minimal agent using an OpenAI-compatible Databricks endpoint."""

import json
from typing import Any, Protocol, cast

from databricks.sdk import WorkspaceClient
from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)

from polderbot.tools import ToolInfo

_TOKEN_LIFETIME_SECONDS = 1200
_DEFAULT_MAX_ITERATIONS = 10


class MessageStore(Protocol):
    """Append and retrieve JSON-compatible messages for a session."""

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in order."""
        ...

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append new messages to a session."""
        ...


class PolderAgent:
    """Call a chat model and execute its requested function tools."""

    def __init__(
        self,
        workspace: WorkspaceClient,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo] | None = None,
        memory: MessageStore | None = None,
    ) -> None:
        """Create an agent using the supplied workspace.

        Args:
            workspace: Authenticated Databricks workspace client.
            llm_endpoint: AI Gateway model service name.
            system_prompt: Instructions sent with each conversation.
            tools: Optional tools the model can call.
            memory: Optional message store for session-based conversations.

        """
        token = workspace.tokens.create(lifetime_seconds=_TOKEN_LIFETIME_SECONDS).token_value

        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory

        self.tools = {tool.name: tool for tool in tools or []}
        self.tool_specs = [tool.spec for tool in self.tools.values()]

        base_url = f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1"
        self.client = OpenAI(api_key=token, base_url=base_url)

    def generate_message(
        self,
        message: str,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        *,
        session_id: str | None = None,
        history: list[ChatCompletionMessageParam] | None = None,
    ) -> str:
        """Send a message and execute requested tools until the model responds.

        Pass a session ID to load its history from the configured memory and
        save each new user, assistant, and tool message as it is generated.
        Alternatively, pass history directly to bypass memory entirely. History
        should not contain the system prompt. Passing neither starts a one-off
        conversation; passing both is not allowed.

        Args:
            message: User request to send to the model.
            max_iterations: Maximum number of model calls.
            session_id: Session to load and append; requires configured memory.
            history: Prior messages to use without reading or writing memory.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If the session and history conflict, the store is absent,
                or the model requests an unsupported tool call type.
            RuntimeError: If the model provides no text or exhausts the limit.

        """
        if session_id is not None and history is not None:
            raise ValueError("Provide either session_id or history, not both")
        if session_id is not None and self.memory is None:
            raise ValueError("A message store is required for a session")

        if session_id is not None and self.memory is not None:
            previous = cast(list[ChatCompletionMessageParam], self.memory.load_messages(session_id))
        else:
            previous = list(history) if history is not None else []

        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": self.system_prompt}, *previous]
        user_message: ChatCompletionMessageParam = {"role": "user", "content": message}
        self._append_message(messages, user_message, session_id)

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.tool_specs or omit,  # type: ignore
            )
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                if assistant_message.content is None:
                    raise RuntimeError("Model did not provide a final answer.")

                self._append_message(messages, {"role": "assistant", "content": assistant_message.content}, session_id)
                return assistant_message.content

            calls: list[ChatCompletionMessageFunctionToolCallParam] = []

            for tool_call in assistant_message.tool_calls:
                if tool_call.type != "function":
                    raise ValueError(f"Unsupported tool call type: {tool_call.type}")

                calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            self._append_message(
                messages, {"role": "assistant", "content": assistant_message.content, "tool_calls": calls}, session_id
            )

            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                result = self.tools[name].exec_fn(**arguments)
                self._append_message(
                    messages, {"role": "tool", "tool_call_id": call["id"], "content": str(result)}, session_id
                )

        raise RuntimeError("Model did not provide a final answer within the allowed iterations.")

    def _append_message(
        self,
        messages: list[ChatCompletionMessageParam],
        message: ChatCompletionMessageParam,
        session_id: str | None,
    ) -> None:
        """Persist a new message before adding it to the conversation."""
        if session_id is not None and self.memory is not None:
            self.memory.save_messages(session_id, [dict(message)])
        messages.append(message)
