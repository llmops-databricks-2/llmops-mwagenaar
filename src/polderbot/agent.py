"""Minimal agent using an OpenAI-compatible Databricks endpoint."""

import json
from typing import Any, Protocol, cast
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)

from polderbot.tools import ToolInfo

_TOKEN_LIFETIME_SECONDS = 1200
_DEFAULT_MAX_ITERATIONS = 10


def remove_session_id_from_span(span: LiveSpan) -> None:
    """Remove session IDs from captured span inputs before export.

    Args:
        span: Span to filter.

    """
    inputs = span.inputs
    if isinstance(inputs, dict) and "session_id" in inputs:
        span.set_inputs({key: value for key, value in inputs.items() if key != "session_id"})


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
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Create an agent using the supplied workspace.

        Args:
            workspace: Authenticated Databricks workspace client.
            llm_endpoint: AI Gateway model service name.
            system_prompt: Instructions sent with each conversation.
            tools: Optional tools the model can call.
            memory: Optional message store for session-based conversations.
            max_iterations: Maximum number of model calls per message.

        """
        token = workspace.tokens.create(lifetime_seconds=_TOKEN_LIFETIME_SECONDS).token_value

        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory
        self.max_iterations = max_iterations

        self.tools = {tool.name: tool for tool in tools or []}
        self.tool_specs = [tool.spec for tool in self.tools.values()]

        base_url = f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1"
        self.client = OpenAI(api_key=token, base_url=base_url)

    def generate_message(
        self,
        message: str,
        *,
        session_id: str | None = None,
    ) -> str:
        """Send a message and execute requested tools until the model responds.

        Pass a session ID to continue a conversation using the configured memory.
        Otherwise, create a new session ID. When memory is configured, load and
        save each session's messages as they are generated.

        Args:
            message: User request to send to the model.
            session_id: Session to continue; requires configured memory when
                supplied. A new ID is generated when omitted.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If a supplied session has no store or the model requests
                an unsupported tool call type.
            RuntimeError: If the model provides no text or exhausts the limit.

        """
        try:
            return self._generate_message(message, session_id=session_id)
        finally:
            mlflow.flush_trace_async_logging()

    @mlflow.trace(name="generate_message", span_type=SpanType.AGENT)
    def _generate_message(
        self,
        message: str,
        *,
        session_id: str | None,
    ) -> str:
        """Generate a response within the agent trace.

        Args:
            message: User request to send to the model.
            session_id: Session to continue, or None to create a new one.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If a supplied session has no store or a tool call is unsupported.
            RuntimeError: If the model provides no text or exhausts the limit.

        """
        if session_id is not None and self.memory is None:
            raise ValueError("A message store is required for a session")

        if session_id is None:
            session_id = str(uuid4())

        mlflow.update_current_trace(session_id=session_id)

        messages = self._initialize_messages(message, session_id=session_id)

        for _ in range(self.max_iterations):
            answer = self._run_iteration(messages, session_id)

            if answer is not None:
                return answer

        raise RuntimeError("Model did not provide a final answer within the allowed iterations.")

    @mlflow.trace(name="initialize_messages", span_type=SpanType.MEMORY)
    def _initialize_messages(
        self,
        message: str,
        *,
        session_id: str,
    ) -> list[ChatCompletionMessageParam]:
        """Load prior conversation history and append the incoming user message.

        Args:
            message: User request to append.
            session_id: Session ID to load from and persist to when configured.

        Returns:
            The initialized message list including system prompt and user turn.

        """
        if self.memory is not None:
            previous = cast(list[ChatCompletionMessageParam], self.memory.load_messages(session_id))
        else:
            previous = []

        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": self.system_prompt}, *previous]
        user_message: ChatCompletionMessageParam = {"role": "user", "content": message}
        self._append_message(messages, user_message, session_id)

        return messages

    @mlflow.trace(name="run_iteration", span_type=SpanType.CHAIN)
    def _run_iteration(
        self,
        messages: list[ChatCompletionMessageParam],
        session_id: str,
    ) -> str | None:
        """Execute one model call and any requested function tools.

        Args:
            messages: Conversation messages updated in place.
            session_id: Session ID for persisting new messages when configured.

        Returns:
            The model's final text answer, or None when tools were executed.

        Raises:
            ValueError: If the model requests an unsupported tool call type.
            RuntimeError: If the model returns neither tool calls nor text.

        """
        assistant_message = self._call_model(messages)

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
            result = self._execute_tool(call["function"]["name"], call["function"]["arguments"])
            self._append_message(messages, {"role": "tool", "tool_call_id": call["id"], "content": result}, session_id)

        return None

    @mlflow.trace(name="call_model", span_type=SpanType.CHAT_MODEL)
    def _call_model(self, messages: list[ChatCompletionMessageParam]) -> ChatCompletionMessage:
        """Request the next assistant message from the chat model.

        Args:
            messages: Full conversation history to send to the model.

        Returns:
            The first choice's assistant message.

        """
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            tools=self.tool_specs or omit,  # type: ignore
        )
        return response.choices[0].message

    @mlflow.trace(name="execute_tool", span_type=SpanType.TOOL)
    def _execute_tool(self, name: str, raw_arguments: str) -> str:
        """Parse JSON tool arguments and invoke the registered tool function.

        Args:
            name: Registered tool name.
            raw_arguments: JSON-encoded keyword arguments for the tool.

        Returns:
            Stringified result returned by the tool's execution function.

        """
        arguments = json.loads(raw_arguments)

        return str(self.tools[name].exec_fn(**arguments))

    def _append_message(
        self,
        messages: list[ChatCompletionMessageParam],
        message: ChatCompletionMessageParam,
        session_id: str,
    ) -> None:
        """Persist a new message before adding it to the conversation."""
        if self.memory is not None:
            self.memory.save_messages(session_id, [dict(message)])

        messages.append(message)
