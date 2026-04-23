from __future__ import annotations

import json
from typing import Any

from backend.toolcore.types import (
    CanonicalToolCall,
    CanonicalToolResult,
    ToolChoicePolicy,
    ToolCoreRequest,
    ToolDefinition,
)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if raw_messages is None:
        return []
    if isinstance(raw_messages, str):
        return [{"role": "user", "content": raw_messages}]
    if isinstance(raw_messages, list):
        return [item for item in raw_messages if isinstance(item, dict)]
    if isinstance(raw_messages, dict):
        return [raw_messages]
    raise ValueError("messages/input must be a string, object, or list")


def _normalize_tools(raw_tools: Any) -> list[ToolDefinition]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list")

    tools: list[ToolDefinition] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        function_payload = _dict_or_empty(raw_tool.get("function"))
        name = str(
            raw_tool.get("name")
            or function_payload.get("name")
            or ""
        ).strip()
        if not name:
            continue
        description = str(
            raw_tool.get("description")
            or function_payload.get("description")
            or ""
        )
        parameters = raw_tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = function_payload.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                raw=raw_tool,
            )
        )
    return tools


def _parse_tool_choice(raw_tool_choice: Any, declared_tool_names: set[str]) -> tuple[ToolChoicePolicy, str | None]:
    if raw_tool_choice is None:
        return ToolChoicePolicy.AUTO, None

    if isinstance(raw_tool_choice, str):
        lowered = raw_tool_choice.strip().lower()
        if lowered == "auto":
            return ToolChoicePolicy.AUTO, None
        if lowered in {"required", "any"}:
            return ToolChoicePolicy.REQUIRED, None
        if lowered == "none":
            return ToolChoicePolicy.NONE, None
        raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")

    if not isinstance(raw_tool_choice, dict):
        raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")

    choice_type = str(raw_tool_choice.get("type") or "").strip().lower()
    if choice_type in {"", "auto"}:
        return ToolChoicePolicy.AUTO, None
    if choice_type in {"required", "any"}:
        return ToolChoicePolicy.REQUIRED, None
    if choice_type == "none":
        return ToolChoicePolicy.NONE, None
    if choice_type in {"function", "tool"}:
        function_payload = _dict_or_empty(raw_tool_choice.get("function"))
        forced_name = str(function_payload.get("name") or raw_tool_choice.get("name") or "").strip()
        if not forced_name:
            raise ValueError("Invalid tool_choice: missing forced tool name")
        if declared_tool_names and forced_name not in declared_tool_names:
            raise ValueError(f"Invalid tool_choice: undeclared tool {forced_name}")
        return ToolChoicePolicy.FORCED, forced_name
    raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")


def _canonical_tool_call_from_assistant_message(message: dict[str, Any]) -> list[CanonicalToolCall]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    canonical_calls: list[CanonicalToolCall] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function_payload = _dict_or_empty(call.get("function"))
        arguments = function_payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_input = json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed_input = {"value": arguments}
        elif isinstance(arguments, dict):
            parsed_input = arguments
        else:
            parsed_input = {}
        if not isinstance(parsed_input, dict):
            parsed_input = {"value": parsed_input}
        canonical_calls.append(
            CanonicalToolCall(
                call_id=str(call.get("id") or call.get("call_id") or "").strip(),
                name=str(function_payload.get("name") or call.get("name") or "").strip(),
                input=parsed_input,
            )
        )
    return [call for call in canonical_calls if call.call_id and call.name]


def _normalize_function_call_output(raw_output: Any) -> list[CanonicalToolResult]:
    if raw_output is None:
        return []
    items = raw_output if isinstance(raw_output, list) else [raw_output]
    out: list[CanonicalToolResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        if not call_id:
            continue
        out.append(
            CanonicalToolResult(
                call_id=call_id,
                output=item.get("output", ""),
                tool_name=str(item.get("name") or "").strip() or None,
            )
        )
    return out


def normalize_chat_request(req_data: dict[str, Any]) -> ToolCoreRequest:
    messages = _normalize_messages(req_data.get("messages"))
    tools = _normalize_tools(req_data.get("tools"))
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = req_data.get("tool_choice")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")

    tool_calls: list[CanonicalToolCall] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() == "assistant":
            tool_calls.extend(_canonical_tool_call_from_assistant_message(message))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        tool_calls=tool_calls,
        raw_tool_choice=raw_tool_choice,
    )


def normalize_responses_request(req_data: dict[str, Any]) -> ToolCoreRequest:
    messages = _normalize_messages(req_data.get("input", req_data.get("messages")))
    tools = _normalize_tools(req_data.get("tools"))
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = req_data.get("tool_choice")
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)

    tool_calls: list[CanonicalToolCall] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() == "assistant":
            tool_calls.extend(_canonical_tool_call_from_assistant_message(message))

    tool_results = _normalize_function_call_output(req_data.get("function_call_output"))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        tool_calls=tool_calls,
        tool_results=tool_results,
        raw_tool_choice=raw_tool_choice,
    )
