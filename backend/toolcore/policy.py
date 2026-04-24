from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from backend.adapter.standard_request import StandardRequest
from backend.toolcall.runtime_tools import (
    is_list_directory_tool_name,
    is_read_tool_name,
    is_shell_tool_name,
    parse_tool_call_arguments,
    read_target_path,
    shell_command_signature,
    stable_tool_input_json,
)


@dataclass(slots=True)
class ToolPolicyDecision:
    kind: str
    reason: str | None = None


def _tool_identity(tool_name: str, tool_input: Any = None) -> str:
    try:
        if is_read_tool_name(tool_name):
            return f"read::{read_target_path(tool_input)}"
        if is_list_directory_tool_name(tool_name):
            return f"list_directory::{stable_tool_input_json(tool_input)}"
        if is_shell_tool_name(tool_name) and isinstance(tool_input, dict):
            command, workdir = shell_command_signature(tool_input)
            return f"bash::{str(command).strip()}::{str(workdir).strip()}"
        return f"{tool_name}::{stable_tool_input_json(tool_input)}"
    except Exception:
        return tool_name or ""


def recent_same_tool_identity_count_in_turn(messages: list[dict[str, Any]] | None, tool_name: str, tool_input: Any = None) -> int:
    target = _tool_identity(tool_name, tool_input)
    count = 0
    started = False
    for msg in reversed(messages or []):
        role = msg.get("role")
        if role == "user":
            break
        if role != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            if started:
                break
            continue
        started = True
        if len(tool_calls) != 1:
            break
        call = tool_calls[0]
        if not isinstance(call, dict):
            break
        fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
        name = fn.get("name", "")
        current_identity = _tool_identity(name, parse_tool_call_arguments(call))
        if current_identity == target:
            count += 1
            continue
        break
    return count


def evaluate_tool_policy(*, request: StandardRequest, state: Any, history_messages: list[dict[str, Any]] | None, can_retry_after_output: bool) -> ToolPolicyDecision:
    if state.blocked_tool_names and request.tools and can_retry_after_output:
        return ToolPolicyDecision(kind="retry", reason=f"blocked_tool_name:{state.blocked_tool_names[0]}")
    return ToolPolicyDecision(kind="accept")
