from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.toolcall.normalize import normalize_tool_name
from backend.toolcore.types import CanonicalToolCall
from backend.services import tool_parser


@dataclass(slots=True)
class ToolDirectiveParseResult:
    canonical_calls: list[CanonicalToolCall] = field(default_factory=list)
    tool_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"


def parse_state_tool_calls(state_tool_calls: list[dict[str, Any]], allowed_tool_names: list[str]) -> ToolDirectiveParseResult:
    canonical_calls: list[CanonicalToolCall] = []
    tool_blocks: list[dict[str, Any]] = []
    for tool_call in state_tool_calls:
        name = normalize_tool_name(str(tool_call.get("name", "")), allowed_tool_names)
        if not name:
            continue
        call_id = str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
        if not call_id:
            continue
        input_data = tool_call.get("input", {}) if isinstance(tool_call.get("input", {}), dict) else {}
        canonical = CanonicalToolCall(call_id=call_id, name=name, input=input_data)
        canonical_calls.append(canonical)
        tool_blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": input_data})
    stop_reason = "tool_use" if canonical_calls else "end_turn"
    return ToolDirectiveParseResult(canonical_calls=canonical_calls, tool_blocks=tool_blocks, stop_reason=stop_reason)


def parse_textual_tool_calls(answer_text: str, tools: list[dict[str, Any]]) -> ToolDirectiveParseResult:
    if not tools or not answer_text:
        return ToolDirectiveParseResult(tool_blocks=[{"type": "text", "text": answer_text}], stop_reason="end_turn")
    tool_blocks, stop_reason = tool_parser.parse_tool_calls_silent(answer_text, tools)
    canonical_calls: list[CanonicalToolCall] = []
    for block in tool_blocks:
        if block.get("type") != "tool_use" or not block.get("id") or not block.get("name"):
            continue
        input_data = block.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}
        canonical_calls.append(
            CanonicalToolCall(
                call_id=str(block.get("id") or "").strip(),
                name=str(block.get("name") or "").strip(),
                input=input_data,
            )
        )
    return ToolDirectiveParseResult(canonical_calls=canonical_calls, tool_blocks=tool_blocks, stop_reason=stop_reason)
