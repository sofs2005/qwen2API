from __future__ import annotations

import json
from typing import Any

from backend.toolcore.types import CanonicalToolResult


def build_response_function_call_item(*, call_id: str, name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(input_data, ensure_ascii=False),
    }


def response_function_call_to_message(item: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = item.get("arguments", "{}")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments or "{}",
                    },
                }
            ],
        }
    ]


def response_function_output_to_message(item: dict[str, Any]) -> list[dict[str, Any]]:
    output = item.get("output", "")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    return [{"role": "tool", "tool_call_id": item.get("call_id", ""), "content": output}]


def canonical_tool_result_from_response_output(item: dict[str, Any]) -> CanonicalToolResult | None:
    call_id = str(item.get("call_id") or item.get("id") or "").strip()
    if not call_id:
        return None
    return CanonicalToolResult(call_id=call_id, output=item.get("output", ""), tool_name=str(item.get("name") or "").strip() or None)
