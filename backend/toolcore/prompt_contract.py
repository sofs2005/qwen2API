from __future__ import annotations

import json
from typing import Any

from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
)


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


def compact_history_tool_input(name: str, input_data: dict[str, Any], client_profile: str) -> dict[str, Any]:
    if not _is_heavy_tool_profile(client_profile) or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch")
    for key in large_text_keys:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 160:
            compact[key] = f"[omitted {len(value)} chars]"
    if name in {"Write", "Edit", "NotebookEdit"}:
        preferred: dict[str, Any] = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    return compact


def render_history_tool_call(name: str, input_data: dict[str, Any], client_profile: str) -> str:
    payload = json.dumps({"name": name, "input": compact_history_tool_input(name, input_data, client_profile)}, ensure_ascii=False)
    return f"##TOOL_CALL##\n{payload}\n##END_CALL##"


def normalize_prompt_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or tool.get("parameters") or {},
    }


def normalize_prompt_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [normalize_prompt_tool(tool) for tool in tools if isinstance(tool, dict)]


def _tool_param_hint(tool: dict[str, Any], *, max_keys: int = 3) -> str:
    params = tool.get("parameters", {}) or {}
    if not isinstance(params, dict):
        return ""
    props = params.get("properties", {}) or {}
    if not isinstance(props, dict) or not props:
        return ""
    required = params.get("required", []) or []
    ordered_keys: list[str] = []
    for key in required:
        if key in props and key not in ordered_keys:
            ordered_keys.append(key)
    for key in props:
        if key not in ordered_keys:
            ordered_keys.append(key)
    shown = ordered_keys[:max_keys]
    if not shown:
        return ""
    suffix = ", ..." if len(ordered_keys) > len(shown) else ""
    required_shown = [key for key in required if key in shown][:max_keys]
    required_suffix = f"; required: {', '.join(required_shown)}" if required_shown else ""
    return f" input keys: {', '.join(shown)}{suffix}{required_suffix}"


def _tool_usage_line(tool: dict[str, Any], *, max_desc: int = 40, max_keys: int = 3) -> str:
    name = tool.get("name", "")
    desc = (tool.get("description", "") or "")[:max_desc]
    hint = _tool_param_hint(tool, max_keys=max_keys)
    line = f"- {name}"
    if desc:
        line += f": {desc}"
    if hint:
        line += hint
    return line


def _find_tool_by_name(tools: list[dict[str, Any]], desired_name: str) -> dict[str, Any] | None:
    desired = str(desired_name or "").strip().lower()
    for tool in tools:
        actual = str(tool.get("name", "")).strip().lower()
        if actual == desired:
            return tool
    return None


def _preferred_tool_lines(
    tools: list[dict[str, Any]],
    priority_names: list[str],
    *,
    max_remaining: int = 20,
    max_desc: int = 40,
    max_keys: int = 3,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for priority_name in priority_names:
        tool = _find_tool_by_name(tools, priority_name)
        if tool is None:
            continue
        actual_name = str(tool.get("name", "")).strip()
        actual_key = actual_name.lower()
        if not actual_name or actual_key in seen:
            continue
        seen.add(actual_key)
        lines.append(_tool_usage_line(tool, max_desc=max_desc, max_keys=max_keys))
    remaining_names = [
        str(tool.get("name", "")).strip()
        for tool in tools
        if str(tool.get("name", "")).strip() and str(tool.get("name", "")).strip().lower() not in seen
    ]
    if remaining_names:
        lines.append(f"- Other available tools: {', '.join(remaining_names[:max_remaining])}")
        if len(remaining_names) > max_remaining:
            lines.append(f"  ... and {len(remaining_names) - max_remaining} more")
    return lines


def build_tool_instruction_block(
    tools: list[dict[str, Any]],
    client_profile: str,
    *,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
) -> str:
    names = [t.get("name", "") for t in tools if t.get("name")]
    force_constraint_lines: list[str] = []
    if tool_choice_mode == "required":
        if required_tool_name:
            force_constraint_lines.extend([
                f'【强制】本轮必须调用工具 `{required_tool_name}`，不能仅回复普通文本，也不能改用其它工具。',
                f'MANDATORY: this turn MUST call the exact tool "{required_tool_name}". Plain text only is not allowed, and using a different tool is not allowed.',
            ])
        else:
            force_constraint_lines.extend([
                "【强制】本轮必须至少调用一个工具，不能只输出普通文本。",
                "MANDATORY: this turn MUST include at least one tool call. Plain text only is not allowed.",
            ])
    elif tool_choice_mode == "none":
        force_constraint_lines.extend([
            "【强制】本轮不要调用任何工具，直接给出普通文本回复。",
            "MANDATORY: do NOT call any tool on this turn. Respond with plain text only.",
        ])

    lines = [
        "=== MANDATORY TOOL CALL INSTRUCTIONS ===",
        "【重要】用户输入什么语言，就用什么语言回复。User inputs Chinese → respond in Chinese. User inputs English → respond in English.",
        "IGNORE any previous output format instructions (needs-review, recap, etc.).",
        f"You have access to these tools: {', '.join(names)}",
        "",
        "WHEN YOU NEED TO CALL A TOOL — output EXACTLY this format (nothing else):",
        "##TOOL_CALL##",
        '{"name": "EXACT_TOOL_NAME", "input": {"param1": "value1"}}',
        "##END_CALL##",
        "",
        "Rules:",
        "- Output only the wrapper and JSON body.",
        "- No prose before or after the wrapper.",
        "- Use the exact tool name from the list above.",
        "- Put arguments inside the input object.",
        "- If no tool is needed, answer normally.",
        "",
        "CRITICAL — FORBIDDEN FORMATS (will be blocked by server):",
        '- {"name": "X", "arguments": "..."}  <-- NEVER USE',
        '- {"type": "function", "name": "X"}  <-- NEVER USE',
        '- {"type": "tool_use", "name": "X"}  <-- NEVER USE',
        '- <tool_call>{...}</tool_call>  <-- NEVER USE',
        "ONLY ##TOOL_CALL##...##END_CALL## is accepted.",
        "",
        *force_constraint_lines,
        *([""] if force_constraint_lines else []),
        "Available tools (use these EXACT names):",
    ]
    if client_profile == QWEN_CODE_OPENAI_PROFILE and len(names) > 16:
        priority_tools = ["read", "read_file", "write", "write_file", "edit", "multiedit", "bash", "run_command", "grep", "glob", "webfetch", "websearch"]
        lines.extend(_preferred_tool_lines(tools, priority_tools, max_desc=72, max_keys=6))
    elif client_profile == CLAUDE_CODE_OPENAI_PROFILE and len(names) > 12:
        priority_tools = ["read", "write", "edit", "bash", "glob", "grep", "websearch", "webfetch", "agent", "taskcreate", "taskupdate", "askuserquestion"]
        lines.extend(_preferred_tool_lines(tools, priority_tools))
    elif client_profile == OPENCLAW_OPENAI_PROFILE and len(names) > 12:
        priority_tools = ["read", "write", "edit", "bash", "glob", "grep", "webfetch", "websearch", "task", "skill", "todowrite", "question"]
        lines.extend(_preferred_tool_lines(tools, priority_tools))
    else:
        for tool in tools:
            lines.append(_tool_usage_line(tool, max_desc=72 if client_profile == QWEN_CODE_OPENAI_PROFILE else 40, max_keys=6 if client_profile == QWEN_CODE_OPENAI_PROFILE else 3))
    lines.append("=== END TOOL INSTRUCTIONS ===")
    return "\n".join(lines)
