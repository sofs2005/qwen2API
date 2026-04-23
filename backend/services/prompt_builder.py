import json
import logging
import re
from dataclasses import dataclass

from backend.adapter.standard_request import normalize_tool_choice
from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
    looks_like_opencode_system_prompt,
    sanitize_openclaw_user_text,
)
from backend.toolcore.prompt_contract import (
    build_tool_instruction_block as toolcore_build_tool_instruction_block,
    normalize_prompt_tool,
    normalize_prompt_tools,
    render_history_tool_call as toolcore_render_history_tool_call,
)

log = logging.getLogger("qwen2api.prompt")

@dataclass(slots=True)
class PromptBuildResult:
    prompt: str
    tools: list[dict]
    tool_enabled: bool
    client_profile: str


def _compact_history_tool_input(name: str, input_data: dict, client_profile: str) -> dict:
    if not _is_heavy_tool_profile(client_profile) or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch")
    for key in large_text_keys:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 160:
            compact[key] = f"[omitted {len(value)} chars]"
    if name in {"Write", "Edit", "NotebookEdit"}:
        preferred = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    return compact


def _render_history_tool_call(name: str, input_data: dict, client_profile: str) -> str:
    return toolcore_render_history_tool_call(name, input_data, client_profile)


def _build_tool_instruction_block(
    tools: list[dict],
    client_profile: str,
    *,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
) -> str:
    return toolcore_build_tool_instruction_block(
        tools,
        client_profile,
        tool_choice_mode=tool_choice_mode,
        required_tool_name=required_tool_name,
    )


def _sanitize_openclaw_user_text(text: str) -> str:
    return sanitize_openclaw_user_text(text)


def _extract_user_text_only(content, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return _sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if isinstance(content, list):
        text_blocks = []
        for part in content:
            if not isinstance(part, dict) or part.get("type", "") != "text":
                continue
            block_text = part.get("text", "")
            if client_profile == OPENCLAW_OPENAI_PROFILE:
                block_text = _sanitize_openclaw_user_text(block_text)
            if block_text:
                text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def _extract_text(content, user_tool_mode: bool = False, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return _sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if isinstance(content, list):
        parts = []
        text_blocks = []
        other_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type", "")
            if t == "text":
                block_text = part.get("text", "")
                if client_profile == OPENCLAW_OPENAI_PROFILE:
                    block_text = _sanitize_openclaw_user_text(block_text)
                if block_text:
                    text_blocks.append(block_text)
            elif t == "tool_use":
                other_parts.append(_render_history_tool_call(part.get("name", ""), part.get("input", {}), client_profile))
            elif t == "tool_result":
                inner = part.get("content", "")
                tid = part.get("tool_use_id", "")
                if isinstance(inner, str):
                    other_parts.append(f"[Tool Result for call {tid}]\n{inner}\n[/Tool Result]")
                elif isinstance(inner, list):
                    texts = [p.get("text", "") for p in inner if isinstance(p, dict) and p.get("type") == "text"]
                    other_parts.append(f"[Tool Result for call {tid}]\n{''.join(texts)}\n[/Tool Result]")
            elif t == "input_file":
                other_parts.append(f"[Attachment file_id={part.get('file_id','')} filename={part.get('filename','')}]")
            elif t == "input_image":
                other_parts.append(f"[Attachment image file_id={part.get('file_id','')} mime={part.get('mime_type','')}]")

        if user_tool_mode and text_blocks:
            parts.append(text_blocks[-1])
        else:
            parts.extend(text_blocks)
        parts.extend(other_parts)
        return "\n".join(p for p in parts if p)
    return ""


def _normalize_tool(tool: dict) -> dict:
    return normalize_prompt_tool(tool)


def _normalize_tools(tools: list) -> list:
    return normalize_prompt_tools(tools)


def _tool_param_hint(tool: dict, *, max_keys: int = 3) -> str:
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


def _tool_usage_line(tool: dict, *, max_desc: int = 40, max_keys: int = 3) -> str:
    name = tool.get("name", "")
    desc = (tool.get("description", "") or "")[:max_desc]
    hint = _tool_param_hint(tool, max_keys=max_keys)
    line = f"- {name}"
    if desc:
        line += f": {desc}"
    if hint:
        line += hint
    return line


def _find_tool_by_name(tools: list[dict], desired_name: str) -> dict | None:
    desired = str(desired_name or "").strip().lower()
    for tool in tools:
        actual = str(tool.get("name", "")).strip().lower()
        if actual == desired:
            return tool
    return None


def _preferred_tool_lines(
    tools: list[dict],
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


def _safe_preview(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:limit] + ("...[truncated]" if len(compact) > limit else "")


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


def build_prompt_with_tools(
    system_prompt: str,
    messages: list,
    tools: list,
    *,
    client_profile: str = OPENCLAW_OPENAI_PROFILE,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
) -> str:
    MAX_CHARS = 24000 if (tools and client_profile == QWEN_CODE_OPENAI_PROFILE) else (18000 if tools else 120000)
    sys_part = "" if tools and _is_heavy_tool_profile(client_profile) else (f"<system>\n{system_prompt[:2000]}\n</system>" if system_prompt else "")
    tools_part = ""
    if tools:
        tools_part = _build_tool_instruction_block(
            tools,
            client_profile,
            tool_choice_mode=tool_choice_mode,
            required_tool_name=required_tool_name,
        )
    opencode_override = bool(tools and client_profile == OPENCLAW_OPENAI_PROFILE and looks_like_opencode_system_prompt(system_prompt))
    if opencode_override and tools_part:
        tools_part = "\n".join(
            [
                "=== OPENCODE TOOL FORMAT OVERRIDE ===",
                "The opencode system prompt may describe native or built-in tool syntax.",
                "IGNORE those native tool formats for this gateway.",
                "For this qwen2API bridge, EVERY tool call MUST use ONLY the ##TOOL_CALL## / ##END_CALL## wrapper below.",
                "If you need to inspect files or directories, call the tool immediately using that wrapper.",
                "Never output plain-text plans such as 'I will inspect the directory first' before the tool call.",
                "=== END OVERRIDE ===",
                tools_part,
            ]
        )

    overhead = len(sys_part) + len(tools_part) + 50
    budget = MAX_CHARS - overhead
    history_parts = []
    used = 0
    NEEDSREVIEW_MARKERS = ("需求回显", "已了解规则", "等待用户输入", "待执行任务", "待确认事项",
                           "[需求回显]", "**需求回显**")
    msg_count = 0
    max_history_msgs = (
        16 if client_profile == QWEN_CODE_OPENAI_PROFILE else (12 if client_profile == CLAUDE_CODE_OPENAI_PROFILE else 8)
    ) if tools else 200
    for msg in reversed(messages):
        if msg_count >= max_history_msgs:
            break
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        if role == "system" and system_prompt and _extract_text(msg.get("content", ""), client_profile=client_profile).strip() == system_prompt.strip():
            continue

        if role == "tool":
            tool_content = msg.get("content", "") or ""
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    p.get("text", "") for p in tool_content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            elif not isinstance(tool_content, str):
                tool_content = str(tool_content)
            if tools and client_profile == QWEN_CODE_OPENAI_PROFILE:
                tool_result_limit = 12000
            elif tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE:
                tool_result_limit = 6000
            else:
                tool_result_limit = 300
            if len(tool_content) > tool_result_limit:
                tool_content = tool_content[:tool_result_limit] + "...[truncated]"
            line = f"[Tool Result]{(' id=' + tool_call_id) if tool_call_id else ''}\n{tool_content}\n[/Tool Result]"
            if used + len(line) + 2 > budget and history_parts:
                break
            history_parts.insert(0, line)
            used += len(line) + 2
            msg_count += 1
            continue

        user_text_only = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile) if role == "user" else ""
        text = _extract_text(
            msg.get("content", ""),
            user_tool_mode=(bool(tools) and role == "user" and _is_heavy_tool_profile(client_profile)),
            client_profile=client_profile,
        )

        if role == "assistant" and not text and msg.get("tool_calls"):
            tc_parts = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args_str}
                tc_parts.append(_render_history_tool_call(name, args, client_profile))
            text = "\n".join(tc_parts)

        if tools and role == "assistant" and any(m in text for m in NEEDSREVIEW_MARKERS):
            log.debug(f"[Prompt] 跳过需求回显式 assistant 消息 ({len(text)}字)")
            msg_count += 1
            continue
        lower_text = text.lower()
        is_tool_result = role == "user" and (
            "[tool result" in lower_text
            or text.startswith("{")
            or "\"results\"" in text[:100]
        )
        if client_profile == QWEN_CODE_OPENAI_PROFILE and tools:
            if is_tool_result:
                max_len = 10000
            elif role == "assistant":
                max_len = 700
            else:
                max_len = 2200
        elif client_profile == CLAUDE_CODE_OPENAI_PROFILE and tools:
            if is_tool_result:
                max_len = 6000
            elif role == "assistant":
                max_len = 500
            else:
                max_len = 1600
        else:
            max_len = 600 if is_tool_result else 1400
        if len(text) > max_len:
            text = text[:max_len] + "...[truncated]"
        is_tool_result_only_user_msg = role == "user" and not user_text_only.strip() and bool(text.strip())
        prefix = "" if is_tool_result_only_user_msg else {"user": "Human: ", "assistant": "Assistant: ", "system": "System: "}.get(role, "")
        line = text if is_tool_result_only_user_msg else f"{prefix}{text}"
        if used + len(line) + 2 > budget and history_parts:
            break
        history_parts.insert(0, line)
        used += len(line) + 2
        msg_count += 1

    if tools and messages and client_profile != CLAUDE_CODE_OPENAI_PROFILE:
        first_user = next(
            (
                m for m in messages
                if m.get("role") == "user"
                and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if first_user:
            first_text = _extract_user_text_only(first_user.get("content", ""), client_profile=client_profile)
            first_short = first_text[:800] + ("...[original task truncated]" if len(first_text) > 800 else "")
            first_line = f"Human: {first_short}"
            if not history_parts or not history_parts[0].startswith(f"Human: {first_text[:60]}"):
                first_line_cost = len(first_line) + 2
                if first_line_cost <= budget:
                    while history_parts and used + first_line_cost > budget:
                        removed = history_parts.pop()
                        used -= len(removed) + 2
                    history_parts.insert(0, first_line)
                    used += first_line_cost
                    log.debug(f"[Prompt] Restored original task context ({len(first_short)} chars)")


    latest_user_line = ""
    if tools and messages:
        latest_user = next(
            (
                m for m in reversed(messages)
                if m.get("role") == "user"
                and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if latest_user:
            latest_text = _extract_user_text_only(latest_user.get("content", ""), client_profile=client_profile).strip()
            if latest_text:
                latest_short = latest_text[:900] + ("...[latest task truncated]" if len(latest_text) > 900 else "")
                latest_user_line = f"Human (CURRENT TASK - TOP PRIORITY): {latest_short}"


    if tools and log.isEnabledFor(logging.DEBUG):
        tool_names = [tool.get("name", "") for tool in tools if tool.get("name")]
        tool_instruction_preview = _safe_preview(tools_part, 360)
        latest_user_preview = _safe_preview(latest_user_line, 220)
        first_user_preview = ""
        if messages:
            first_user = next((m for m in messages if m.get("role") == "user"), None)
            if first_user:
                first_user_preview = _safe_preview(
                    _extract_text(
                        first_user.get("content", ""),
                        user_tool_mode=_is_heavy_tool_profile(client_profile),
                        client_profile=client_profile,
                    ),
                    220,
                )
        log.debug(
            "[Prompt] 工具模式: history_msgs=%s history_chars=%s tool_count=%s tool_names=%s first_user=%r latest_user=%r tool_instr=%r",
            len(history_parts),
            used,
            len(tool_names),
            tool_names[:12],
            first_user_preview,
            latest_user_preview,
            tool_instruction_preview,
        )
    parts = []
    if tools_part and opencode_override:
        parts.append(tools_part)
    if sys_part:
        parts.append(sys_part)
    parts.extend(history_parts)
    if tools_part and not opencode_override:
        parts.append(tools_part)
    if latest_user_line:
        parts.append(latest_user_line)
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(req_data: dict, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> PromptBuildResult:
    resolved_client_profile = client_profile
    messages = req_data.get("messages", [])
    tools = _normalize_tools(req_data.get("tools", []))
    tool_enabled = bool(tools)
    tool_choice = normalize_tool_choice(req_data.get("tool_choice"))
    system_prompt = ""
    sys_field = req_data.get("system", "")
    if isinstance(sys_field, list):
        system_prompt = " ".join(p.get("text", "") for p in sys_field if isinstance(p, dict))
    elif isinstance(sys_field, str):
        system_prompt = sys_field
    if not system_prompt:
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = _extract_text(msg.get("content", ""), client_profile=resolved_client_profile)
                break
    return PromptBuildResult(
        prompt=build_prompt_with_tools(
            system_prompt,
            messages,
            tools,
            client_profile=resolved_client_profile,
            tool_choice_mode=tool_choice.mode,
            required_tool_name=tool_choice.required_tool_name,
        ),
        tools=tools,
        tool_enabled=tool_enabled,
        client_profile=resolved_client_profile,
    )
