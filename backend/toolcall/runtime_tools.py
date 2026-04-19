from __future__ import annotations

import json
from typing import Any


READ_TOOL_NAMES = {"read", "read_file", "readfile"}
LIST_DIRECTORY_TOOL_NAMES = {
    "glob",
    "search_files",
    "list_dir",
    "list_directory",
    "listdirectory",
    "listdir",
    "listfiles",
    "list_files",
    "ls",
}
SHELL_TOOL_NAMES = {
    "bash",
    "exec_command",
    "execcommand",
    "shell_command",
    "shellcommand",
    "run_shell_command",
    "runshellcommand",
}


def normalized_tool_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_read_tool_name(tool_name: str) -> bool:
    return normalized_tool_name(tool_name) in READ_TOOL_NAMES


def is_list_directory_tool_name(tool_name: str) -> bool:
    return normalized_tool_name(tool_name) in LIST_DIRECTORY_TOOL_NAMES


def is_shell_tool_name(tool_name: str) -> bool:
    return normalized_tool_name(tool_name) in SHELL_TOOL_NAMES


def read_target_path(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ""
    return str(
        tool_input.get("file_path")
        or tool_input.get("filePath")
        or tool_input.get("path")
        or tool_input.get("filename")
        or tool_input.get("target_file")
        or tool_input.get("target")
        or ""
    ).strip()


def shell_command_signature(tool_input: dict[str, Any]) -> tuple[str, str]:
    command = str(tool_input.get("command") or tool_input.get("cmd") or tool_input.get("script") or "").strip()
    workdir = str(tool_input.get("workdir") or tool_input.get("cwd") or tool_input.get("path") or "").strip()
    return command, workdir


def looks_like_listing_shell_command(command: str) -> bool:
    stripped = command.strip().lower()
    if not stripped:
        return False
    return stripped.split()[0] in {"ls", "dir", "pwd", "tree", "find"}


def stable_tool_input_json(tool_input: Any) -> str:
    try:
        return json.dumps(tool_input or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(tool_input or "")


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    raw_args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"raw": raw_args}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw_args}


def tool_target_preview(tool_name: str, args: dict[str, Any]) -> str:
    if is_read_tool_name(tool_name):
        return read_target_path(args)
    if is_list_directory_tool_name(tool_name):
        return str(args.get("path") or args.get("directory") or args.get("pattern") or ".").strip()
    if is_shell_tool_name(tool_name):
        command, _ = shell_command_signature(args)
        return command
    return ""
