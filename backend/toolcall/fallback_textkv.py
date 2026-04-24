from __future__ import annotations

from .normalize import normalize_arguments


def parse_textkv_format(text: str, allowed_names: set[str]) -> list[dict[str, object]]:
    name = None
    arguments = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "function.name":
            name = value
        elif key == "function.arguments":
            arguments = value

    if not name:
        return []
    if name not in allowed_names:
        return []

    return [{
        "name": name,
        "input": normalize_arguments(arguments),
    }]
