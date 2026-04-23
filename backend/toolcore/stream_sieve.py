from __future__ import annotations

from typing import Any

from backend.services.tool_parser import parse_tool_calls_silent


TOOL_START_MARKERS = ('{"name":', '<tool_call>', '##tool_call##', 'tool_call##', 'function.name:')


def looks_like_tool_fragment(text: str) -> bool:
    lowered = (text or "").lower()
    if "```" in lowered:
        return False
    return any(marker in lowered for marker in TOOL_START_MARKERS)


class ToolStreamSieve:
    def __init__(self, tool_names: list[str]):
        self.tool_names = [name for name in tool_names if isinstance(name, str) and name]
        self.pending = ""
        self.capture = ""
        self.capturing = False

    def process_chunk(self, chunk: str) -> list[dict[str, Any]]:
        if not chunk:
            return []

        self.pending += chunk
        events: list[dict[str, Any]] = []

        if self.capturing:
            self.capture += self.pending
            self.pending = ""
            prefix, calls, suffix, ready = self._consume_capture()
            if ready:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                if suffix:
                    self.pending = suffix
                self.capture = ""
                self.capturing = False
            return events

        start = self._find_tool_start(self.pending)
        if start >= 0:
            prefix = self.pending[:start]
            if prefix:
                events.append({"type": "content", "text": prefix})
            self.capture = self.pending[start:]
            self.pending = ""
            self.capturing = True
            prefix, calls, suffix, ready = self._consume_capture()
            if ready:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                if suffix:
                    self.pending = suffix
                self.capture = ""
                self.capturing = False
            return events

        safe, hold = self._split_safe_content(self.pending)
        if safe:
            events.append({"type": "content", "text": safe})
        self.pending = hold
        return events

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.capturing and self.capture:
            prefix, calls, suffix, ready = self._consume_capture()
            if ready:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                if suffix:
                    events.append({"type": "content", "text": suffix})
            else:
                events.append({"type": "content", "text": self.capture})
            self.capture = ""
            self.capturing = False
        if self.pending:
            events.append({"type": "content", "text": self.pending})
            self.pending = ""
        return events

    def _find_tool_start(self, text: str) -> int:
        lowered = text.lower()
        if "```" in lowered:
            return -1
        positions = [lowered.find(marker) for marker in TOOL_START_MARKERS if lowered.find(marker) >= 0]
        return min(positions) if positions else -1

    def _consume_capture(self) -> tuple[str, list[dict[str, Any]], str, bool]:
        if not self.capture:
            return "", [], "", False
        lowered = self.capture.lower()
        if "##tool_call##" in lowered and "##end_call##" not in lowered:
            return "", [], "", False
        if "<tool_call>" in lowered and "</tool_call>" not in lowered:
            return "", [], "", False
        blocks, stop_reason = parse_tool_calls_silent(self.capture, [{"name": name} for name in self.tool_names])
        if stop_reason == "tool_use":
            tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
            text_blocks = [block for block in blocks if block.get("type") == "text"]
            prefix = text_blocks[0].get("text", "") if text_blocks else ""
            calls = [{"name": block["name"], "input": block.get("input", {})} for block in tool_blocks]
            if calls:
                return prefix, calls, "", True
        if looks_like_tool_fragment(self.capture):
            return "", [], "", False
        return self.capture, [], "", True

    def _split_safe_content(self, text: str) -> tuple[str, str]:
        if len(text) < 24:
            return "", text
        return text[:-12], text[-12:]
