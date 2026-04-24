import unittest

from backend.toolcore.formatter import (
    build_canonical_anthropic_message,
    build_canonical_gemini_payload,
    build_canonical_openai_chat_payload,
    build_canonical_openai_responses_payload,
)


class ToolCoreFormatterTests(unittest.TestCase):
    def test_openai_chat_formatter_renders_tool_calls(self) -> None:
        payload = build_canonical_openai_chat_payload(
            completion_id="chatcmpl_1",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "Read")

    def test_openai_responses_formatter_renders_function_call_items(self) -> None:
        payload = build_canonical_openai_responses_payload(
            response_id="resp_1",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["name"], "Read")

    def test_anthropic_formatter_renders_tool_use_blocks(self) -> None:
        payload = build_canonical_anthropic_message(
            msg_id="msg_1",
            model_name="claude-3-5-sonnet",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "tool_use")

    def test_gemini_formatter_renders_text_payload(self) -> None:
        payload = build_canonical_gemini_payload(answer_text="hello")

        self.assertEqual(payload["candidates"][0]["content"]["parts"][0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
