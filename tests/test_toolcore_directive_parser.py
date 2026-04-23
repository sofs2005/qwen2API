import unittest

from backend.toolcore.directive_parser import parse_state_tool_calls, parse_textual_tool_calls


class ToolDirectiveParserTests(unittest.TestCase):
    def test_canonical_executable_block_parses_to_canonical_call(self) -> None:
        result = parse_textual_tool_calls(
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##',
            [{"name": "Read", "parameters": {}}],
        )

        self.assertEqual(result.stop_reason, "tool_use")
        self.assertEqual(len(result.canonical_calls), 1)
        self.assertEqual(result.canonical_calls[0].name, "Read")
        self.assertEqual(result.canonical_calls[0].input, {"file_path": "README.md"})

    def test_invalid_block_returns_non_tool_result(self) -> None:
        result = parse_textual_tool_calls("plain text response", [{"name": "Read", "parameters": {}}])

        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.canonical_calls, [])
        self.assertEqual(result.tool_blocks, [{"type": "text", "text": "plain text response"}])

    def test_finalize_path_matches_stream_state_path(self) -> None:
        stream_result = parse_state_tool_calls(
            [{"id": "toolu_123", "name": "Read", "input": {"file_path": "README.md"}}],
            ["Read"],
        )
        text_result = parse_textual_tool_calls(
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##',
            [{"name": "Read", "parameters": {}}],
        )

        self.assertEqual(stream_result.stop_reason, text_result.stop_reason)
        self.assertEqual(stream_result.canonical_calls[0].name, text_result.canonical_calls[0].name)
        self.assertEqual(stream_result.canonical_calls[0].input, text_result.canonical_calls[0].input)

    def test_plain_legacy_text_stays_non_tool(self) -> None:
        result = parse_textual_tool_calls("I will read the file next.", [{"name": "Read", "parameters": {}}])

        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.canonical_calls, [])


if __name__ == "__main__":
    unittest.main()
