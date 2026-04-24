import unittest

from backend.services.tool_parser import parse_tool_calls_silent


class ToolCoreShellTests(unittest.TestCase):
    def test_tool_parser_shell_delegates_to_toolcore_for_hash_wrapper(self) -> None:
        blocks, stop_reason = parse_tool_calls_silent(
            '##TOOL_CALL##\n{"name": "Read", "input": {"path": "README.md"}}\n##END_CALL##',
            [{"name": "Read", "parameters": {"properties": {"path": {"type": "string"}}}}],
        )

        self.assertEqual(stop_reason, "tool_use")
        self.assertEqual(blocks[0]["type"], "tool_use")
        self.assertEqual(blocks[0]["name"], "Read")


if __name__ == "__main__":
    unittest.main()
