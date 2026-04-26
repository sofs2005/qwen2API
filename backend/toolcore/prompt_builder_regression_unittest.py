from __future__ import annotations

import unittest

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.toolcore.prompt_builder import messages_to_prompt


class OpenClawPromptPollutionRegressionTest(unittest.TestCase):
    def test_short_opencode_system_prompt_is_removed(self) -> None:
        result = messages_to_prompt(
            {
                "messages": [
                    {"role": "system", "content": "You are opencode, an AI coding agent."},
                    {"role": "user", "content": "Explain decorators in Python."},
                ]
            },
            client_profile=OPENCLAW_OPENAI_PROFILE,
        )

        self.assertNotIn("You are opencode", result.prompt)
        self.assertNotIn("System: You are opencode", result.prompt)
        self.assertIn("Human: Explain decorators in Python.", result.prompt)

    def test_openclaw_runtime_native_tools_are_not_rendered_upstream(self) -> None:
        result = messages_to_prompt(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are opencode, an AI coding agent. Tool availability (filtered by policy): Read, Bash.",
                    },
                    {"role": "user", "content": "List the files in this repository."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Reads a file from the local filesystem.",
                            "parameters": {
                                "type": "object",
                                "properties": {"file_path": {"type": "string"}},
                                "required": ["file_path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Runs a shell command in the user's workspace.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    },
                ],
            },
            client_profile=OPENCLAW_OPENAI_PROFILE,
        )

        self.assertFalse(result.tool_enabled)
        self.assertEqual([], result.tools)
        self.assertNotIn("MANDATORY TOOL CALL INSTRUCTIONS", result.prompt)
        self.assertNotIn("OPENCODE TOOL FORMAT OVERRIDE", result.prompt)
        self.assertNotIn("You have access to these tools: Read, Bash", result.prompt)

    def test_claude_code_runtime_native_tools_are_not_rendered_upstream(self) -> None:
        result = messages_to_prompt(
            {
                "messages": [
                    {"role": "system", "content": "You are Claude Code, an AI coding agent."},
                    {"role": "user", "content": "Read pyproject.toml."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Reads a file from the local filesystem.",
                            "parameters": {
                                "type": "object",
                                "properties": {"file_path": {"type": "string"}},
                                "required": ["file_path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Runs a shell command in the user's workspace.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    },
                ],
            },
            client_profile=CLAUDE_CODE_OPENAI_PROFILE,
        )

        self.assertFalse(result.tool_enabled)
        self.assertEqual([], result.tools)
        self.assertNotIn("MANDATORY TOOL CALL INSTRUCTIONS", result.prompt)
        self.assertNotIn("You have access to these tools: Read, Bash", result.prompt)

    def test_regular_client_function_tools_are_preserved(self) -> None:
        result = messages_to_prompt(
            {
                "messages": [{"role": "user", "content": "What is the weather in Hangzhou?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Fetch weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
            client_profile=OPENCLAW_OPENAI_PROFILE,
        )

        self.assertTrue(result.tool_enabled)
        self.assertEqual(["get_weather"], [tool.get("name") for tool in result.tools])
        self.assertIn("You have access to these tools: get_weather", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): What is the weather in Hangzhou?", result.prompt)


if __name__ == "__main__":
    unittest.main()
