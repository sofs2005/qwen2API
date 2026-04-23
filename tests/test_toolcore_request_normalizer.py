import unittest

from backend.toolcore.request_normalizer import normalize_chat_request, normalize_responses_request
from backend.toolcore.types import ToolChoicePolicy, ToolCoreRequest


class ChatRequestNormalizationTests(unittest.TestCase):
    def test_chat_request_with_tools(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        self.assertIsInstance(result, ToolCoreRequest)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].name, "get_weather")
        self.assertEqual(result.tools[0].description, "Get weather for a location")
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.AUTO)
        self.assertEqual(result.messages, [{"role": "user", "content": "What's the weather in Tokyo?"}])

    def test_chat_request_required_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Call a tool"}],
                "tools": [{"type": "function", "function": {"name": "test_tool"}}],
                "tool_choice": "required",
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.REQUIRED)

    def test_chat_request_none_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Don't call tools"}],
                "tools": [{"type": "function", "function": {"name": "test_tool"}}],
                "tool_choice": "none",
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.NONE)

    def test_chat_request_specific_function_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Call get_weather"}],
                "tools": [
                    {"type": "function", "function": {"name": "get_weather"}},
                    {"type": "function", "function": {"name": "get_time"}},
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.FORCED)
        self.assertEqual(result.forced_tool_name, "get_weather")

    def test_chat_request_invalid_tool_choice_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid tool_choice"):
            normalize_chat_request(
                {
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "tool_choice": "invalid_value",
                }
            )


class ResponsesRequestNormalizationTests(unittest.TestCase):
    def test_responses_request_with_tools(self) -> None:
        result = normalize_responses_request(
            {
                "input": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        self.assertIsInstance(result, ToolCoreRequest)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].name, "get_weather")
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.AUTO)
        self.assertEqual(result.messages, [{"role": "user", "content": "What's the weather?"}])

    def test_responses_request_with_function_call_output(self) -> None:
        result = normalize_responses_request(
            {
                "input": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                            }
                        ],
                    }
                ],
                "tools": [],
                "function_call_output": {"call_id": "call_123", "output": "Sunny, 25°C"},
            }
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].call_id, "call_123")
        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].input, {"city": "Tokyo"})
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0].call_id, "call_123")
        self.assertEqual(result.tool_results[0].output, "Sunny, 25°C")

    def test_responses_request_without_tools_rejects_tool_choice(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool_choice provided but no tools"):
            normalize_responses_request(
                {
                    "input": [{"role": "user", "content": "Hi"}],
                    "tools": [],
                    "tool_choice": "auto",
                }
            )

    def test_responses_request_invalid_tool_choice_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid tool_choice"):
            normalize_responses_request(
                {
                    "input": [{"role": "user", "content": "Test"}],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "tool_choice": "invalid",
                }
            )


if __name__ == "__main__":
    unittest.main()
