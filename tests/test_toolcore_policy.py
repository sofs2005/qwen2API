import unittest

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import RuntimeAttemptState
from backend.toolcore.directive_parser import parse_textual_tool_calls
from backend.toolcore.policy import (
    ToolPolicyDecision,
    evaluate_tool_policy,
    recent_same_tool_identity_count_in_turn,
)


class ToolCorePolicyTests(unittest.TestCase):
    def test_parser_produces_canonical_directive_without_retry_decision(self) -> None:
        parsed = parse_textual_tool_calls(
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##',
            [{"name": "Read", "parameters": {}}],
        )

        self.assertEqual(parsed.stop_reason, "tool_use")
        self.assertEqual(len(parsed.canonical_calls), 1)
        self.assertEqual(parsed.canonical_calls[0].name, "Read")

    def test_recent_same_tool_identity_stops_at_user_turn_boundary(self) -> None:
        history_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": '{"command": "echo hi"}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "do it again"},
        ]

        count = recent_same_tool_identity_count_in_turn(history_messages, "exec", {"command": "echo hi"})

        self.assertEqual(count, 0)

    def test_policy_rejects_blocked_tool_text_without_mutating_parser_truth(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Bash", "parameters": {}}],
            tool_names=["Bash"],
            tool_enabled=True,
        )
        state = RuntimeAttemptState(
            answer_text="Tool exec does not exists.",
            blocked_tool_names=["exec"],
            emitted_visible_output=True,
        )

        decision = evaluate_tool_policy(
            request=request,
            state=state,
            history_messages=[],
            can_retry_after_output=True,
        )

        self.assertEqual(decision.kind, "retry")
        self.assertEqual(decision.reason, "blocked_tool_name:exec")


if __name__ == "__main__":
    unittest.main()
