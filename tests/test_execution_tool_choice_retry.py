import unittest

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import RuntimeAttemptState, evaluate_retry_directive


class ExecutionToolChoiceRetryTests(unittest.TestCase):
    def _request(self) -> StandardRequest:
        return StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Read", "parameters": {}}, {"name": "Write", "parameters": {}}],
            tool_names=["Read", "Write"],
            tool_enabled=True,
            tool_choice_mode="required",
            required_tool_name="Read",
        )

    def test_required_tool_choice_retries_when_no_tool_call_present(self) -> None:
        retry = evaluate_retry_directive(
            request=self._request(),
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(answer_text="plain text response", emitted_visible_output=True),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "required_tool_choice_missing_tool_call")

    def test_required_tool_choice_retries_when_wrong_tool_is_called(self) -> None:
        retry = evaluate_retry_directive(
            request=self._request(),
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(
                answer_text='<tool_call>{"name": "Write", "input": {"file_path": "a.txt", "content": "x"}}</tool_call>',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "required_tool_choice_wrong_tool:Write")

    def test_tool_choice_none_blocks_tool_call(self) -> None:
        request = self._request()
        request.tool_choice_mode = "none"
        request.required_tool_name = None

        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(
                answer_text='<tool_call>{"name": "Read", "input": {"file_path": "a.txt"}}</tool_call>',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "tool_choice_none_blocked_tool_call")

    def test_repeated_same_tool_does_not_cross_user_turn_boundary(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "exec", "parameters": {}}],
            tool_names=["exec"],
            tool_enabled=True,
        )

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
                            "arguments": '{"command": "mcporter call amap.maps_direction_driving --origin \\"上海市\\" --destination \\"无锡市\\""}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "用高德查一下上海到无锡的路线做成卡片发给我"},
        ]

        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=history_messages,
            attempt_index=0,
            max_attempts=2,
            state=RuntimeAttemptState(
                answer_text='##TOOL_CALL##\n{"name": "exec", "input": {"command": "mcporter call amap.maps_direction_driving --origin \\"上海市\\" --destination \\"无锡市\\""}}\n##END_CALL##',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertFalse(retry.retry)

    def test_analysis_task_does_not_retry_first_same_read(self) -> None:
        request = StandardRequest(
            prompt="Human: analyze this local script and explain how it works\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "read", "parameters": {}}],
            tool_names=["read"],
            tool_enabled=True,
        )

        history_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "script.py"}',
                        },
                    }
                ],
            }
        ]

        retry = evaluate_retry_directive(
            request=request,
            current_prompt=request.prompt,
            history_messages=history_messages,
            attempt_index=0,
            max_attempts=2,
            state=RuntimeAttemptState(
                answer_text='##TOOL_CALL##\n{"name": "read", "input": {"path": "script.py"}}\n##END_CALL##',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertFalse(retry.retry)


if __name__ == "__main__":
    unittest.main()
