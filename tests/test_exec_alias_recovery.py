import unittest

from backend.runtime.execution import RuntimeAttemptState, evaluate_retry_directive, normalize_streamed_tool_calls
from backend.adapter.standard_request import StandardRequest


class ExecAliasRecoveryTests(unittest.TestCase):
    def test_native_tool_call_alias_is_normalized_before_streaming(self) -> None:
        normalized = normalize_streamed_tool_calls(
            [{"id": "call_1", "name": "exec", "input": {"command": "echo hi"}}],
            ["Bash"],
        )

        self.assertEqual(normalized[0]["name"], "Bash")

    def test_blocked_exec_name_rewrites_prompt_toward_real_shell_tool(self) -> None:
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            client_profile="openclaw_openai",
            tools=[{"name": "Bash", "parameters": {}}],
            tool_names=["Bash"],
            tool_enabled=True,
        )
        state = RuntimeAttemptState(
            answer_text='Tool exec does not exists.\n##TOOL_CALL##\n{"name": "exec", "input": {"command": "echo hi"}}\n##END_CALL##',
            blocked_tool_names=["exec"],
            emitted_visible_output=True,
        )

        retry = evaluate_retry_directive(
            request=request,
            current_prompt=request.prompt,
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=state,
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertIn("Bash", retry.next_prompt)
        self.assertNotIn("'exec'", retry.next_prompt)


if __name__ == "__main__":
    unittest.main()
