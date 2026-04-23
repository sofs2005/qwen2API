import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.api import anthropic
from backend.services.response_formatters import build_anthropic_message_payload


class AnthropicToolCoreIntegrationTests(unittest.TestCase):
    def test_build_standard_request_preserves_tool_choice_fields(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read file",
                        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            }
        )

        self.assertEqual(request.tool_choice_mode, "required")
        self.assertEqual(request.required_tool_name, "Read")
        self.assertEqual(request.tool_choice_raw, {"type": "function", "function": {"name": "Read"}})

    def test_build_standard_request_rejects_undeclared_forced_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            anthropic._build_standard_request(
                {
                    "model": "claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "read the file"}],
                    "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
                    "tool_choice": {"type": "function", "function": {"name": "WebSearch"}},
                }
            )

    def test_build_standard_request_normalizes_anthropic_tools(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read file",
                        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    }
                ],
            }
        )

        self.assertEqual(request.tool_names, ["Read"])
        self.assertEqual(request.tools[0]["parameters"], {"type": "object", "properties": {"file_path": {"type": "string"}}})

    def test_anthropic_message_payload_formatter_matches_tool_directive(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
            }
        )
        execution = SimpleNamespace(state=SimpleNamespace(answer_text="", reasoning_text="", tool_calls=[{"id": "call_123", "name": "Read", "input": {"file_path": "README.md"}}]))

        payload = build_anthropic_message_payload(
            msg_id="msg_123",
            model_name="claude-3-5-sonnet",
            prompt="prompt",
            execution=execution,
            standard_request=request,
        )

        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "tool_use")
        self.assertEqual(payload["content"][0]["name"], "Read")


class AnthropicBridgeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_path_uses_retryable_completion_bridge(self) -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=SimpleNamespace(hold=lambda _key: _DummyLock()),
                account_pool=SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(
            app,
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
            },
        )
        standard_request = anthropic._build_standard_request(request._payload)
        bridge_result = SimpleNamespace(
            execution=SimpleNamespace(state=SimpleNamespace(answer_text="", reasoning_text="", tool_calls=[{"id": "call_123", "name": "Read", "input": {"file_path": "README.md"}}]), acc=None, chat_id=None),
            prompt="prompt",
            directive=None,
        )

        with patch.object(anthropic, "resolve_auth_context", AsyncMock(return_value=SimpleNamespace(token="tok"))), \
             patch.object(anthropic, "derive_session_key", return_value="session"), \
             patch.object(anthropic, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(anthropic, "plan_persistent_session_turn", AsyncMock(return_value=SimpleNamespace(enabled=False))), \
             patch.object(anthropic, "preprocess_attachments", AsyncMock(side_effect=lambda payload, *_args, **_kwargs: SimpleNamespace(payload=payload, attachments=[], uploaded_file_ids=[]))), \
             patch.object(anthropic, "run_retryable_completion_bridge", AsyncMock(return_value=bridge_result)) as bridge_mock, \
             patch.object(anthropic, "persist_session_turn", AsyncMock()), \
             patch.object(anthropic, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(anthropic, "update_request_context"), \
             patch.object(anthropic, "build_anthropic_assistant_history_message", return_value={"role": "assistant", "content": []}):
            response = await anthropic.anthropic_messages(request)

        self.assertEqual(response.status_code, 200)
        bridge_mock.assert_awaited_once()


class _DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb


class _FakeRequest:
    def __init__(self, app, payload):
        self.app = app
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
