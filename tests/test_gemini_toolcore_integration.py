import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.responses import StreamingResponse

from backend.api import gemini


class GeminiToolCoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_and_validate_request_uses_standard_request_builder(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(users_db=object(), qwen_client=object()))
        request = _FakeRequest(
            app,
            {
                "contents": [{"role": "user", "parts": [{"text": "read the file"}]}],
                "tools": [{"functionDeclarations": [{"name": "Read", "description": "Read file", "parameters": {}}]}],
            },
        )
        standard_request = SimpleNamespace(prompt="prompt", resolved_model="qwen3.6-plus", tools=[], stream=False)

        with patch.object(gemini, "resolve_auth_context", AsyncMock(return_value=SimpleNamespace(token="tok"))), \
             patch.object(gemini, "build_chat_standard_request", return_value=standard_request) as builder_mock, \
             patch.object(gemini, "update_request_context"):
            _users_db, _client, _token, returned_request = await gemini._load_and_validate_request(request, "gemini-2.5-pro", force_stream=False)

        self.assertIs(returned_request, standard_request)
        builder_mock.assert_called_once()

    async def test_generate_content_uses_retryable_completion_bridge(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(users_db=object(), qwen_client=object()))
        request = _FakeRequest(app, {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]})
        standard_request = SimpleNamespace(prompt="prompt", resolved_model="qwen3.6-plus", response_model="gemini-2.5-pro", stream=False, tools=[])
        bridge_result = SimpleNamespace(
            execution=SimpleNamespace(state=SimpleNamespace(answer_text="hello", reasoning_text="", tool_calls=[]), acc=None, chat_id=None),
            prompt="prompt",
            directive=None,
        )

        with patch.object(gemini, "resolve_auth_context", AsyncMock(return_value=SimpleNamespace(token="tok"))), \
             patch.object(gemini, "build_chat_standard_request", return_value=standard_request), \
             patch.object(gemini, "run_retryable_completion_bridge", AsyncMock(return_value=bridge_result)) as bridge_mock, \
             patch.object(gemini, "update_request_context"), \
             patch.object(gemini, "build_gemini_generate_payload", return_value={"candidates": [{"content": {"parts": [{"text": "hello"}], "role": "model"}}]}):
            response = await gemini.gemini_generate_content("gemini-2.5-pro", request)

        self.assertEqual(response.status_code, 200)
        bridge_mock.assert_awaited_once()

    async def test_stream_generate_content_still_streams_text_chunks(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(users_db=object(), qwen_client=object()))
        request = _FakeRequest(app, {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]})
        standard_request = SimpleNamespace(prompt="prompt", resolved_model="qwen3.6-plus", response_model="gemini-2.5-pro", stream=True, tools=[])
        bridge_result = SimpleNamespace(
            execution=SimpleNamespace(state=SimpleNamespace(answer_text="done", reasoning_text="", tool_calls=[]), acc=None, chat_id=None),
            prompt="prompt",
            directive=None,
        )

        async def fake_bridge(**kwargs):
            await kwargs["on_delta"]({"phase": "answer"}, "chunk-1", None)
            return bridge_result

        with patch.object(gemini, "resolve_auth_context", AsyncMock(return_value=SimpleNamespace(token="tok"))), \
             patch.object(gemini, "build_chat_standard_request", return_value=standard_request), \
             patch.object(gemini, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(gemini, "update_request_context"):
            response = await gemini.gemini_stream_generate_content("gemini-2.5-pro", request)
            self.assertIsInstance(response, StreamingResponse)
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertTrue(any("chunk-1" in str(chunk) for chunk in chunks))


class _FakeRequest:
    def __init__(self, app, payload):
        self.app = app
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
