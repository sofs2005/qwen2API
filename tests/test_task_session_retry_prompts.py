import unittest

from backend.adapter.standard_request import StandardRequest
from backend.toolcore.task_session import build_retry_rebase_prompt


class TaskSessionRetryPromptTests(unittest.TestCase):
    def test_search_no_results_prompt_is_generic(self) -> None:
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tool_names=["web_fetch"],
            tools=[{"name": "web_fetch", "parameters": {}}],
            tool_enabled=True,
        )

        prompt = build_retry_rebase_prompt(request, reason="search_no_results")

        self.assertIn("last search tool returned no results", prompt)
        self.assertNotIn("WebSearch", prompt)


if __name__ == "__main__":
    unittest.main()
