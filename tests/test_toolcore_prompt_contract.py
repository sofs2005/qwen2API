import unittest

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.toolcore.prompt_contract import build_tool_instruction_block, normalize_prompt_tools, render_history_tool_call


class PromptContractTests(unittest.TestCase):
    def test_same_tools_produce_same_contract_after_normalization(self) -> None:
        chat_tools = [{"type": "function", "function": {"name": "Read", "description": "Read file content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}]
        responses_tools = [{"name": "Read", "description": "Read file content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]

        chat_contract = build_tool_instruction_block(normalize_prompt_tools(chat_tools), OPENCLAW_OPENAI_PROFILE)
        responses_contract = build_tool_instruction_block(normalize_prompt_tools(responses_tools), OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(chat_contract, responses_contract)

    def test_required_tool_choice_adds_forced_tool_constraint(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "Read", "description": "Read file", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
            tool_choice_mode="required",
            required_tool_name="Read",
        )

        self.assertIn('MUST call the exact tool "Read"', contract)
        self.assertIn("##TOOL_CALL##", contract)

    def test_none_tool_choice_suppresses_required_guidance(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "Read", "description": "Read file", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
            tool_choice_mode="none",
        )

        self.assertIn("do NOT call any tool", contract)
        self.assertNotIn("MUST include at least one tool call", contract)

    def test_history_tool_call_uses_canonical_wrapper_style(self) -> None:
        rendered = render_history_tool_call("Read", {"file_path": "README.md"}, CLAUDE_CODE_OPENAI_PROFILE)
        self.assertEqual(rendered, '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##')


if __name__ == "__main__":
    unittest.main()
