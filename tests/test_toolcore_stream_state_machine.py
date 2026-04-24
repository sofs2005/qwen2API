import unittest

from backend.toolcore.stream_state_machine import ToolStreamStateMachine


class ToolCoreStreamStateMachineTests(unittest.TestCase):
    def test_partial_tool_wrapper_does_not_leak_before_classification(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        events = machine.process_text_delta('##TOOL_CALL##\n{"name": "Read"')

        self.assertEqual(events, [])

    def test_cross_chunk_marker_is_held_until_safe(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        events1 = machine.process_text_delta("##TOOL_C")
        events2 = machine.process_text_delta('ALL##\n{"name": "Read", "input": {"path": "README.md"}}\n##END_CALL##')

        self.assertEqual(events1, [])
        self.assertTrue(any(event.type == "tool_calls" for event in events2))

    def test_malformed_wrapper_is_suppressed_if_later_tool_call_wins(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        machine.process_text_delta('##TOOL_CALL##\n{"name": "exec", "input": {"command": "ls -la /tmp"')
        machine.process_tool_calls([{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}])
        events = machine.flush(final_tool_use=True)

        self.assertFalse(any(event.type == "content" and event.text and "##TOOL_CALL##" in event.text for event in events))

    def test_failed_attempt_output_is_isolated_from_later_success(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        machine.process_text_delta("Tool exec does not exists.")
        machine.reset_attempt()
        events = machine.process_tool_calls([{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}])

        self.assertTrue(any(event.type == "tool_calls" for event in events))
        flushed = machine.flush(final_tool_use=True)
        self.assertFalse(any(event.type == "content" and event.text and "Tool exec does not exists." in event.text for event in flushed))


if __name__ == "__main__":
    unittest.main()
