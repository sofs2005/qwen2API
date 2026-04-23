import unittest

from backend.toolcore.stream_sieve import ToolStreamSieve


class ToolStreamSieveTests(unittest.TestCase):
    def test_complete_tool_block_extracted_from_streamed_chunks(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk('##TOOL_CALL##\n')
        second = sieve.process_chunk('{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##')

        self.assertEqual(first, [])
        tool_events = [event for event in second if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["calls"][0]["name"], "Read")

    def test_partial_tool_block_is_held_until_complete(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk('##TOOL_CALL##\n{"name": "Read"')

        self.assertEqual(events, [])

        final_events = sieve.process_chunk(', "input": {"file_path": "README.md"}}\n##END_CALL##')
        tool_events = [event for event in final_events if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)

    def test_fenced_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        content = '```json\n##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##\n```'
        events = sieve.process_chunk(content)
        events.extend(sieve.flush())

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, content)

    def test_incomplete_tool_block_flushes_as_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        sieve.process_chunk('##TOOL_CALL##\n{"name": "Read"')
        events = sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertIn('##TOOL_CALL##', text)


if __name__ == "__main__":
    unittest.main()
