import unittest
import pandas as pd
from agent import DataAnalysisAgent


class FakeAgent(DataAnalysisAgent):
    """Replace the HTTP/model call so tests do not need Ollama running."""
    def __init__(self, responses):
        super().__init__(model="fake-model", max_steps=3)
        self.responses = list(responses)
        self.seen_messages = []

    def _chat(self, messages):
        self.seen_messages.append(messages.copy())
        if not self.responses:
            raise AssertionError("Test did not provide enough fake responses")
        return self.responses.pop(0)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Region": ["North", "South", "North"],
            "Revenue": [100.0, 200.0, 50.0],
        })

    def test_tool_call_then_final_answer(self):
        agent = FakeAgent([
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "grouped_aggregation",
                "arguments": {"group_by": "Region", "metric": "Revenue", "aggregation": "sum"}
            }}]},
            {"role": "assistant", "content": "South has the highest revenue at 200."},
        ])
        result = agent.ask(self.df, "Which region generated the most revenue?")
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertEqual(result["tool_trace"][0]["status"], "success")
        self.assertIn("South", result["answer"])
        # The model sees the result returned by the Python tool on its second call.
        second_call_messages = agent.seen_messages[1]
        self.assertTrue(any(m.get("role") == "tool" and '"value": 200.0' in m.get("content", "") for m in second_call_messages))

    def test_unknown_tool_is_not_executed(self):
        agent = FakeAgent([
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "execute_python", "arguments": {"code": "print('unsafe')"}
            }}]},
            {"role": "assistant", "content": "That tool is not available."},
        ])
        result = agent.ask(self.df, "Run some code")
        self.assertEqual(result["tool_trace"][0]["status"], "error")
        self.assertIn("not allowed", result["tool_trace"][0]["result"]["error"])
        self.assertIn("That tool", result["answer"])


if __name__ == "__main__":
    unittest.main()
