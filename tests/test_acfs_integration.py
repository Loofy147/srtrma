import unittest
import asyncio
import torch
import numpy as np
from srtr.engine import SRTREngine
from srtr.utils.acfs import FieldGraph, AgentField

class TestACFSIntegration(unittest.TestCase):
    def setUp(self):
        # input_dim=1, hidden_dim=4, node_dim=4
        self.engine = SRTREngine(input_dim=1, hidden_dim=4, node_dim=4)

    def test_engine_cycle_reactive(self):
        """Verify that the engine runs correctly with the reactive FieldGraph."""
        # Layer 1 expects (batch, seq_len, input_dim)
        # Layer 2 expects nodes (batch, num_nodes, node_dim)
        # In SRTREngine, Layer 2 'nodes' is the output 'psi' from Layer 1.
        # Layer 1 'psi' shape is (batch, seq_len, hidden_dim)
        # So in Layer 2: num_nodes = seq_len, node_dim = hidden_dim.
        # But adj_matrix and edge_weights are (batch, num_nodes, num_nodes).

        seq_len = 5
        input_data = torch.randn(1, seq_len, 1)
        adj_matrix = torch.ones(1, seq_len, seq_len)
        edge_weights = torch.ones(1, seq_len, seq_len)
        current_state = 100.0

        loop = asyncio.get_event_loop()
        new_state, result = loop.run_until_complete(self.engine.run_cycle(
            input_data, adj_matrix, edge_weights, current_state
        ))

        self.assertIsInstance(new_state, (float, np.float32, np.float64, np.ndarray))
        self.assertTrue(result["success"])

    def test_agent_field_execution(self):
        """Verify AgentField with mock LLM and tool spawning."""
        graph = FieldGraph("AgentTest")
        graph.source("_messages", [{"role": "user", "content": "How is the weather?"}])

        def mock_tool_fn(query):
            return f"Weather in {query} is sunny."

        tools = [{"name": "get_weather", "fn": mock_tool_fn, "description": "Get weather"}]

        graph.agent("assistant", tools=tools, mock=True)

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(graph.get("assistant"))

        self.assertIn("text", result)
        self.assertIn("[MOCK]", result["text"])
        # Check that tool fields were cleaned up
        for fid in graph._F:
            self.assertFalse(fid.startswith("_tool_"))

    def test_circular_dependency(self):
        """T2-FIX: Verify circular dependency detection."""
        graph = FieldGraph("Circular")
        graph.derive("A", lambda B: B, ["B"])
        graph.derive("B", lambda A: A, ["A"])

        loop = asyncio.get_event_loop()
        with self.assertRaises(RecursionError):
            loop.run_until_complete(graph.get("A"))

    def test_concurrency_coalescing(self):
        """T5-FIX: Verify concurrent readers coalesce."""
        graph = FieldGraph("Coalesce")
        calls = 0
        async def slow_fn():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.1)
            return "done"

        graph.derive("slow", slow_fn, [])

        loop = asyncio.get_event_loop()
        async def run():
            return await asyncio.gather(graph.get("slow"), graph.get("slow"))

        results = loop.run_until_complete(run())
        self.assertEqual(results, ["done", "done"])
        self.assertEqual(calls, 1)

if __name__ == "__main__":
    unittest.main()
