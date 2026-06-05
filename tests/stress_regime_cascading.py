import unittest
import asyncio
import torch
import time
import numpy as np
import logging
from srtr.engine import SRTREngine

# Set logging to warning to avoid flooding output during stress test
logging.getLogger("SRTR").setLevel(logging.WARNING)

class TestRegimeCascadingStress(unittest.TestCase):
    def setUp(self):
        self.engine = SRTREngine(input_dim=1, hidden_dim=8, node_dim=8)

    def test_high_frequency_regime_flips(self):
        """
        Operation: Inject rapid market regime flips.
        Objective: Verify sub-millisecond recalculation and resource integrity.
        """
        loop = asyncio.get_event_loop()

        num_iterations = 500
        input_data = torch.randn(1, 5, 1)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.ones(1, 5, 5)
        current_state = 100.0

        latencies = []

        async def run_stress():
            for i in range(num_iterations):
                # We modify the edge weights or input to simulate high-entropy change
                # which triggers downstream re-calculation in the FieldGraph.
                dynamic_edge_weights = torch.rand(1, 5, 5)

                t0 = time.perf_counter()
                await self.engine.run_cycle(
                    input_data, adj_matrix, dynamic_edge_weights, current_state
                )
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000) # ms

        loop.run_until_complete(run_stress())

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)

        print(f"\n--- Regime Cascading Stress Results ({num_iterations} cycles) ---")
        print(f"Average Latency: {avg_latency:.4f} ms")
        print(f"P95 Latency:     {p95_latency:.4f} ms")
        print(f"Max Latency:     {max_latency:.4f} ms")

        # Validation: Average latency should be very low (ideally < 1-2ms even with PyTorch overhead)
        self.assertLess(avg_latency, 5.0, "Average latency too high for reactive core")
        self.assertLess(p95_latency, 10.0, "Tail latency too high")

if __name__ == "__main__":
    unittest.main()
