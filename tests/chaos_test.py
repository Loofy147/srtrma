import unittest
import torch
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.chaos import PoisonedAPI, ChaosGenerator

class TestSRTRChaos(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.input_dim = 10
        self.hidden_dim = 16
        self.engine = SRTREngine(self.input_dim, self.hidden_dim, self.hidden_dim)
        self.chaos = ChaosGenerator(self.engine)

    def test_semantic_drift_regeneration(self):
        """
        Verify that Layer 1 catches structural anomalies and triggers hot-swap.
        """
        input_data = torch.randn(1, 5, self.input_dim)
        poisoned_input = self.chaos.inject_semantic_drift(input_data, intensity=20.0)

        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        # Ensure threshold is low enough to trigger
        self.engine.regeneration.threshold = 0.1

        new_state, result = self.engine.run_cycle(poisoned_input, adj_matrix, edge_weights, 1.0)

        # Verify adapter was hot-swapped
        self.assertIsNotNone(self.engine.layer3.adapter)
        self.assertTrue(result["success"])

    def test_api_failure_recovery(self):
        """
        Verify that Layer 3 execution failure triggers regeneration and recovery.
        """
        input_data = torch.randn(1, 5, self.input_dim)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        # Setup Poisoned API that always fails once
        api = PoisonedAPI(failure_rate=1.0)

        # Run cycle - first attempt fails, triggers regen, second attempt (in engine.run_cycle) succeeds if adapter is robust
        # Actually our mock PoisonedAPI will keep failing if failure_rate is 1.0.
        # Let's mock a recovery by reducing failure rate after first call.

        class RecoveringAPI(PoisonedAPI):
            def call(self, payload):
                res = super().call(payload)
                self.failure_rate = 0.0 # Recover after first hit
                return res

        api = RecoveringAPI(failure_rate=1.0)

        new_state, result = self.engine.run_cycle(input_data, adj_matrix, edge_weights, 1.0, api_client=api)

        self.assertTrue(result["success"])
        self.assertIsNotNone(self.engine.layer3.adapter)

if __name__ == "__main__":
    unittest.main()
