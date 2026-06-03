import unittest
import torch
import numpy as np
import sys
import os
import logging

logging.getLogger("SRTR").setLevel(logging.ERROR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine

class TestSRTREngine(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 16
        self.engine = SRTREngine(self.input_dim, self.hidden_dim, self.hidden_dim)

    def test_single_cycle(self):
        batch_size = 1
        seq_len = 5
        input_data = torch.randn(batch_size, seq_len, self.input_dim)
        adj_matrix = torch.ones(batch_size, seq_len, seq_len)
        edge_weights = torch.rand(batch_size, seq_len, seq_len)

        new_state, result = self.engine.run_cycle(input_data, adj_matrix, edge_weights, 1.0)
        self.assertTrue(result["success"])

    def test_hot_swap_effect(self):
        # Trigger scaling adapter via Layer 1 anomaly
        self.engine.regeneration.threshold = -1.0 # Force anomaly

        input_data = torch.randn(1, 5, self.input_dim)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        new_state, result = self.engine.run_cycle(input_data, adj_matrix, edge_weights, 1.0)

        # In the new implementation, compute_ou_process returns scalar, prepare_payload applies adapter
        payload = self.engine.layer3.prepare_payload(1.0)
        self.assertAlmostEqual(payload["state"], 1.1, places=5)

if __name__ == "__main__":
    unittest.main()
