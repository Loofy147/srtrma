import unittest
import torch
import numpy as np
import sys
import os
import logging

# Suppress expected warnings during tests
logging.getLogger("SRTR").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.regeneration import SelfRegenerationLoop

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

        current_state = 1.0
        new_state, success = self.engine.run_cycle(input_data, adj_matrix, edge_weights, current_state)

        self.assertIsInstance(new_state, (float, np.float32, np.float64))
        self.assertTrue(success)

    def test_hot_swap_effect(self):
        # Force an anomaly by lowering threshold
        self.engine.regeneration.threshold = -1.0 # Always trigger

        input_data = torch.randn(1, 5, self.input_dim)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        current_state = 1.0
        # Normal OU process would yield something close to 1.0
        # Adaptive adapter logic returns current_state * 1.05
        new_state, success = self.engine.run_cycle(input_data, adj_matrix, edge_weights, current_state)

        self.assertAlmostEqual(new_state, 1.05, places=5)
        self.assertIsNotNone(self.engine.layer3.adapter)

    def test_cohomology_drift(self):
        reg = SelfRegenerationLoop(threshold=0.5)
        # High constant covariant derivative leads to high mean drift
        high_cd = torch.ones(1, 5, 16) * 10.0
        self.assertTrue(reg.detect_anomaly(high_cd))

        # Zero derivative leads to zero drift
        zero_cd = torch.zeros(1, 5, 16)
        self.assertFalse(reg.detect_anomaly(zero_cd))

if __name__ == "__main__":
    unittest.main()
