import unittest
import torch
import logging
import sys
import os

logging.getLogger("SRTR").setLevel(logging.ERROR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.chaos import PoisonedAPI

class TestSRTRChaos(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 16
        self.engine = SRTREngine(self.input_dim, self.hidden_dim, self.hidden_dim)

    def test_schema_drift_recovery(self):
        class RenamedFieldAPI(PoisonedAPI):
            def call(self, payload):
                if "target_state" not in payload:
                    return {"error": "missing_field", "status": 400}
                return {"data": payload, "status": 200}

        api = RenamedFieldAPI(failure_rate=0.0)
        input_data = torch.randn(1, 5, self.input_dim)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        _, result = self.engine.run_cycle(input_data, adj_matrix, edge_weights, 1.0, api_client=api)
        self.assertTrue(result["success"])

    def test_recovering_api(self):
        class RecoveringAPI(PoisonedAPI):
            def call(self, payload):
                res = super().call(payload)
                self.failure_rate = 0.0
                return res

        api = RecoveringAPI(failure_rate=1.0)
        input_data = torch.randn(1, 5, self.input_dim)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        _, result = self.engine.run_cycle(input_data, adj_matrix, edge_weights, 1.0, api_client=api)
        self.assertTrue(result["success"])

if __name__ == "__main__":
    unittest.main()
