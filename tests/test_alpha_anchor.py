import unittest
import asyncio
import sys
import os
import logging
import torch
import numpy as np

# Setup path and logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("SRTR").setLevel(logging.ERROR)

from srtr.engine import SRTREngine
from srtr.utils.exchange_mock import MockExchangeClient

class TestAlphaAnchor(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.input_dim = 1
        self.hidden_dim = 8
        self.engine = SRTREngine(self.input_dim, self.hidden_dim, self.hidden_dim)
        self.exchange = MockExchangeClient(initial_balance=100000.0)
        self.exchange.error_rate = 0.0 # Disable random errors for stability
        self.exchange.rate_limit_delay = 0.0 # Disable rate limit for tests

    def tearDown(self):
        self.loop.close()

    def test_anchor_cycle_resilience(self):
        """
        Verifies that the anchor cycle maintains state parity during a forced hot-swap.
        """
        async def run():
            # Force anomaly threshold low to trigger hot-swap
            self.engine.regeneration.threshold = -1.0

            # Simulated order state
            self.engine.layer3.active_orders = {"BTC_001": "LIMIT_BUY"}

            # Input data (Vector Alpha style)
            input_data = torch.randn(1, 5, 1)
            adj_matrix = torch.ones(1, 5, 5)
            edge_weights = torch.rand(1, 5, 5)

            # Execute cycle
            new_state, result = await self.engine.run_cycle(
                input_data,
                adj_matrix,
                edge_weights,
                50000.0,
                api_client=self.exchange
            )

            # Verify Parity: Active orders must survive the swap
            self.assertTrue(result["success"])
            self.assertIn("BTC_001", self.engine.layer3.active_orders)
            self.assertEqual(self.engine.layer3.active_orders["BTC_001"], "LIMIT_BUY")

        self.loop.run_until_complete(run())

    def test_idempotency_protection(self):
        """
        Verifies that the anchor prevents duplicate trade dispatches for the same state.
        """
        async def run():
            input_data = torch.zeros(1, 5, 1)
            adj_matrix = torch.ones(1, 5, 5)
            edge_weights = torch.zeros(1, 5, 5)

            # First cycle
            state1, res1 = await self.engine.run_cycle(input_data, adj_matrix, edge_weights, 50000.0, api_client=self.exchange)
            count1 = self.exchange.balance # Initial balance check

            # Second cycle with identical input - should be blocked by idempotency in DCE
            # Note: OU process has noise, but if we mock/force result it works.
            # In DCE, the payload hash is checked.

            # Let's manually trigger duplicate payload
            payload = self.engine.layer3.prepare_payload(state1)
            res2 = await self.engine.layer3.execute_api_payload(payload, [], api_client=self.exchange)

            self.assertTrue(res2["success"])
            self.assertEqual(res2.get("reason"), "idempotency_hit")
            self.assertEqual(self.exchange.balance, count1) # Balance should not have changed

        self.loop.run_until_complete(run())

if __name__ == "__main__":
    unittest.main()
