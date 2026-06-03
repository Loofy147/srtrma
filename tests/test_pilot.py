import unittest
import asyncio
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.exchange_mock import MockExchangeClient

# Helper for async tests
def async_test(f):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestAlphaPilot(unittest.TestCase):
    def setUp(self):
        self.engine = SRTREngine(1, 8, 8)
        self.exchange = MockExchangeClient(initial_balance=1000.0)

    @async_test
    async def test_full_trading_loop(self):
        # Mock high-frequency input
        input_data = torch.randn(1, 5, 1)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        # 1. Run cycle with exchange client
        new_state, result = await self.engine.run_cycle(
            input_data,
            adj_matrix,
            edge_weights,
            100.0,
            api_client=self.exchange
        )

        self.assertTrue(result["success"])

        # Wait for rate limit
        await asyncio.sleep(0.25)

        # 2. Execute a buy order directly to verify exchange client
        buy_payload = {"action": "buy", "amount": 1, "price": 100.0}
        trade_result = await self.exchange.call(buy_payload)

        self.assertEqual(trade_result["status"], 200)
        self.assertEqual(self.exchange.balance, 900.0)
        self.assertEqual(self.exchange.position, 1.0)

    @async_test
    async def test_exchange_rate_limit(self):
        # Force a quick second call to trigger rate limit
        payload = {"action": "buy", "amount": 1, "price": 10.0}
        await self.exchange.call(payload)

        # Second call immediately after should fail (or return 429)
        trade_result = await self.exchange.call(payload)
        self.assertEqual(trade_result["status"], 429)

if __name__ == "__main__":
    unittest.main()
