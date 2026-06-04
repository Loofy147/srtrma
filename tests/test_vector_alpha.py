import unittest
import sys
import os
import logging
import torch
import asyncio
from aiohttp import web

# Setup path and logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("SRTR").setLevel(logging.ERROR)

from srtr.engine import SRTREngine
from srtr.layer1_tgp.splicer import DepthSplicer
from srtr.utils.exchange_client import LiveExchangeClient
from srtr.utils.templates import AdapterSynthesis

class TestVectorAlpha(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Vector Alpha Config
        self.engine = SRTREngine(input_dim=2, hidden_dim=8, node_dim=8)
        self.splicer = DepthSplicer(depth_levels=5)
        self.client = LiveExchangeClient(base_url="http://localhost:8082")

    def tearDown(self):
        self.loop.close()

    async def _run_mock_api_server(self, port=8082):
        async def handle_order(request):
            headers = request.headers
            # Check for Vector Alpha security headers
            if "X-SRTR-SIGNATURE" not in headers:
                return web.json_response({"status": 401, "msg": "Unauthorized"}, status=401)

            return web.json_response({"status": "FILLED", "orderId": 12345}, status=200)

        app = web.Application()
        app.add_routes([web.post('/api/v3/order', handle_order)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        return runner

    def test_full_alpha_cycle_with_signed_egress(self):
        """
        Verifies a full SRTR cycle with Depth Splicing and HMAC signed egress.
        """
        async def run():
            runner = await self._run_mock_api_server(port=8082)

            # 1. Perception: Splice simulated order book flow
            trades = [[50000.0, 1.0], [50005.0, 0.5], [49995.0, 2.0]]
            input_data = self.splicer.splice_order_flow(trades, window_size=5)

            # 2. Secure Egress: Synthesize signed adapter
            adapter = AdapterSynthesis.synthesize("Egress: signature required for Vector Alpha")
            self.engine.layer3.adapter = adapter

            # 3. Execution: Cycle with LiveExchangeClient
            adj = torch.ones(1, 5, 5)
            weights = torch.rand(1, 5, 5)

            new_state, result = await self.engine.run_cycle(
                input_data,
                adj,
                weights,
                50000.0,
                api_client=self.client
            )

            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(len(self.engine.layer3.executed_intents), 1)

            await runner.cleanup()

        self.loop.run_until_complete(run())

if __name__ == "__main__":
    unittest.main()
