import unittest
import asyncio
import sys
import os
import logging
import json
from aiohttp import web

# Setup path and logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("SRTR").setLevel(logging.ERROR)

from srtr.utils.streaming import LiveStreamAdapter

class TestStreamingIngress(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    async def _run_mock_ws_server(self, port=8080):
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)

            # Send mock financial data
            await ws.send_json({"p": "50000.0", "T": 123456789})
            await ws.send_json({"p": "50005.0", "T": 123456790})

            await asyncio.sleep(0.1)
            await ws.close()
            return ws

        app = web.Application()
        app.add_routes([web.get('/', websocket_handler)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        return runner

    def test_stream_mapping(self):
        """
        Verifies that LiveStreamAdapter correctly maps raw WS data.
        """
        async def run():
            runner = await self._run_mock_ws_server(port=8081)
            adapter = LiveStreamAdapter(uri="http://localhost:8081")

            telemetry_list = []
            async for telemetry in adapter.stream_telemetry():
                telemetry_list.append(telemetry)
                if len(telemetry_list) >= 2:
                    adapter.stop()

            self.assertEqual(len(telemetry_list), 2)
            self.assertEqual(telemetry_list[0]["value"], 50000.0)
            self.assertEqual(telemetry_list[1]["value"], 50005.0)

            await runner.cleanup()

        self.loop.run_until_complete(run())

    def test_stream_resilience(self):
        """
        Verifies that the adapter handles connection failures gracefully.
        """
        async def run():
            # Connect to non-existent server
            adapter = LiveStreamAdapter(uri="http://localhost:9999")

            telemetry_list = []
            async for telemetry in adapter.stream_telemetry():
                telemetry_list.append(telemetry)

            self.assertEqual(len(telemetry_list), 0)
            self.assertFalse(adapter.is_running)

        self.loop.run_until_complete(run())

if __name__ == "__main__":
    unittest.main()
