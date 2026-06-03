import unittest
import asyncio
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.utils.networking import LiveNetworkBridge
from srtr.layer3_dce.monads import ExecutionMonad

# Helper for async tests
def async_test(f):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class MockAPIClient:
    async def call(self, payload):
        await asyncio.sleep(0.1)
        return {"status": 200, "data": "mock_success"}

class TestNetworkingAndMonad(unittest.TestCase):
    def setUp(self):
        self.bridge = LiveNetworkBridge()
        self.bridge.set_simulation_params(jitter_range=(0.001, 0.01), drop_rate=0.0)

    @async_test
    async def test_live_bridge_async(self):
        # We test with a real endpoint if possible, but for reliability we can mock or use a very stable one.
        # Here we just verify it runs without crashing.
        result = await self.bridge.fetch_telemetry_async()
        if "error" in result:
             self.skipTest(f"Live network error: {result['error']}")

        self.assertIn("value", result)
        self.assertIsInstance(result["value"], float)

    @async_test
    async def test_async_monad_binding(self):
        client = MockAPIClient()
        payload = {"test": "data"}

        # Unit and Bind
        monad = ExecutionMonad.unit(payload).bind(client.call)

        # Since client.call is async, bind returns a coroutine
        self.assertTrue(asyncio.iscoroutine(monad))

        monad_result = await monad
        value, error, logs = monad_result.unwrap()

        self.assertIsNone(error)
        self.assertEqual(value["status"], 200)
        self.assertIn("Success (Async): call", logs)

    @async_test
    async def test_async_monad_timeout(self):
        async def slow_func(x):
            await asyncio.sleep(0.5)
            return x

        monad = await ExecutionMonad.unit("test").bind(slow_func, timeout=0.1)
        value, error, logs = monad.unwrap()

        self.assertIsNotNone(error)
        self.assertIn("timeout", error)

if __name__ == "__main__":
    unittest.main()
