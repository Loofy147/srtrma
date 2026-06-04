import unittest
import sys
import os
import logging
import asyncio
import hmac
import hashlib
import json
import time

# Setup path and logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("SRTR").setLevel(logging.ERROR)

from srtr.layer3_dce.dce import DeterministicConstrainedExecution
from srtr.utils.templates import AdapterSynthesis

class MockAPIClient:
    def __init__(self):
        self.call_count = 0
        self.last_payload = None

    async def call(self, payload):
        self.call_count += 1
        self.last_payload = payload
        return {"status": 200, "success": True}

class TestEgressSecurity(unittest.TestCase):
    def setUp(self):
        self.dce = DeterministicConstrainedExecution(0.1, 1.0, 0.2)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_hmac_signature_verification(self):
        """
        Verifies that the HMAC signed adapter produces valid signatures from ENV.
        """
        os.environ["SRTR_API_KEY"] = "TEST_KEY"
        os.environ["SRTR_API_SECRET"] = "TEST_SECRET"

        adapter = AdapterSynthesis.synthesize("Egress Failure: signature required")
        self.dce.adapter = adapter

        payload = self.dce.prepare_payload(50000.0)

        headers = payload["headers"]
        data = payload["payload"]

        self.assertIn("X-SRTR-SIGNATURE", headers)
        self.assertEqual(headers["X-SRTR-API-KEY"], "TEST_KEY")

        # Verify signature manually
        expected_sig = hmac.new(
            b"TEST_SECRET",
            json.dumps(data, sort_keys=True).encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        self.assertEqual(headers["X-SRTR-SIGNATURE"], expected_sig)

    def test_intent_based_idempotency(self):
        """
        Verifies that idempotency holds even if transmission payload changes (e.g. timestamp).
        """
        async def run():
            client = MockAPIClient()

            # Simulated adapter that adds a dynamic timestamp (like hmac_signed_adapter)
            def adaptive_payload(state):
                return {"payload": {"state": state}, "timestamp": time.time()}

            self.dce.adapter = adaptive_payload

            # First dispatch
            p1 = self.dce.prepare_payload(1.0)
            res1 = await self.dce.execute_api_payload(p1, [], api_client=client, intent_nonce="cycle_1")
            self.assertTrue(res1["success"])
            self.assertEqual(client.call_count, 1)

            # Second dispatch with SAME intent but NEW transmission payload (regenerated)
            await asyncio.sleep(0.01)
            p2 = self.dce.prepare_payload(1.0)
            self.assertNotEqual(p1["timestamp"], p2["timestamp"]) # Verified payload changed

            res2 = await self.dce.execute_api_payload(p2, [], api_client=client, intent_nonce="cycle_1")
            self.assertTrue(res2["success"])
            self.assertEqual(res2.get("reason"), "idempotency_hit")
            self.assertEqual(client.call_count, 1) # Blocked!

        self.loop.run_until_complete(run())

    def test_nested_boundary_check(self):
        """
        Verifies that constraints correctly find the state in nested payloads.
        """
        async def run():
            # Constraint: state must be < 100
            constraints = [lambda x: x < 100]

            # Nested payload
            payload = {"payload": {"state": 150.0}, "meta": "data"}

            res = await self.dce.execute_api_payload(payload, constraints)
            self.assertFalse(res["success"])
            self.assertEqual(res["reason"], "constraint_violation")

            # Valid nested payload
            payload_ok = {"payload": {"state": 50.0}}
            res_ok = await self.dce.execute_api_payload(payload_ok, constraints)
            self.assertTrue(res_ok["success"])

        self.loop.run_until_complete(run())

if __name__ == "__main__":
    unittest.main()
