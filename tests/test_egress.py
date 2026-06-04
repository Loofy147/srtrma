import unittest
import sys
import os
import logging
import asyncio
import hmac
import hashlib
import json

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
        Verifies that the HMAC signed adapter produces valid signatures.
        """
        adapter = AdapterSynthesis.synthesize("Egress Failure: signature required for live trading")
        self.dce.adapter = adapter

        payload = self.dce.prepare_payload(50000.0)

        headers = payload["headers"]
        data = payload["payload"]

        self.assertIn("X-SRTR-SIGNATURE", headers)
        self.assertEqual(headers["X-SRTR-API-KEY"], "SRTR-KEY-PROD-001")

        # Verify signature manually
        secret = "SRTR-SECRET-ALPHA-99"
        expected_sig = hmac.new(
            secret.encode('utf-8'),
            json.dumps(data, sort_keys=True).encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        self.assertEqual(headers["X-SRTR-SIGNATURE"], expected_sig)

    def test_egress_idempotency(self):
        """
        Verifies that duplicate payloads are blocked at the egress point.
        """
        async def run():
            client = MockAPIClient()
            # Simple identity adapter
            self.dce.adapter = lambda x: {"state": x, "nonce": "static_idempotency_test"}

            # First dispatch
            result1 = await self.dce.execute_api_payload({"state": 1.0, "nonce": "static"}, [], api_client=client)
            self.assertTrue(result1["success"])
            self.assertEqual(client.call_count, 1)

            # Second dispatch (duplicate)
            result2 = await self.dce.execute_api_payload({"state": 1.0, "nonce": "static"}, [], api_client=client)
            self.assertTrue(result2["success"])
            self.assertEqual(result2.get("reason"), "idempotency_hit")
            self.assertEqual(client.call_count, 1) # Should NOT have called API again

            # Third dispatch (different payload)
            result3 = await self.dce.execute_api_payload({"state": 2.0, "nonce": "static"}, [], api_client=client)
            self.assertTrue(result3["success"])
            self.assertEqual(client.call_count, 2)

        self.loop.run_until_complete(run())

if __name__ == "__main__":
    unittest.main()
