import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.utils.templates import AdapterSynthesis

class TestAdapterSynthesis(unittest.TestCase):
    def test_schema_mapping_synthesis(self):
        adapter = AdapterSynthesis.synthesize("anomaly: missing_field detected in schema")
        result = adapter(100)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["target_state"], 100)
        self.assertEqual(result["status"], "adapted")

    def test_scaling_compensation_synthesis(self):
        adapter = AdapterSynthesis.synthesize("high curvature drift anomaly")
        result = adapter(10.0)
        self.assertAlmostEqual(result, 11.0, places=5)

    def test_rest_auth_synthesis(self):
        adapter = AdapterSynthesis.synthesize("API Error: 401 Unauthorized token missing")
        result = adapter(50)
        self.assertIsInstance(result, dict)
        self.assertIn("headers", result)
        self.assertEqual(result["headers"]["Authorization"], "Bearer SRTR-AUTH-LIVE-ALPHA-01")

    def test_pagination_synthesis(self):
        adapter = AdapterSynthesis.synthesize("REST Error: next_page_token field missing for pagination")
        result = adapter(1)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["pagination"]["enabled"])
        self.assertEqual(result["pagination"]["next_key"], "next_page_token")

    def test_websocket_stream_synthesis(self):
        adapter = AdapterSynthesis.synthesize("Stream Failure: websocket connection lost")
        result = adapter("btc_usdt")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["protocol"], "ws")
        self.assertEqual(result["stream_filter"], "btc_usdt")

    def test_webhook_listener_synthesis(self):
        adapter = AdapterSynthesis.synthesize("Metadata: external webhook hook missing")
        result = adapter("alpha_v1")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["mode"], "listener")
        self.assertEqual(result["endpoint"], "/webhooks/alpha")

    def test_identity_fallback(self):
        adapter = AdapterSynthesis.synthesize("standard operational noise")
        result = adapter(123)
        self.assertEqual(result, 123)

if __name__ == "__main__":
    unittest.main()
