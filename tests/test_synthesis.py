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
        self.assertEqual(result["payload"]["state"], 50)

    def test_identity_fallback(self):
        adapter = AdapterSynthesis.synthesize("standard operational noise")
        result = adapter(123)
        self.assertEqual(result, 123)

if __name__ == "__main__":
    unittest.main()
