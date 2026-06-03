import unittest
import sys
import os
import logging
import asyncio

# Setup path and logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("SRTR").setLevel(logging.ERROR)

from srtr.layer3_dce.dce import DeterministicConstrainedExecution
from srtr.utils.regeneration import SelfRegenerationLoop

class TestStateParityAudit(unittest.TestCase):
    def setUp(self):
        self.dce = DeterministicConstrainedExecution(0.1, 1.0, 0.2)
        self.regeneration = SelfRegenerationLoop()

    def test_hot_swap_parity_success(self):
        """
        Tests that hot-swapping an adapter preserves state parity.
        """
        # Mock some active state
        self.dce.active_orders = {"order_1": "pending", "order_2": "executed"}

        # New adapter (identity)
        def new_adapter(x): return x

        success = self.regeneration.hot_swap_adapter(self.dce, new_adapter)

        self.assertTrue(success)
        self.assertEqual(self.dce.active_orders["order_1"], "pending")
        self.assertEqual(self.dce.adapter, new_adapter)

    def test_hot_swap_parity_failure_and_rollback(self):
        """
        Tests that a parity failure triggers a rollback.
        """
        # We simulate a parity failure by monkey-patching get_state_snapshot
        # to return a different state mid-swap (this is a bit artificial but tests the logic)

        original_snapshot_method = self.dce.get_state_snapshot

        call_count = 0
        def corrupt_snapshot():
            nonlocal call_count
            call_count += 1
            snap = original_snapshot_method()
            if call_count == 2: # Post-swap snapshot
                snap["theta_base"] = 999.0 # Corruption!
            return snap

        self.dce.get_state_snapshot = corrupt_snapshot

        def new_adapter(x): return x

        success = self.regeneration.hot_swap_adapter(self.dce, new_adapter)

        self.assertFalse(success)
        self.assertIsNone(self.dce.adapter) # Should have rolled back to None
        self.assertNotEqual(self.dce.theta_base, 999.0) # Logic check

    def test_cryptographic_integrity(self):
        """
        Verifies that even small state changes are detected as parity failures.
        """
        from srtr.utils.audit import StateAuditor

        state1 = {"a": 1, "b": {"c": 3}}
        state2 = {"a": 1, "b": {"c": 3}}
        state3 = {"a": 1, "b": {"c": 4}}

        hash1 = StateAuditor.compute_state_hash(state1)
        hash2 = StateAuditor.compute_state_hash(state2)
        hash3 = StateAuditor.compute_state_hash(state3)

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)

if __name__ == "__main__":
    unittest.main()
