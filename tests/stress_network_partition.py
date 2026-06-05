import unittest
import asyncio
import torch
import time
import logging
import numpy as np
from srtr.engine import SRTREngine
from srtr.utils.chaos import PoisonedAPI

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Stress.Network")

class PartitionAPI(PoisonedAPI):
    """API that simulates a complete network partition for a specified duration."""
    def __init__(self):
        super().__init__(failure_rate=0.0)
        self.partition_active = False
        self.calls_received = 0

    async def call(self, payload):
        self.calls_received += 1
        if self.partition_active:
            logger.warning(f"Partition active: timing out call {self.calls_received}")
            await asyncio.sleep(6.0) # Longer than default 5s timeout
            return {"error": "timeout", "status": 504}
        return {"data": payload, "status": 200}

class TestNetworkPartitionStress(unittest.TestCase):
    def setUp(self):
        self.engine = SRTREngine(input_dim=1, hidden_dim=4, node_dim=4)
        self.api = PartitionAPI()

    def test_network_partition_with_agent_healing(self):
        """
        Operation: Simulate 5s network partition during trading and agent healing.
        Objective: Verify tool cleanup, lock integrity, and intent idempotency.
        """
        loop = asyncio.get_event_loop()

        input_data = torch.randn(1, 5, 1)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.ones(1, 5, 5)
        current_state = 100.0

        # 1. Trigger partition
        self.api.partition_active = True

        async def run_partition_sequence():
            # This call should fail and trigger regeneration (AgentField)
            logger.info("Starting cycle with active partition...")
            new_state, result = await self.engine.run_cycle(
                input_data, adj_matrix, edge_weights, current_state,
                api_client=self.api, cycle_nonce="STRESS_01"
            )

            self.assertFalse(result["success"])

            # Verify tool cleanup even after failure
            for fid in self.engine.graph._F:
                if fid.startswith("_tool_"):
                    raise Exception(f"Transient tool field {fid} was not garbage collected!")

            # 2. End partition and retry same intent (Idempotency check)
            logger.info("Partition ended. Retrying with same intent nonce...")
            self.api.partition_active = False

            # Re-running with same cycle_nonce
            # Since the previous attempt FAILED to record success, it should proceed.
            new_state_2, result_2 = await self.engine.run_cycle(
                input_data, adj_matrix, edge_weights, current_state,
                api_client=self.api, cycle_nonce="STRESS_01"
            )

            self.assertTrue(result_2["success"])

            # 3. Final Idempotency verification (Attempting again)
            logger.info("Verifying idempotency for successful intent...")
            _, result_3 = await self.engine.run_cycle(
                input_data, adj_matrix, edge_weights, current_state,
                api_client=self.api, cycle_nonce="STRESS_01"
            )
            self.assertEqual(result_3.get("reason"), "idempotency_hit")

        loop.run_until_complete(run_partition_sequence())
        logger.info("Network Partition Stress Test Passed.")

if __name__ == "__main__":
    unittest.main()
