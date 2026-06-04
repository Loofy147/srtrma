import asyncio
import torch
import numpy as np
import sys
import os
import logging
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.streaming import LiveStreamAdapter
from srtr.utils.exchange_mock import MockExchangeClient

# Configuration for Vector Alpha: The Quantitative Regime-Switching Engine
BINANCE_BTC_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("SRTR.MainnetAnchor")

class AnchorControlLoop:
    """
    Production Control Loop for the Sovereign Mainnet Anchor.
    Coordinates real-time perception and idempotent execution.
    """
    def __init__(self, iterations=100, dry_run=True):
        self.iterations = iterations
        self.dry_run = dry_run

        # Initialize Engine with Vector Alpha dimensions
        self.engine = SRTREngine(input_dim=1, hidden_dim=16, node_dim=16)

        # Production sensory cortical setup
        self.ingress = LiveStreamAdapter(uri=BINANCE_BTC_WS)

        # Egress setup (using Mock for safety during dry run)
        self.egress = MockExchangeClient(initial_balance=1000000.0)

        # Internal State tracking
        self.price_history = []
        self.current_state = 0.0

    async def run(self):
        logger.info("⚔️  Sovereign Mainnet Anchor: Initiating Vector Alpha Deployment")
        logger.info(f"Target: {BINANCE_BTC_WS} | Dry Run: {self.dry_run}")

        # 1. Calibrate Initial State
        logger.info("Calibrating structural perception thresholds...")
        self.engine.regeneration.threshold = 300000000.0 # Calibrated in Phase 1

        # 2. Ingress Loop
        count = 0
        async for telemetry in self.ingress.stream_telemetry():
            if count >= self.iterations:
                break

            price = telemetry["value"]
            self.price_history.append(price)
            if len(self.price_history) > 100:
                self.price_history.pop(0)

            # --- Structural perception via Layer 1 (TGP) ---
            # ingest_stream_chunk handles normalization and history windowing
            input_data = self.engine.layer1.ingest_stream_chunk(self.price_history)

            # Simple fully connected graph for Vector Alpha local topology
            seq_len = input_data.shape[1]
            adj_matrix = torch.ones(1, seq_len, seq_len)
            edge_weights = torch.rand(1, seq_len, seq_len)

            # --- Engine Cycle Execution ---
            # run_cycle handles Layer 2 Regime Detection and Layer 3 DCE
            # It also triggers Self-Regeneration and State Parity Auditing internally
            new_state, result = await self.engine.run_cycle(
                input_data,
                adj_matrix,
                edge_weights,
                price if self.current_state == 0.0 else self.current_state,
                api_client=self.egress
            )

            self.current_state = new_state

            # --- Quantitative Strategy Egress ---
            diff = new_state - price
            trade_payload = None

            # Mean Reversion Logic
            if diff > 0.01:
                trade_payload = {"action": "buy", "amount": 0.01, "price": price}
            elif diff < -0.01:
                trade_payload = {"action": "sell", "amount": 0.01, "price": price}

            if trade_payload and not self.dry_run:
                # Idempotent execution via DCE
                trade_result = await self.egress.call(trade_payload)
                if trade_result.get("status") != 200:
                    logger.warning(f"Live Execution Failed: {trade_result.get('error')}")

            # Monitoring
            curvature = self.engine.telemetry.metrics["semantic_curvature"][-1]
            logger.info(f"Seq {count}: Price={price:.2f} | Curvature={curvature:.2e} | Balance={self.egress.balance:.2f}")

            count += 1

        logger.info("🏁 Mainnet Anchor Sequence Terminated.")
        logger.info(f"Final Statistics: Net Balance={self.egress.balance:.2f} | Idempotency Keys Cached={len(self.engine.layer3.executed_payloads)}")

if __name__ == "__main__":
    # In a real environment, we'd pull dry_run from env vars
    anchor = AnchorControlLoop(iterations=20, dry_run=True)
    try:
        asyncio.run(anchor.run())
    except KeyboardInterrupt:
        logger.info("Mainnet Anchor Aborted by User.")
