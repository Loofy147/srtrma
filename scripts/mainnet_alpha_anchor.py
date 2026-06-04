import asyncio
import torch
import numpy as np
import sys
import os
import logging
import time
import uuid

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.streaming import LiveStreamAdapter
from srtr.utils.exchange_client import LiveExchangeClient
from srtr.utils.exchange_mock import MockExchangeClient
from srtr.layer1_tgp.splicer import DepthSplicer

# Configuration for Vector Alpha: The Quantitative Regime-Switching Engine
BINANCE_BTC_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("SRTR.MainnetAnchor")

class AnchorControlLoop:
    """
    Production Control Loop for the Sovereign Mainnet Anchor.
    Coordinates real-time depth-enriched perception and intent-based idempotency.
    """
    def __init__(self, iterations=100, dry_run=True):
        self.iterations = iterations
        self.dry_run = dry_run

        # Initialize Engine with Vector Alpha dimensions
        self.engine = SRTREngine(input_dim=2, hidden_dim=32, node_dim=32)

        # Production sensory cortical setup
        self.ingress = LiveStreamAdapter(uri=BINANCE_BTC_WS)
        self.splicer = DepthSplicer(depth_levels=10)

        # Egress setup
        if self.dry_run:
            self.egress = MockExchangeClient(initial_balance=1000000.0)
        else:
            self.egress = LiveExchangeClient(base_url="https://api.binance.com")

        # Internal State tracking
        self.trade_history = []
        self.current_state = 0.0

    async def run(self):
        logger.info("⚔️  Sovereign Mainnet Anchor: Initiating Vector Alpha Deployment")
        logger.info(f"Ingress: {BINANCE_BTC_WS} | Egress: {type(self.egress).__name__} | Dry Run: {self.dry_run}")

        # 1. Calibrate Initial State
        self.engine.regeneration.threshold = 500000000.0

        # 2. Ingress Loop
        count = 0
        async for telemetry in self.ingress.stream_telemetry():
            if count >= self.iterations:
                break

            # Use unique cycle ID for intent-based idempotency
            cycle_id = f"cycle_{uuid.uuid4().hex[:8]}"

            price = telemetry["value"]
            quantity = telemetry["raw"].get("q", "0.0")
            try:
                qty = float(quantity)
            except:
                qty = 0.0

            self.trade_history.append([price, qty])
            if len(self.trade_history) > 100:
                self.trade_history.pop(0)

            # --- Structural perception via Layer 1 ---
            input_data = self.splicer.splice_order_flow(self.trade_history, window_size=20)
            seq_len = input_data.shape[1]
            adj_matrix = torch.ones(1, seq_len, seq_len)
            edge_weights = torch.rand(1, seq_len, seq_len)

            # --- Engine Cycle Execution ---
            # We pass cycle_id as a nonce to ensure distinct price points in different cycles
            # are treated as distinct intents.
            new_state, result = await self.engine.run_cycle(
                input_data,
                adj_matrix,
                edge_weights,
                price if self.current_state == 0.0 else self.current_state,
                api_client=self.egress,
                cycle_nonce=cycle_id
            )

            self.current_state = new_state

            # --- Quantitative Strategy Egress ---
            diff = new_state - price

            if abs(diff) > 0.02:
                logger.info(f"Targeting State Shift: {diff:.4f}. Invoking secure egress.")
                payload = self.engine.layer3.prepare_payload(price)

                if not self.dry_run:
                    # execute_api_payload is already called inside engine.run_cycle for the state transition,
                    # but if the strategy performs additional trades, we use the same cycle_id for safety.
                    dispatch_result = await self.engine.layer3.execute_api_payload(
                        payload, [], api_client=self.egress, intent_nonce=cycle_id
                    )

            curvature = self.engine.telemetry.metrics["semantic_curvature"][-1]
            logger.info(f"Seq {count}: Price={price:.2f} | Curvature={curvature:.2e} | Target={new_state:.4f}")

            count += 1

        logger.info("🏁 Mainnet Anchor Sequence Terminated.")
