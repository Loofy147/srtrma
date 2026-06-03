import asyncio
import torch
import numpy as np
import sys
import os
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.networking import LiveNetworkBridge
from srtr.utils.exchange_mock import MockExchangeClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SRTR.AlphaPilot")

async def run_alpha_pilot(iterations=50):
    """
    Alpha Pilot: Quantitative Regime-Switching Engine.
    Executes a constrained Mean Reversion strategy against live market data.
    """
    input_dim = 1
    hidden_dim = 8
    engine = SRTREngine(input_dim, hidden_dim, hidden_dim)
    bridge = LiveNetworkBridge()
    exchange = MockExchangeClient(initial_balance=100000.0)

    # Set anomaly threshold based on prior calibration
    engine.regeneration.threshold = 1.5 # Adjusted for pilot sensitivity

    current_price = 0.0
    history = []

    logger.info("🚀 Launching SRTR Alpha Pilot: Quantitative Regime-Switching Engine")

    for i in range(iterations):
        # 1. Ingest Live Telemetry
        telemetry = await bridge.fetch_telemetry_async()
        if "error" in telemetry:
            continue

        price = telemetry["value"]
        history.append(price)
        if len(history) > 50:
            history.pop(0)

        # 2. Map into Topological State
        # (batch=1, seq=5, dim=1)
        input_data = torch.tensor([[[price] for _ in range(5)]], dtype=torch.float32)
        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        # 3. Execute SRTR Cycle
        # We pass the current price as the state to revert around
        new_state, result = await engine.run_cycle(
            input_data,
            adj_matrix,
            edge_weights,
            price,
            api_client=exchange
        )

        # 4. Strategy Logic: Convert mathematical state to trading action
        # If new_state (target price) is significantly higher than current, buy.
        # This is a simplified mean reversion where Layer 3 mu is the 'anchor'.
        diff = new_state - price

        trade_payload = None
        if diff > 0.05: # Buy if target suggests upward mean reversion
            trade_payload = {"action": "buy", "amount": 0.1, "price": price}
        elif diff < -0.05: # Sell if target suggests downward mean reversion
            trade_payload = {"action": "sell", "amount": 0.1, "price": price}

        if trade_payload:
            logger.info(f"Pilot Decision: {trade_payload['action'].upper()} triggered by state drift {diff:.4f}")
            trade_result = await exchange.call(trade_payload)
            if trade_result.get("status") != 200:
                logger.warning(f"Trade Execution Failed: {trade_result.get('error')}")

        # 5. Telemetry Logging
        curvature = engine.telemetry.metrics["semantic_curvature"][-1]
        logger.info(f"Iteration {i}: Price={price:.2f}, Curvature={curvature:.6f}, Balance={exchange.balance:.2f}")

        # Dynamic rate control
        await asyncio.sleep(0.5)

    logger.info("🏁 Alpha Pilot Sequence Terminated.")
    logger.info(f"Final Balance: {exchange.balance:.2f} | Net Position: {exchange.position:.4f}")

if __name__ == "__main__":
    try:
        asyncio.run(run_alpha_pilot())
    except KeyboardInterrupt:
        logger.info("Alpha Pilot Aborted by User.")
