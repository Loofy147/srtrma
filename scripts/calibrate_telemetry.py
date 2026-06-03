import asyncio
import torch
import numpy as np
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srtr.engine import SRTREngine
from srtr.utils.networking import LiveNetworkBridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SRTR.Calibration")

async def run_calibration(iterations=20):
    input_dim = 1
    hidden_dim = 8
    engine = SRTREngine(input_dim, hidden_dim, hidden_dim)
    bridge = LiveNetworkBridge()

    # Calibrating thresholds: No anomalies yet
    engine.regeneration.threshold = 100.0

    current_state = 1.0
    curvatures = []

    logger.info(f"Starting Calibration over {iterations} iterations...")

    for i in range(iterations):
        # Fetch real telemetry
        telemetry = await bridge.fetch_telemetry_async()

        if "error" in telemetry:
            logger.warning(f"Iteration {i}: Fetch Error, skipping.")
            continue

        # Map telemetry value to engine input
        # We wrap it in a pseudo-sequence (batch=1, seq=5, dim=1)
        val = telemetry["value"]
        # Normalize/Scale if necessary. Here we just use a small window around the value.
        input_data = torch.tensor([[[val] for _ in range(5)]], dtype=torch.float32)

        adj_matrix = torch.ones(1, 5, 5)
        edge_weights = torch.rand(1, 5, 5)

        # Run engine cycle
        new_state, result = await engine.run_cycle(input_data, adj_matrix, edge_weights, current_state)

        current_state = new_state

        # Collect telemetry
        curvature = engine.telemetry.metrics["semantic_curvature"][-1]
        curvatures.append(curvature)

        logger.info(f"Iteration {i}: Value={val:.2f}, Curvature={curvature:.6f}")

    avg_curvature = np.mean(curvatures)
    std_curvature = np.std(curvatures)

    # Suggested anomaly threshold: mean + 3*std
    suggested_threshold = avg_curvature + 3 * std_curvature

    logger.info("--- Calibration Complete ---")
    logger.info(f"Average Curvature: {avg_curvature:.6f}")
    logger.info(f"Standard Deviation: {std_curvature:.6f}")
    logger.info(f"Suggested Anomaly Threshold (H^n Cohomology): {suggested_threshold:.6f}")

if __name__ == "__main__":
    asyncio.run(run_calibration())
