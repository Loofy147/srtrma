import torch
import numpy as np
import logging
import asyncio
from srtr.layer1_tgp.tgp import TopologicalGaugePerception
from srtr.layer2_rmc.rmc import RelationalMetaController
from srtr.layer3_dce.dce import DeterministicConstrainedExecution
from srtr.utils.regeneration import SelfRegenerationLoop
from srtr.utils.telemetry import SRTRTelemetry

logger = logging.getLogger("SRTR.Engine")

class SRTREngine:
    """
    Main Engine coordinating TGP, RMC, and DCE layers.
    Optimized for Stream-Native execution and State Parity Auditing.
    """
    def __init__(self, input_dim, hidden_dim, node_dim, num_regimes=5):
        self.layer1 = TopologicalGaugePerception(input_dim, hidden_dim)
        self.layer2 = RelationalMetaController(node_dim, num_regimes)
        self.layer3 = DeterministicConstrainedExecution(theta_base=0.1, mu_base=1.0, sigma_base=0.2)

        self.regeneration = SelfRegenerationLoop()
        self.telemetry = SRTRTelemetry()

    async def run_cycle(self, input_data, adj_matrix, edge_weights, current_state, constraints=[], api_client=None):
        """
        Executes one full SRTR cycle (Async).
        Now includes automated State Parity Auditing during hot-swaps.
        """
        # --- Layer 1: Topological Gauge Perception ---
        # psi: (batch, seq, hidden), covariant_derivative: (batch, seq, hidden)
        psi, covariant_derivative = self.layer1(input_data)

        # Anomaly Detection (Layer 1)
        # If structural reality shifts (high curvature), trigger hot-swap with parity check
        if self.regeneration.detect_anomaly(covariant_derivative):
            anomaly_desc = f"Topological Anomaly (Curvature: {torch.abs(torch.mean(covariant_derivative)).item():.2e})"
            new_adapter = self.regeneration.trigger_closure_lemma(anomaly_desc)

            # The hot_swap_adapter method in regeneration.py handles the SHA-256 audit internally
            swap_success = self.regeneration.hot_swap_adapter(self.layer3, new_adapter)
            if not swap_success:
                logger.error("State Parity Audit failed during Layer 1 hot-swap! Cycle execution might be compromised.")

        # --- Layer 2: Relational Meta-Controller ---
        nodes = psi
        h_next, z_t = self.layer2(nodes, adj_matrix, edge_weights)

        # --- Layer 3: Deterministic Constrained Execution ---
        z_t_np = z_t.detach().numpy()[0]

        # 1. Compute mathematical next state
        new_state = self.layer3.compute_ou_process(current_state, z_t_np)

        # 2. Track metrics (always scalar state)
        self.telemetry.compute_semantic_curvature(covariant_derivative)
        self.telemetry.track_convergence(torch.tensor([new_state], dtype=torch.float32), torch.tensor([self.layer3.mu_base], dtype=torch.float32))

        # 3. Prepare and Execute Payload
        payload = self.layer3.prepare_payload(new_state)

        # execute_api_payload handles idempotency hashing and integrity checks
        result = await self.layer3.execute_api_payload(payload, constraints, api_client=api_client)

        # Check if execution failure triggers regeneration (Layer 3)
        if not result["success"] and result.get("reason") != "idempotency_hit":
            logger.warning(f"Execution failed: {result['reason']}. Initiating Self-Regeneration.")
            anomaly_desc = f"API Execution Failure: {result['reason']}"
            new_adapter = self.regeneration.trigger_closure_lemma(anomaly_desc)

            # Hot-swap with State Parity Audit
            self.regeneration.hot_swap_adapter(self.layer3, new_adapter)

            # Re-attempt preparation with new adapter
            payload = self.layer3.prepare_payload(new_state)
            result = await self.layer3.execute_api_payload(payload, constraints, api_client=api_client)

        self.telemetry.log_metrics()

        return new_state, result
