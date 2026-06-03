import torch
import numpy as np
import logging
from srtr.layer1_tgp.tgp import TopologicalGaugePerception
from srtr.layer2_rmc.rmc import RelationalMetaController
from srtr.layer3_dce.dce import DeterministicConstrainedExecution
from srtr.utils.regeneration import SelfRegenerationLoop
from srtr.utils.telemetry import SRTRTelemetry

logger = logging.getLogger("SRTR.Engine")

class SRTREngine:
    """
    Main Engine coordinating TGP, RMC, and DCE layers.
    """
    def __init__(self, input_dim, hidden_dim, node_dim, num_regimes=5):
        self.layer1 = TopologicalGaugePerception(input_dim, hidden_dim)
        self.layer2 = RelationalMetaController(node_dim, num_regimes)
        self.layer3 = DeterministicConstrainedExecution(theta_base=0.1, mu_base=1.0, sigma_base=0.2)

        self.regeneration = SelfRegenerationLoop()
        self.telemetry = SRTRTelemetry()

    def run_cycle(self, input_data, adj_matrix, edge_weights, current_state, constraints=[], api_client=None):
        """
        Executes one full SRTR cycle.
        """
        # --- Layer 1: Topological Gauge Perception ---
        psi, covariant_derivative = self.layer1(input_data)

        # Telemetry & Anomaly Detection
        self.telemetry.compute_semantic_curvature(covariant_derivative)
        if self.regeneration.detect_anomaly(covariant_derivative):
            new_adapter = self.regeneration.trigger_closure_lemma("Topological Anomaly detected in Layer 1")
            self.regeneration.hot_swap_adapter(self.layer3, new_adapter)

        # --- Layer 2: Relational Meta-Controller ---
        nodes = psi
        h_next, z_t = self.layer2(nodes, adj_matrix, edge_weights)

        # --- Layer 3: Deterministic Constrained Execution ---
        z_t_np = z_t.detach().numpy()[0]
        new_state = self.layer3.compute_ou_process(current_state, z_t_np)

        # Track convergence
        self.telemetry.track_convergence(torch.tensor([new_state]), torch.tensor([self.layer3.mu_base]))

        # Execution
        result = self.layer3.execute_api_payload(new_state, constraints, api_client=api_client)

        # Check if execution failure triggers regeneration (Phase 1 Chaos)
        if not result["success"]:
            logger.warning(f"Execution failed due to: {result['reason']}. Initiating Self-Regeneration.")
            new_adapter = self.regeneration.trigger_closure_lemma(f"API Execution Failure: {result['reason']}")
            self.regeneration.hot_swap_adapter(self.layer3, new_adapter)
            # Re-attempt with adapter
            new_state = self.layer3.compute_ou_process(current_state, z_t_np)
            result = self.layer3.execute_api_payload(new_state, constraints, api_client=api_client)

        self.telemetry.log_metrics()

        return new_state, result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_dim = 10
    hidden_dim = 16
    engine = SRTREngine(input_dim, hidden_dim, hidden_dim)

    input_data = torch.randn(1, 5, input_dim)
    adj_matrix = torch.ones(1, 5, 5)
    edge_weights = torch.rand(1, 5, 5)

    current_state = 0.5
    new_state, result = engine.run_cycle(input_data, adj_matrix, edge_weights, current_state)
    print(f"Cycle Complete. Success: {result['success']}")
