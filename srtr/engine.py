import torch
import numpy as np
import logging
import asyncio
from srtr.layer1_tgp.tgp import TopologicalGaugePerception
from srtr.layer2_rmc.rmc import RelationalMetaController
from srtr.layer3_dce.dce import DeterministicConstrainedExecution
from srtr.utils.regeneration import SelfRegenerationLoop
from srtr.utils.telemetry import SRTRTelemetry
from srtr.utils.acfs import FieldGraph

logger = logging.getLogger("SRTR.Engine")

class SRTREngine:
    """
    Main Engine coordinating TGP, RMC, and DCE layers.
    Refactored to use ACFS v2.0 reactive FieldGraph for state and dependency management.
    """
    def __init__(self, input_dim, hidden_dim, node_dim, num_regimes=5):
        self.layer1 = TopologicalGaugePerception(input_dim, hidden_dim)
        self.layer2 = RelationalMetaController(node_dim, num_regimes)
        self.layer3 = DeterministicConstrainedExecution(theta_base=0.1, mu_base=1.0, sigma_base=0.2)

        self.regeneration = SelfRegenerationLoop()
        self.telemetry = SRTRTelemetry()

        self.graph = FieldGraph("SRTR_Core")
        self._setup_reactive_pipeline()

    def _setup_reactive_pipeline(self):
        """Initializes the reactive dependency graph for the SRTR cycle."""

        # 1. Sources (Ingress)
        self.graph.source("input_data")
        self.graph.source("adj_matrix")
        self.graph.source("edge_weights")
        self.graph.source("current_state")
        self.graph.source("constraints")
        self.graph.source("api_client")
        self.graph.source("cycle_nonce")

        # 2. Layer 1: Topological Gauge Perception
        async def resolve_tgp(input_data):
            psi, covariant_derivative = self.layer1(input_data)

            # Anomaly Detection (Layer 1)
            if self.regeneration.detect_anomaly(covariant_derivative):
                anomaly_desc = f"Topological Anomaly (Curvature: {torch.abs(torch.mean(covariant_derivative)).item():.2e})"
                new_adapter = self.regeneration.trigger_closure_lemma(anomaly_desc)

                swap_success = self.regeneration.hot_swap_adapter(self.layer3, new_adapter)
                if not swap_success:
                    logger.error("State Parity Audit failed during Layer 1 hot-swap!")

            return {"psi": psi, "covariant_derivative": covariant_derivative}

        self.graph.derive("tgp", resolve_tgp, ["input_data"])

        # 3. Layer 2: Relational Meta-Controller
        def resolve_rmc(tgp, adj_matrix, edge_weights):
            nodes = tgp["psi"]
            h_next, z_t = self.layer2(nodes, adj_matrix, edge_weights)
            return {"h_next": h_next, "z_t": z_t}

        self.graph.derive("rmc", resolve_rmc, ["tgp", "adj_matrix", "edge_weights"])

        # 4. Layer 3: Deterministic Constrained Execution (Math)
        def resolve_new_state(current_state, rmc, tgp):
            z_t = rmc["z_t"]
            z_t_np = z_t.detach().numpy()[0]

            new_state = self.layer3.compute_ou_process(current_state, z_t_np)

            # Track metrics using telemetry
            self.telemetry.compute_semantic_curvature(tgp["covariant_derivative"])
            self.telemetry.track_convergence(
                torch.tensor([new_state], dtype=torch.float32),
                torch.tensor([self.layer3.mu_base], dtype=torch.float32)
            )
            return new_state

        self.graph.derive("new_state", resolve_new_state, ["current_state", "rmc", "tgp"])

        # 5. Layer 3: Execution Egress
        async def resolve_execution(new_state, constraints, api_client, cycle_nonce):
            payload = self.layer3.prepare_payload(new_state)

            result = await self.layer3.execute_api_payload(
                payload, constraints, api_client=api_client, intent_nonce=cycle_nonce)

            # Check if execution failure triggers regeneration (Layer 3)
            if not result["success"] and result.get("reason") != "idempotency_hit":
                logger.warning(f"Execution failed: {result['reason']}. Initiating Self-Regeneration.")
                anomaly_desc = f"API Execution Failure: {result['reason']}"
                new_adapter = self.regeneration.trigger_closure_lemma(anomaly_desc)

                self.regeneration.hot_swap_adapter(self.layer3, new_adapter)

                # Re-attempt preparation with new adapter
                payload = self.layer3.prepare_payload(new_state)
                result = await self.layer3.execute_api_payload(
                    payload, constraints, api_client=api_client, intent_nonce=cycle_nonce)

            self.telemetry.log_metrics()
            return result

        self.graph.derive("execution", resolve_execution, ["new_state", "constraints", "api_client", "cycle_nonce"])

    async def run_cycle(self, input_data, adj_matrix, edge_weights, current_state, constraints=[], api_client=None, cycle_nonce=None):
        """
        Executes one full SRTR cycle using the reactive FieldGraph.
        """
        # Set new values for the current cycle
        await self.graph.set("input_data", input_data)
        await self.graph.set("adj_matrix", adj_matrix)
        await self.graph.set("edge_weights", edge_weights)
        await self.graph.set("current_state", current_state)
        await self.graph.set("constraints", constraints)
        await self.graph.set("api_client", api_client)
        await self.graph.set("cycle_nonce", cycle_nonce)

        # Drive execution through the graph
        # We need both the new_state (for the next cycle) and the execution result
        new_state, result = await asyncio.gather(
            self.graph.get("new_state"),
            self.graph.get("execution")
        )

        return new_state, result
