import torch
import logging

logger = logging.getLogger("SRTR")

class SRTRTelemetry:
    """
    Tracks operational stability metrics.
    """
    def __init__(self):
        self.metrics = {
            "semantic_curvature": [],
            "subgoal_convergence": [],
            "regeneration_latency": []
        }

    def compute_semantic_curvature(self, covariant_derivative):
        """
        R_uv measures the rate of contextual distortion.
        Simplified: Average magnitude of the covariant derivative.
        """
        curvature = torch.mean(torch.abs(covariant_derivative)).item()
        self.metrics["semantic_curvature"].append(curvature)
        return curvature

    def track_convergence(self, state, target_goal):
        """
        Quantifies how effectively the low-level worker achieves targets.
        """
        convergence = torch.norm(state - target_goal).item()
        self.metrics["subgoal_convergence"].append(convergence)
        return convergence

    def log_metrics(self):
        logger.info("--- SRTR Telemetry ---")
        for k, v in self.metrics.items():
            if v:
                logger.info(f"{k}: {v[-1]:.6f}")
