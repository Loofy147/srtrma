import random
import logging

logger = logging.getLogger("SRTR.Chaos")

class PoisonedAPI:
    """
    Mocks an external API that can exhibit chaotic behavior.
    """
    def __init__(self, failure_rate=0.2):
        self.failure_rate = failure_rate
        self.schema_mutated = False

    def call(self, payload):
        """
        Simulates an API call that might fail or return inconsistent data.
        """
        if random.random() < self.failure_rate:
            if random.random() < 0.5:
                logger.error("API Call: Field Drop Anomaly detected.")
                return {"error": "missing_field", "status": 400}
            else:
                logger.error("API Call: High Latency Anomaly detected.")
                return {"error": "timeout", "status": 504}

        if self.schema_mutated:
            logger.warning("API Call: Schema Mutation detected.")
            return {"data": payload, "status": 200, "metadata": "MUTATED"}

        return {"data": payload, "status": 200}

    def mutate_schema(self):
        logger.info("Chaos: Poisoning API Schema...")
        self.schema_mutated = True

class ChaosGenerator:
    """
    Injects structural anomalies into the SRTR data stream.
    """
    def __init__(self, engine):
        self.engine = engine

    def inject_semantic_drift(self, input_data, intensity=5.0):
        """
        Adds high-intensity noise to input data to trigger topological anomalies.
        """
        logger.info(f"Chaos: Injecting semantic drift (intensity={intensity})...")
        noise = torch.randn_like(input_data) * intensity
        return input_data + noise

import torch # Ensure torch is available for ChaosGenerator
