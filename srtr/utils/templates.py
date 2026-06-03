import logging

logger = logging.getLogger("SRTR.Templates")

class AdapterSynthesis:
    """
    Synthesizes adapter callables based on anomaly metadata.
    """
    @staticmethod
    def synthesize(anomaly_description):
        """
        Logic to choose a template based on keyword matching.
        """
        desc = anomaly_description.lower()

        if any(k in desc for k in ["missing_field", "rename", "schema"]):
            logger.info("Synthesizing Schema Drift Adapter (mapping state -> target_state)")
            return lambda state: {"target_state": state}

        if any(k in desc for k in ["drift", "timeout", "curvature", "anomaly"]):
            logger.info("Synthesizing Scaling Compensation Adapter (x1.1)")
            return lambda state: state * 1.1

        logger.info("Synthesizing Identity Adapter")
        return lambda x: x
