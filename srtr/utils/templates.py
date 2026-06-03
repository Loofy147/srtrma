import logging
import ast

logger = logging.getLogger("SRTR.Templates")

# Repository of AST-compatible code templates
TEMPLATES = {
    "schema_mapping": """
def adapter(state):
    # Mapping state to target_state
    return {{"target_state": state, "status": "adapted"}}
""",
    "scaling_compensation": """
def adapter(state):
    # Applying scaling factor
    return state * {factor}
""",
    "rest_auth_adapter": """
def adapter(state):
    # Injecting authentication and wrapping payload
    return {{
        "payload": {{"state": state}},
        "headers": {{"Authorization": "Bearer {token}"}},
        "method": "POST"
    }}
""",
    "identity": """
def adapter(x):
    return x
"""
}

class AdapterSynthesis:
    """
    Synthesizes adapter callables based on anomaly metadata using a template repository.
    """
    @staticmethod
    def synthesize(anomaly_description):
        """
        Logic to choose and hydrate a template based on keyword matching.
        """
        desc = anomaly_description.lower()
        template_key = "identity"
        context = {}

        if any(k in desc for k in ["missing_field", "rename", "schema"]):
            logger.info("Selected Schema Mapping Template")
            template_key = "schema_mapping"

        elif any(k in desc for k in ["drift", "timeout", "curvature", "anomaly"]):
            logger.info("Selected Scaling Compensation Template")
            template_key = "scaling_compensation"
            context = {"factor": 1.1}

        elif any(k in desc for k in ["auth", "unauthorized", "token"]):
            logger.info("Selected REST Auth Template")
            template_key = "rest_auth_adapter"
            context = {"token": "SRTR-AUTH-LIVE-ALPHA-01"}

        raw_code = TEMPLATES.get(template_key, TEMPLATES["identity"])

        # Hydrate template if context provided
        try:
            hydrated_code = raw_code.format(**context)
        except KeyError as e:
            logger.error(f"Template hydration failed: missing key {e}")
            hydrated_code = TEMPLATES["identity"]

        # Compile code into a callable
        try:
            # Parse to ensure it's a valid AST
            ast.parse(hydrated_code)

            local_vars = {}
            exec(hydrated_code, {}, local_vars)
            adapter_func = local_vars.get("adapter")

            if not adapter_func:
                raise ValueError("Template did not define 'adapter' function")

            return adapter_func

        except Exception as e:
            logger.error(f"Synthesis failed to compile template: {str(e)}")
            return lambda x: x
