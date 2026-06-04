import logging
import ast
import hashlib
import hmac
import time
import json
import os

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
    "hmac_signed_adapter": """
import hmac
import hashlib
import json
import time
import os

def adapter(state):
    # Blueprint for HMAC-signed payloads (Live Exchange Egress)
    secret = os.environ.get("SRTR_API_SECRET", "DEMO_SECRET")
    api_key = os.environ.get("SRTR_API_KEY", "DEMO_KEY")

    payload = {{"state": state, "timestamp": int(time.time() * 1000)}}
    payload_str = json.dumps(payload, sort_keys=True)

    signature = hmac.new(
        secret.encode('utf-8'),
        payload_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return {{
        "payload": payload,
        "headers": {{
            "X-SRTR-API-KEY": api_key,
            "X-SRTR-SIGNATURE": signature
        }},
        "method": "POST"
    }}
""",
    "oauth2_authorized_adapter": """
import os

def adapter(state):
    # Blueprint for OAuth2 authorized enterprise workflows
    return {{
        "payload": {{"data": state}},
        "auth": {{
            "type": "oauth2",
            "token_url": "{token_url}",
            "scope": "{scope}",
            "client_id": os.environ.get("SRTR_OAUTH_CLIENT_ID"),
            "client_secret": os.environ.get("SRTR_OAUTH_CLIENT_SECRET")
        }},
        "method": "PATCH"
    }}
""",
    "rest_pagination_adapter": """
def adapter(state):
    # Blueprint for handling paginated REST requests
    return {{
        "payload": {{"state": state, "limit": 100}},
        "pagination": {{"enabled": True, "next_key": "next_page_token"}},
        "method": "GET"
    }}
""",
    "websocket_stream_adapter": """
def adapter(state):
    # Blueprint for persistent WebSocket stream initialization
    return {{
        "protocol": "ws",
        "payload": {{"action": "subscribe", "topic": "telemetry"}},
        "stream_filter": state
    }}
""",
    "webhook_listener_adapter": """
def adapter(state):
    # Blueprint for mapping incoming webhook payloads to internal state
    return {{
        "mode": "listener",
        "endpoint": "/webhooks/alpha",
        "expected_schema": {{"id": "str", "value": "float"}},
        "filter_condition": state
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

        # Order of checks matters if descriptions overlap.
        # Check for specific protocols first.
        if "pagination" in desc or "page" in desc:
            logger.info("Selected REST Pagination Template")
            template_key = "rest_pagination_adapter"

        elif "websocket" in desc or "stream" in desc or "ws" in desc:
            logger.info("Selected WebSocket Stream Template")
            template_key = "websocket_stream_adapter"

        elif "webhook" in desc or "hook" in desc or "listener" in desc:
            logger.info("Selected Webhook Listener Template")
            template_key = "webhook_listener_adapter"

        elif any(k in desc for k in ["missing_field", "rename", "schema"]):
            logger.info("Selected Schema Mapping Template")
            template_key = "schema_mapping"

        elif any(k in desc for k in ["drift", "timeout", "curvature", "anomaly"]):
            logger.info("Selected Scaling Compensation Template")
            template_key = "scaling_compensation"
            context = {"factor": 1.1}

        elif any(k in desc for k in ["signature", "hmac", "sign"]):
            logger.info("Selected HMAC Signed Template")
            template_key = "hmac_signed_adapter"
            # Secrets now loaded via os.environ inside the template

        elif any(k in desc for k in ["oauth2", "authorize_enterprise"]):
            logger.info("Selected OAuth2 Template")
            template_key = "oauth2_authorized_adapter"
            context = {
                "token_url": "https://auth.contacto.dz/token",
                "scope": "alpha.pilot.execute"
            }

        elif any(k in desc for k in ["oauth", "bearer", "token", "auth", "unauthorized"]):
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

            # We need to make sure the adapter has access to the modules it needs
            # The template itself should contain the imports
            local_vars = {}
            exec(hydrated_code, globals(), local_vars)
            adapter_func = local_vars.get("adapter")

            if not adapter_func:
                raise ValueError("Template did not define 'adapter' function")

            return adapter_func

        except Exception as e:
            logger.error(f"Synthesis failed to compile template: {str(e)}")
            return lambda x: x
