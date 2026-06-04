import numpy as np
import logging
import asyncio
from srtr.layer3_dce.monads import APIWrapper
from srtr.utils.audit import StateAuditor

logger = logging.getLogger("SRTR.Layer3")

class DeterministicConstrainedExecution:
    """
    Layer 3: Deterministic Constrained Execution (DCE)
    Implements exact, risk-bounded instructions via external APIs.
    Uses constrained Ornstein-Uhlenbeck (Mean Reversion).
    Now supports asynchronous execution, state parity auditing, and intent-based idempotency.
    """
    def __init__(self, theta_base, mu_base, sigma_base):
        self.theta_base = theta_base
        self.mu_base = mu_base
        self.sigma_base = sigma_base
        self.adapter = None
        self.active_orders = {} # Simulating active order states
        self.executed_intents = set() # Idempotency log for logical state intent

    def get_state_snapshot(self):
        """
        Captures the current parameters and active execution context.
        Excludes the adapter itself as it is expected to change.
        """
        return {
            "theta_base": self.theta_base,
            "mu_base": self.mu_base,
            "sigma_base": self.sigma_base,
            "active_orders": self.active_orders,
            "idempotency_keys": list(self.executed_intents)
        }

    def restore_state_snapshot(self, snapshot):
        """
        Restores state from a snapshot.
        """
        self.theta_base = snapshot.get("theta_base", self.theta_base)
        self.mu_base = snapshot.get("mu_base", self.mu_base)
        self.sigma_base = snapshot.get("sigma_base", self.sigma_base)
        self.active_orders = snapshot.get("active_orders", self.active_orders)
        self.executed_intents = set(snapshot.get("idempotency_keys", []))

    def compute_ou_process(self, x_t, z_t, dt=0.01):
        """
        Calculates the next mathematical state.
        """
        regime_idx = np.argmax(z_t)
        theta = self.theta_base * (1.0 + 0.5 * regime_idx)
        mu = self.mu_base * (1.0 - 0.2 * regime_idx)
        sigma = self.sigma_base / (1.0 + regime_idx)

        dw = np.random.normal(0, np.sqrt(dt))
        dx = theta * (mu - x_t) * dt + sigma * dw

        return x_t + dx

    def prepare_payload(self, target_state):
        """
        Converts mathematical state into an execution payload.
        Applies adapter if present.
        """
        if self.adapter is not None:
            adapted = self.adapter(target_state)
            if isinstance(adapted, dict):
                return adapted
            return {"state": adapted}

        return {"state": target_state}

    async def execute_api_payload(self, payload, constraints, api_client=None, intent_nonce=None):
        """
        Deterministic interaction with external API (Async).
        Enforces intent-based idempotency via state auditing.
        """
        logger.info(f"Executing payload: {payload}")

        # Intent-based Idempotency Check:
        # We hash the logical target state (intent) rather than the transmission payload
        # to ensure retries with new timestamps/signatures are blocked if the intent is same.
        logical_intent = {"target": payload.get("state") or payload.get("payload", {}).get("state")}
        if intent_nonce:
            logical_intent["nonce"] = intent_nonce

        intent_hash = StateAuditor.compute_state_hash(logical_intent)

        if intent_hash in self.executed_intents:
            logger.warning(f"Duplicate intent detected! ({intent_hash}). Skipping dispatch.")
            return {"success": True, "reason": "idempotency_hit"}

        # Boundary Check: locate state even in nested payloads
        target_state = logical_intent["target"]
        if target_state is not None:
            for c in constraints:
                if not c(target_state):
                    logger.error(f"Constraint violation for state {target_state}!")
                    return {"success": False, "reason": "constraint_violation"}

        if api_client:
            # We pass the intent_hash as the integrity token
            wrapper = APIWrapper(api_client)
            monad = wrapper.safe_call(payload, integrity_hash=intent_hash)

            # If the bind returned a coroutine, await it
            if asyncio.iscoroutine(monad):
                monad = await monad

            _, error, logs = monad.unwrap()

            if error:
                logger.error(f"Execution Monad Error: {error}")
                return {"success": False, "reason": error}

            # Record success for idempotency
            self.executed_intents.add(intent_hash)

        logger.info("API Call Successful.")
        return {"success": True}
