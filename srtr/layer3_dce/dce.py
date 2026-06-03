import numpy as np
import logging
import asyncio
from srtr.layer3_dce.monads import APIWrapper

logger = logging.getLogger("SRTR.Layer3")

class DeterministicConstrainedExecution:
    """
    Layer 3: Deterministic Constrained Execution (DCE)
    Implements exact, risk-bounded instructions via external APIs.
    Uses constrained Ornstein-Uhlenbeck (Mean Reversion).
    Now supports asynchronous execution.
    """
    def __init__(self, theta_base, mu_base, sigma_base):
        self.theta_base = theta_base
        self.mu_base = mu_base
        self.sigma_base = sigma_base
        self.adapter = None

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

    async def execute_api_payload(self, payload, constraints, api_client=None):
        """
        Deterministic interaction with external API (Async).
        """
        logger.info(f"Executing payload: {payload}")

        # Boundary Check for scalar target_state if present in payload
        if "state" in payload:
            target_state = payload["state"]
            for c in constraints:
                if not c(target_state):
                    logger.error("Constraint violation!")
                    return {"success": False, "reason": "constraint_violation"}

        if api_client:
            wrapper = APIWrapper(api_client)
            monad = wrapper.safe_call(payload)

            # If the bind returned a coroutine, await it
            if asyncio.iscoroutine(monad):
                monad = await monad

            _, error, logs = monad.unwrap()

            if error:
                logger.error(f"Execution Monad Error: {error}")
                return {"success": False, "reason": error}

        logger.info("API Call Successful.")
        return {"success": True}
