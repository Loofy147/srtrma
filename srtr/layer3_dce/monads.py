import logging
import time

logger = logging.getLogger("SRTR.Monad")

class ExecutionMonad:
    """
    Encapsulates an execution step with safety boundaries.
    Enforces timeouts, budget caps, and error propagation.
    """
    def __init__(self, value=None, error=None, logs=None):
        self.value = value
        self.error = error
        self.logs = logs or []

    @classmethod
    def unit(cls, value):
        return cls(value=value)

    def bind(self, func, timeout=5.0, budget_cost=0):
        """
        Monadic bind: chains execution steps with safety checks.
        """
        if self.error:
            return self

        start_time = time.time()
        try:
            # Simulate budget check (placeholder)
            if budget_cost > 100: # Arbitrary cap
                raise Exception("Budget exceeded")

            result = func(self.value)

            # Handle API responses that contain errors but didn't raise exceptions
            if isinstance(result, dict) and result.get("status") and result.get("status") != 200:
                raise Exception(result.get("error") or f"API Error: {result.get('status')}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"Execution timeout: {elapsed:.2f}s > {timeout}s")

            return ExecutionMonad(value=result, logs=self.logs + [f"Success: {func.__name__}"])
        except Exception as e:
            logger.error(f"Monadic Execution Error: {str(e)}")
            return ExecutionMonad(error=str(e), logs=self.logs + [f"Error in {func.__name__}: {str(e)}"])

    def unwrap(self):
        return self.value, self.error, self.logs

class APIWrapper:
    """
    Wraps external API calls in the ExecutionMonad.
    """
    def __init__(self, api_client):
        self.api_client = api_client

    def safe_call(self, payload):
        def call_logic(p):
            return self.api_client.call(p)

        return ExecutionMonad.unit(payload).bind(call_logic)
