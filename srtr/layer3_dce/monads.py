import logging
import time
import asyncio
import inspect

logger = logging.getLogger("SRTR.Monad")

class ExecutionMonad:
    """
    Encapsulates an execution step with safety boundaries.
    Enforces timeouts, budget caps, and error propagation.
    Now supports integrity check hooks for State Parity.
    """
    def __init__(self, value=None, error=None, logs=None, integrity_hash=None):
        self.value = value
        self.error = error
        self.logs = logs or []
        self.integrity_hash = integrity_hash

    @classmethod
    def unit(cls, value, integrity_hash=None):
        return cls(value=value, integrity_hash=integrity_hash)

    def bind(self, func, timeout=5.0, budget_cost=0):
        """
        Monadic bind: chains execution steps with safety checks.
        Handles both sync and async functions.
        """
        if self.error:
            return self

        # If it's an async function, return a wrapper that needs to be awaited
        if asyncio.iscoroutinefunction(func):
            return self._bind_async(func, timeout, budget_cost)

        return self._bind_sync(func, timeout, budget_cost)

    def _bind_sync(self, func, timeout, budget_cost):
        start_time = time.time()
        try:
            if budget_cost > 100:
                raise Exception("Budget exceeded")

            result = func(self.value)

            if isinstance(result, dict) and result.get("status") and result.get("status") != 200:
                raise Exception(result.get("error") or f"API Error: {result.get('status')}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"Execution timeout: {elapsed:.2f}s > {timeout}s")

            return ExecutionMonad(
                value=result,
                logs=self.logs + [f"Success: {func.__name__}"],
                integrity_hash=self.integrity_hash
            )
        except Exception as e:
            logger.error(f"Monadic Execution Error (Sync): {str(e)}")
            return ExecutionMonad(error=str(e), logs=self.logs + [f"Error in {func.__name__}: {str(e)}"])

    async def _bind_async(self, func, timeout, budget_cost):
        start_time = time.time()
        try:
            if budget_cost > 100:
                raise Exception("Budget exceeded")

            # Execute with timeout
            try:
                result = await asyncio.wait_for(func(self.value), timeout=timeout)
            except asyncio.TimeoutError:
                raise Exception(f"Execution timeout (Async) exceeded {timeout}s")

            if isinstance(result, dict) and result.get("status") and result.get("status") != 200:
                raise Exception(result.get("error") or f"API Error: {result.get('status')}")

            return ExecutionMonad(
                value=result,
                logs=self.logs + [f"Success (Async): {func.__name__}"],
                integrity_hash=self.integrity_hash
            )
        except Exception as e:
            logger.error(f"Monadic Execution Error (Async): {str(e)}")
            return ExecutionMonad(error=str(e), logs=self.logs + [f"Error in {func.__name__}: {str(e)}"])

    def unwrap(self):
        return self.value, self.error, self.logs

class APIWrapper:
    """
    Wraps external API calls in the ExecutionMonad.
    """
    def __init__(self, api_client):
        self.api_client = api_client

    def safe_call(self, payload, integrity_hash=None):
        # Determine if api_client.call is async
        call_method = self.api_client.call
        return ExecutionMonad.unit(payload, integrity_hash=integrity_hash).bind(call_method)
