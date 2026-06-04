import aiohttp
import logging
import asyncio
import time

logger = logging.getLogger("SRTR.ExchangeClient")

class LiveExchangeClient:
    """
    Production-grade connector for live external platforms.
    Dispatches requests authenticated by synthesized adapters.
    """
    def __init__(self, base_url="https://api.binance.com"):
        self.base_url = base_url

    async def call(self, payload):
        """
        Executes a live API dispatch strictly within the ExecutionMonad boundary.
        payload: {
            "payload": dict,
            "headers": dict,
            "method": "POST"|"GET"|"PATCH"|"PUT",
            "endpoint": str (optional)
        }
        """
        method = payload.get("method", "POST")
        headers = payload.get("headers", {})
        data = payload.get("payload", {})
        endpoint = payload.get("endpoint", "/api/v3/order")

        url = f"{self.base_url}{endpoint}"

        logger.info(f"LiveExchangeClient: Dispatching {method} to {url}")

        try:
            async with aiohttp.ClientSession() as session:
                if method == "POST":
                    async with session.post(url, json=data, headers=headers) as response:
                        return await self._handle_response(response)
                elif method == "GET":
                    async with session.get(url, params=data, headers=headers) as response:
                        return await self._handle_response(response)
                elif method == "PATCH":
                    async with session.patch(url, json=data, headers=headers) as response:
                        return await self._handle_response(response)
                else:
                    return {"status": 405, "error": f"Method {method} not supported"}
        except Exception as e:
            logger.error(f"LiveExchangeClient: Network Exception: {e}")
            return {"status": 500, "error": f"Network Exception: {str(e)}"}

    async def _handle_response(self, response):
        """Algebraic state normalization for the ExecutionMonad."""
        status = response.status
        try:
            body = await response.json()
        except:
            body = {"message": await response.text()}

        if status in [200, 201, 202]:
            logger.info(f"LiveExchangeClient: Success ({status})")
            return {"status": 200, "data": body, "success": True}
        else:
            error_msg = body.get("msg") or body.get("message") or "Unknown Error"
            logger.warning(f"LiveExchangeClient: API Failure ({status}) - {error_msg}")
            return {"status": status, "error": error_msg, "success": False}
