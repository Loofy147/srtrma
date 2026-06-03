import requests
import time
import random
import logging
import asyncio
import aiohttp

logger = logging.getLogger("SRTR.Networking")

class LiveNetworkBridge:
    """
    Gateway Bridge for streaming real-world entropy into the SRTR engine.
    Provides methods for synchronous and asynchronous telemetry fetching.
    """
    def __init__(self, endpoint="https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"):
        self.endpoint = endpoint
        self.jitter_range = (0.005, 0.05) # Optimized for higher frequency
        self.drop_rate = 0.02 # Reduced for cleaner alpha pilot data

    def set_simulation_params(self, jitter_range=None, drop_rate=None):
        if jitter_range:
            self.jitter_range = jitter_range
        if drop_rate is not None:
            self.drop_rate = drop_rate

    def _simulate_noise(self):
        """Simulates network jitter and potential packet drops."""
        if random.random() < self.drop_rate:
            logger.warning("Network Noise: Packet Drop Simulated")
            raise Exception("Simulated Packet Drop")

        jitter = random.uniform(*self.jitter_range)
        time.sleep(jitter)
        return jitter

    async def _simulate_noise_async(self):
        """Asynchronous version of noise simulation."""
        if random.random() < self.drop_rate:
            logger.warning("Network Noise: Async Packet Drop Simulated")
            raise Exception("Simulated Async Packet Drop")

        jitter = random.uniform(*self.jitter_range)
        await asyncio.sleep(jitter)
        return jitter

    def fetch_telemetry_sync(self):
        """Synchronously fetches telemetry data with simulated noise."""
        try:
            self._simulate_noise()
            response = requests.get(self.endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "bitcoin" in data:
                price = float(data["bitcoin"]["usd"])
            else:
                price = float(data.get("price", 1.0))
            return {"value": price, "timestamp": time.time(), "source": self.endpoint}
        except Exception as e:
            logger.error(f"Sync Telemetry Fetch Failed: {str(e)}")
            return {"error": str(e), "status": 500}

    async def fetch_telemetry_async(self):
        """Asynchronously fetches telemetry data with simulated noise."""
        try:
            await self._simulate_noise_async()
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, timeout=5) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
                    if "bitcoin" in data:
                        price = float(data["bitcoin"]["usd"])
                    else:
                        price = float(data.get("price", 1.0))
                    return {"value": price, "timestamp": time.time(), "source": self.endpoint}
        except Exception as e:
            logger.error(f"Async Telemetry Fetch Failed: {str(e)}")
            return {"error": str(e), "status": 500}
