import asyncio
import json
import logging
import aiohttp
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger("SRTR.Streaming")

class LiveStreamAdapter:
    """
    WebSocket ingress for live production telemetry.
    Handles persistent connections and real-time structural mapping.
    """
    def __init__(self, uri: str, subscription_payload: Optional[dict] = None):
        self.uri = uri
        self.subscription_payload = subscription_payload
        self.is_running = False
        self._last_telemetry = None

    async def stream_telemetry(self) -> AsyncGenerator[dict, None]:
        """
        Connects to the WebSocket and yields mapped telemetry data.
        """
        self.is_running = True
        logger.info(f"Connecting to WebSocket stream: {self.uri}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(self.uri) as ws:
                    if self.subscription_payload:
                        await ws.send_json(self.subscription_payload)
                        logger.info(f"Subscription payload sent: {self.subscription_payload}")

                    async for msg in ws:
                        if not self.is_running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                telemetry = self._map_to_telemetry(data)
                                self._last_telemetry = telemetry
                                yield telemetry
                            except json.JSONDecodeError:
                                logger.warning("Received non-JSON message from stream")
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket connection closed")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
            except Exception as e:
                logger.error(f"Streaming connection failed: {e}")
            finally:
                self.is_running = False
                logger.info("Streaming session terminated.")

    def _map_to_telemetry(self, raw_data: dict) -> dict:
        """
        Maps raw platform data into the SRTR internal telemetry schema.
        Specifically tuned for Financial Gateway (Vector Alpha) targets like Binance/Coinbase.
        """
        # Example mapping for Binance-style trade stream: {"p": "price", "T": "timestamp"}
        # Example mapping for Coinbase-style: {"price": "price", "time": "time"}

        price_str = raw_data.get("p") or raw_data.get("price") or "0.0"
        timestamp = raw_data.get("T") or raw_data.get("time") or time.time()

        # Binance sends price as string
        try:
            price = float(price_str)
        except (ValueError, TypeError):
            price = 0.0

        return {
            "value": price,
            "timestamp": timestamp,
            "raw": raw_data
        }

    def get_latest(self) -> Optional[dict]:
        """Returns the most recent telemetry packet received."""
        return self._last_telemetry

    def stop(self):
        """Signals the stream listener to stop."""
        self.is_running = False
