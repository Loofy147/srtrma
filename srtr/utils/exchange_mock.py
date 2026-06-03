import logging
import random
import asyncio
import time

logger = logging.getLogger("SRTR.Exchange")

class MockExchangeClient:
    """
    Simulates a live quantitative trading exchange.
    Enforces realistic latency, rate limits, and order execution constraints.
    """
    def __init__(self, initial_balance=10000.0):
        self.balance = initial_balance
        self.position = 0.0
        self.last_call_time = 0
        self.rate_limit_delay = 0.2 # Increased delay for more reliable testing
        self.error_rate = 0.02 # 2% random API error rate

    async def call(self, payload):
        """
        Processes a trading payload.
        Expected structure: {"action": "buy"|"sell", "amount": float, "price": float}
        """
        # --- Simulate Network/API Constraints ---
        current_time = time.time()
        if current_time - self.last_call_time < self.rate_limit_delay:
            logger.warning("MockExchange: Rate limit triggered!")
            return {"status": 429, "error": "Too Many Requests"}

        # Update last_call_time
        self.last_call_time = current_time

        # Simulate transient network errors
        if random.random() < self.error_rate:
            logger.error("MockExchange: Internal Server Error")
            return {"status": 500, "error": "Internal Server Error"}

        # Simulate execution latency
        await asyncio.sleep(0.01) # Reduced latency to ensure rate limit hits in tests

        action = payload.get("action")
        amount = payload.get("amount", 0.0)
        price = payload.get("price", 0.0)

        # Handle simple state-based preparation payloads or direct action payloads
        if "state" in payload and not action:
            return {"status": 200, "message": "State tracked successfully"}

        if action == "buy":
            cost = amount * price
            if cost > self.balance:
                return {"status": 400, "error": "Insufficient Funds"}
            self.balance -= cost
            self.position += amount
            logger.info(f"MockExchange: BOUGHT {amount} at {price}. New Balance: {self.balance:.2f}")
            return {"status": 200, "executed": True, "side": "buy", "balance": self.balance}

        elif action == "sell":
            if amount > self.position:
                return {"status": 400, "error": "Insufficient Position"}
            gain = amount * price
            self.balance += gain
            self.position -= amount
            logger.info(f"MockExchange: SOLD {amount} at {price}. New Balance: {self.balance:.2f}")
            return {"status": 200, "executed": True, "side": "sell", "balance": self.balance}

        return {"status": 200, "message": "No action taken"}
