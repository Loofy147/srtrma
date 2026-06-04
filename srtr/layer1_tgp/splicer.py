import torch
import numpy as np
import logging

logger = logging.getLogger("SRTR.Splicer")

class DepthSplicer:
    """
    WebSocket Graph Splicer for Vector Alpha.
    Maps order-book depth distribution into continuous topological manifolds.
    """
    def __init__(self, depth_levels=10):
        self.depth_levels = depth_levels

    def splice_depth(self, bids, asks):
        """
        Converts raw bid/ask lists into a normalized depth tensor.
        bids/asks: List of [price, quantity]
        """
        # Take up to depth_levels
        bids = bids[:self.depth_levels]
        asks = asks[:self.depth_levels]

        # Pad if necessary
        while len(bids) < self.depth_levels:
            bids.append([0.0, 0.0])
        while len(asks) < self.depth_levels:
            asks.append([0.0, 0.0])

        # Create normalized distribution features
        # [price_offset, quantity]
        mid_price = (bids[0][0] + asks[0][0]) / 2.0 if (bids[0][0] + asks[0][0]) > 0 else 1.0

        features = []
        # Bids (offsets from mid, normalized qty)
        for p, q in bids:
            features.append([(p - mid_price) / mid_price, q])
        # Asks (offsets from mid, normalized qty)
        for p, q in asks:
            features.append([(p - mid_price) / mid_price, q])

        # Tensor: (1, seq=depth_levels*2, dim=2)
        return torch.tensor([features], dtype=torch.float32)

    def splice_order_flow(self, trade_history, window_size=20):
        """
        Maps trade sequence into a geometric order flow manifold.
        """
        if not trade_history:
            return torch.zeros(1, window_size, 2)

        # Take the last window_size trades
        trades = trade_history[-window_size:]
        while len(trades) < window_size:
            trades.insert(0, trades[0])

        # Normalize: [price_change, size]
        base_price = trades[0][0] if trades[0][0] != 0 else 1.0
        features = []
        for p, q in trades:
            features.append([(p - base_price) / base_price, q])

        return torch.tensor([features], dtype=torch.float32)
