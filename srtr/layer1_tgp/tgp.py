import torch
import torch.nn as nn

class TopologicalGaugePerception(nn.Module):
    """
    Layer 1: Topological Gauge Perception (TGP)
    Maps continuous streams of data into a fiber bundle representation.
    Enforces semantic integrity via Gauge Fields and Covariant Derivatives.
    """
    def __init__(self, input_dim, hidden_dim, coupling_constant=0.1):
        super(TopologicalGaugePerception, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g = coupling_constant

        # Mapping to semantic fields
        self.phi = nn.Linear(input_dim, hidden_dim)
        # Gauge field generator (A_mu)
        self.gauge_gen = nn.Linear(input_dim, hidden_dim * hidden_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: covariant derivative fields
        """
        batch_size, seq_len, _ = x.shape

        # Semantic field Psi
        psi = self.phi(x) # (batch, seq_len, hidden_dim)

        # Gauge field A_mu (simplified as a transformation matrix per token)
        a_mu = self.gauge_gen(x).view(batch_size, seq_len, self.hidden_dim, self.hidden_dim)

        # Covariant Derivative: D_mu Psi = partial_mu Psi - i * g * A_mu * Psi
        # For discrete sequences, partial_mu Psi is approximated by temporal difference
        psi_diff = torch.zeros_like(psi)
        psi_diff[:, 1:, :] = psi[:, 1:, :] - psi[:, :-1, :]

        # Interaction term: i * g * A_mu * Psi (treating i as a phase shift or complex rotation in real space)
        # Here we model the interaction as a matrix multiplication
        interaction = torch.matmul(a_mu, psi.unsqueeze(-1)).squeeze(-1)

        covariant_derivative = psi_diff - self.g * interaction

        return psi, covariant_derivative

    def ingest_stream_chunk(self, history, window_size=5):
        """
        Helper to prepare streaming data chunks for the forward pass.
        Ensures a sliding window of recent telemetry and applies normalization.
        """
        if len(history) < window_size:
            padding = [history[0]] * (window_size - len(history))
            history = padding + history

        chunk = history[-window_size:]

        # Normalization: Map prices to relative changes (drift-invariant)
        base = chunk[0] if chunk[0] != 0 else 1.0
        normalized = [(h - base) / base for h in chunk]

        return torch.tensor([[ [h] for h in normalized ]], dtype=torch.float32)
