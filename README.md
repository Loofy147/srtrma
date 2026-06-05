# SRTR Engine: Sovereign Mainnet Anchor

The **Self-Regenerating Topological-Relational (SRTR)** Engine is a three-tier hierarchical framework designed for high-frequency, autonomous decision-making in high-entropy environments. It combines geometric topology, machine learning, and deterministic execution to create a self-healing system capable of live financial execution.

## 🏗️ Architecture

### Layer 1: Topological Gauge Perception (TGP)
Maps real-time telemetry (ingress) into a fiber bundle representation. It enforces structural integrity using **Gauge Fields** and **Covariant Derivatives**. Structural shifts are detected as **Cohomology Drift** (^n \neq 0$).

### Layer 2: Relational Meta-Controller (RMC)
Utilizes **Graph Neural Networks (GNN)** and **Hidden Markov Models (HMM)** to classify environmental regimes ($) and coordinate relational sub-goals.

### Layer 3: Deterministic Constrained Execution (DCE)
Implements risk-bounded instructions via an **ExecutionMonad**. It utilizes **Ornstein-Uhlenbeck (Mean Reversion)** processes for action selection and enforces **Intent-based Idempotency** and **State Parity** via cryptographic auditing.

---

## 🛡️ Production Hardening: The Mainnet Anchor

The engine is currently deployed in the **Vector Alpha** (Financial Gateway) domain, optimized for live quantitative strategies.

### Key Safety Features:
- **State Parity Auditing:** Uses SHA-256 deterministic hashing to ensure zero state-loss (e.g., active orders) during runtime adapter hot-swaps.
- **Intent-based Idempotency:** Blocks duplicate external API dispatches by cryptographically tying transmission signatures to logical execution intents.
- **Self-Regeneration Loop:** Autonomous recovery mechanism that synthesizes and hot-swaps code templates (AST) upon detecting structural anomalies or execution failures.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch
- NumPy
- aiohttp

### 2. Setup
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
```

### 3. Running the Engine
For dry-run production execution (Vector Alpha):
```bash
python scripts/mainnet_alpha_anchor.py
```

---

## 🧪 Testing

The system includes a robust test suite covering all architectural layers and production-hardening features.

```bash
python -m unittest discover tests
```

- `tests/test_audit.py`: Verifies SHA-256 state parity locks.
- `tests/test_streaming.py`: Validates WebSocket ingress resilience.
- `tests/test_egress.py`: Verifies HMAC signatures and intent idempotency.
- `tests/test_vector_alpha.py`: High-fidelity production cycle dry-run.

---

## 📜 License
Sovereign Deployment. Reference individual modules for specific domain constraints.
