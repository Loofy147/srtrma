# SRTR Agent Operating Manual

Welcome, Agent. You are operating the **Self-Regenerating Topological-Relational (SRTR)** Engine. This system is designed for autonomous, production-grade execution on live financial mainnets.

## 🧠 Core Directives

1.  **Preserve State Parity:** Every modification to the engine cycle or adapter synthesis MUST be validated against the SHA-256 `StateAuditor`. Zero state-loss during hot-swaps is a non-negotiable architectural invariant.
2.  **Enforce Idempotency:** All egress dispatches to external APIs MUST utilize the "Intent-based Idempotency" pattern. Never hash transmission payloads directly; always hash the logical intent to prevent race conditions during retries.
3.  **Respect the Gauge Field:** Layer 1 anomalies (Cohomology Drift) trigger autonomous self-regeneration. When debugging, distinguish between "Structural Drift" (schema changes) and "Stochastic Noise" (standard market volatility).

## 🛠️ Development Conventions

-   **Async-Native:** The engine operates on an asynchronous event loop. Ensure all network I/O utilizes `aiohttp` and correctly awaits the `ExecutionMonad`.
-   **Monadic Boundaries:** All external side effects (API calls) MUST be wrapped in the `ExecutionMonad`. This provides safety boundaries for timeouts, budget caps, and error propagation.
-   **AST Hydration:** When defining templates in `srtr/utils/templates.py`, escape curly braces (`{{ }}`) to allow for `.format()` hydration.

## 🛡️ Safety Procedures

-   **Regeneration Rollback:** If a hot-swap fails a parity audit, the system automatically rolls back to the previous stable adapter. Do not bypass this check.
-   **Dry Run First:** Always verify architectural changes using `scripts/mainnet_alpha_anchor.py` in dry-run mode before enabling live execution.
-   **Logging:** Use the structured `SRTR` logger. Never use print statements in core layers.

## 🧪 Verification Commands

Before any PR submission, the following check MUST pass:
```bash
python -m unittest discover tests
```

Critical Invariants to check:
- [ ] SHA-256 Parity Lock (test_audit.py)
- [ ] Intent Idempotency (test_egress.py)
- [ ] Stream Resilience (test_streaming.py)
