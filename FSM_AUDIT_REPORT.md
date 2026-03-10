# FSM Audit Report

**Scope:** All skills under `packages/valory/skills/`:
- `registration_abci` (library)
- `reset_pause_abci` (library)
- `transaction_settlement_abci` (library)
- `termination_abci` (library)
- `liquidity_trader_abci` (custom/production)
- `optimus_abci` (composed app)

**Date:** 2026-03-10

---

## CLI Tool Results (post-fix)

| Tool | Result |
|------|--------|
| `autonomy analyse fsm-specs` | PASSED |
| `autonomy analyse handlers` | PASSED (with `-i funds_manager`; see Notes) |
| `autonomy analyse dialogues` | PASSED (with `-i funds_manager`; see Notes) |
| `autonomy analyse docstrings` | PASSED (fixed) |

---

## Critical Findings

No findings.

---

## High Findings

### H3: Unclosed `requests.Session()` in CoinGecko Client -- FIXED
- **File:** `packages/valory/skills/liquidity_trader_abci/models.py:239`
- **Issue:** A `requests.Session()` is created but never closed. In a long-running agent service, this leaks TCP connections and can cause socket exhaustion.
- **Fix applied:** Wrapped session usage in `with session:` context manager to ensure proper cleanup.

---

## Medium Findings

### M2: Unreachable `WITHDRAWAL_COMPLETED` Transition in `WithdrawFundsRound` -- FIXED
- **Files:** `states/withdraw_funds.py:45`, `rounds.py:166`, `fsm_specification.yaml:120`, `optimus_abci/fsm_specification.yaml:186`
- **Issue:** The round defined `withdrawal_completed_event = Event.WITHDRAWAL_COMPLETED` and the transition function mapped `Event.WITHDRAWAL_COMPLETED -> FetchStrategiesRound`, but `end_block()` only ever returned `Event.DONE` or `Event.NO_MAJORITY`. The `WITHDRAWAL_COMPLETED` transition was dead — it could never fire from `WithdrawFundsRound`. (Note: `Event.WITHDRAWAL_COMPLETED` IS correctly used by `PostTxSettlementRound` — only the `WithdrawFundsRound` entry was dead.)
- **Fix applied:** Removed the dead `withdrawal_completed_event` attribute, the unreachable transition from `WithdrawFundsRound` in `rounds.py`, and the corresponding entries in both `fsm_specification.yaml` files.

### Dependency Declaration Issues -- FIXED (refactored)
- **Files:** `packages/valory/customs/{uniswap,balancer,velodrome}_pools_search/*.py`, `packages/valory/skills/liquidity_trader_abci/behaviours/evaluate_strategy.py`
- **Issue:** All 3 custom packages imported `packages.valory.connections.x402` at the top level but couldn't declare it as a dependency (AEA framework's `CustomComponentConfig` doesn't support package dependencies).
- **Fix applied:** Moved x402 session creation into the skill layer (`evaluate_strategy.py`) which properly declares the x402 connection dependency. Custom packages now receive a pre-built `x402_session` object instead of importing x402 directly. This eliminates the undeclared dependency and allows `autonomy analyse handlers` and `dialogues` to pass.

---

## Low Findings

### L3: Docstring Drift in `LiquidityTraderAbciApp` -- FIXED
- **File:** `packages/valory/skills/liquidity_trader_abci/rounds.py:72`
- **Issue:** The class docstring was minimal and didn't document the FSM structure.
- **Fix applied:** Added properly formatted docstring with Initial round, Initial states, Transition states, Final states, and Timeouts sections. Verified via `autonomy analyse docstrings`.

---

## Composition Chain Analysis

The `OptimusAbciApp` (`packages/valory/skills/optimus_abci/composition.py`) chains 4 apps:

```
RegistrationAbci -> LiquidityTraderAbci -> TransactionSettlementAbci -> ResetPauseAbci
```

| Check | Status |
|-------|--------|
| All 11 transition mappings valid | PASS |
| Final states map to valid initial states | PASS |
| `cross_period_persisted_keys` coverage | PASS (5 keys from LiquidityTraderAbci) |
| No conflicting round IDs | PASS |
| Shared event names (DONE, ROUND_TIMEOUT, NO_MAJORITY) | Expected — framework handles via `chain()` merge |

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| Critical | 0 | - |
| High | 1 | 1 |
| Medium | 2 | 2 |
| Low | 1 | 1 |

## Notes

- **False positive excluded:** `reset_pause_abci/models.py` mutating `ResetPauseAbciApp.event_to_timeout` in `SharedState.setup()` is the standard framework pattern for runtime timeout parameterization, not a C1 shared mutable reference bug.
- **Library skill conventions respected:** Unused `ROUND_TIMEOUT` enum members in library skills were not flagged per the audit skill's guidance.
- **`funds_manager` skip:** The `funds_manager` skill is a non-ABCI utility skill with no `handlers.py` file. It must be skipped via `-i funds_manager` when running `handlers` and `dialogues` checks. This is a pre-existing issue outside the audit scope.
- **x402 dependency fix:** Rather than working around the AEA framework limitation (lazy imports), we properly refactored the architecture: the skill layer now creates x402 sessions and passes them to custom packages, eliminating the undeclared cross-package import.
- All 5 FSM skills pass `fsm-specs` and `docstrings` validation.
- All skills pass `handlers` and `dialogues` validation (with `funds_manager` excluded).
