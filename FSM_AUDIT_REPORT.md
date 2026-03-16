# FSM Audit Report

**Scope:** All skills under `packages/valory/skills/`:
- `registration_abci` (library)
- `reset_pause_abci` (library)
- `transaction_settlement_abci` (library)
- `termination_abci` (library)
- `liquidity_trader_abci` (custom/production)
- `optimus_abci` (composed app)

**Initial audit:** 2026-03-10
**Re-audit:** 2026-03-16

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

### H3b: Missing timeout on `session.get()` in CoinGecko Client -- OPEN (2026-03-16)
- **File:** `packages/valory/skills/liquidity_trader_abci/models.py:243`
- **Issue:** While the session leak was fixed (H3 above), the `session.get(url, headers=headers)` call still has no `timeout` parameter. If the CoinGecko server is unresponsive, this blocks the agent indefinitely.
- **Code:**
  ```python
  with session:
      response = session.get(url, headers=headers)
  ```
- **Suggested fix:**
  ```python
  with session:
      response = session.get(url, headers=headers, timeout=30)
  ```

---

## Medium Findings

### M2: Unreachable `WITHDRAWAL_COMPLETED` Transition in `WithdrawFundsRound` -- FIXED
- **Files:** `states/withdraw_funds.py:45`, `rounds.py:166`, `fsm_specification.yaml:120`, `optimus_abci/fsm_specification.yaml:186`
- **Issue:** The round defined `withdrawal_completed_event = Event.WITHDRAWAL_COMPLETED` and the transition function mapped `Event.WITHDRAWAL_COMPLETED -> FetchStrategiesRound`, but `end_block()` only ever returned `Event.DONE` or `Event.NO_MAJORITY`. The `WITHDRAWAL_COMPLETED` transition was dead — it could never fire from `WithdrawFundsRound`. (Note: `Event.WITHDRAWAL_COMPLETED` IS correctly used by `PostTxSettlementRound` — only the `WithdrawFundsRound` entry was dead.)
- **Fix applied:** Removed the dead `withdrawal_completed_event` attribute, the unreachable transition from `WithdrawFundsRound` in `rounds.py`, and the corresponding entries in both `fsm_specification.yaml` files.

### M-NEW-1: Missing `RegistrationEvent` in timeout configuration -- OPEN (2026-03-16)
- **File:** `packages/valory/skills/optimus_abci/models.py:48-52,94-97`
- **Issue:** `SharedState.setup()` configures `ROUND_TIMEOUT` overrides for 3 of the 4 composed apps but omits `RegistrationEvent` from `AgentRegistrationAbciApp`. The `EventType` union and `events` tuple are incomplete.
- **Code:**
  ```python
  EventType = Union[
      Type[LiquidityTraderEvent],
      Type[TransactionSettlementEvent],
      Type[ResetPauseEvent],
  ]
  # ...
  events = (LiquidityTraderEvent, TransactionSettlementEvent, ResetPauseEvent)
  ```
- **Suggested fix:** If `registration_abci` uses `ROUND_TIMEOUT` in the composed transition function, add:
  ```python
  from packages.valory.skills.registration_abci.rounds import Event as RegistrationEvent

  EventType = Union[
      Type[RegistrationEvent],
      Type[LiquidityTraderEvent],
      Type[TransactionSettlementEvent],
      Type[ResetPauseEvent],
  ]
  # ...
  events = (RegistrationEvent, LiquidityTraderEvent, TransactionSettlementEvent, ResetPauseEvent)
  ```

### M-NEW-2: Wrong type annotation on `most_voted_tx_hash` -- OPEN (2026-03-16)
- **File:** `packages/valory/skills/liquidity_trader_abci/states/base.py:79`
- **Issue:** Return type is `Optional[float]` but this property returns a transaction hash (a hex string). The payload definition uses `Optional[str]`.
- **Code:**
  ```python
  @property
  def most_voted_tx_hash(self) -> Optional[float]:
      """Get the token most_voted_tx_hash."""
      return self.db.get("most_voted_tx_hash", None)
  ```
- **Suggested fix:**
  ```python
  @property
  def most_voted_tx_hash(self) -> Optional[str]:
      """Get the most_voted_tx_hash."""
      return self.db.get("most_voted_tx_hash", None)
  ```

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

| Severity | Count | Fixed | Open |
|----------|-------|-------|------|
| Critical | 0 | - | 0 |
| High | 2 | 1 | 1 (H3b) |
| Medium | 4 | 2 | 2 (M-NEW-1, M-NEW-2) |
| Low | 1 | 1 | 0 |

## Notes

- **False positive excluded:** `reset_pause_abci/models.py` mutating `ResetPauseAbciApp.event_to_timeout` in `SharedState.setup()` is the standard framework pattern for runtime timeout parameterization, not a C1 shared mutable reference bug.
- **Library skill conventions respected:** Unused `ROUND_TIMEOUT` enum members in library skills were not flagged per the audit skill's guidance.
- **`funds_manager` skip:** The `funds_manager` skill is a non-ABCI utility skill with no `handlers.py` file. It must be skipped via `-i funds_manager` when running `handlers` and `dialogues` checks. This is a pre-existing issue outside the audit scope.
- **x402 dependency fix:** Rather than working around the AEA framework limitation (lazy imports), we properly refactored the architecture: the skill layer now creates x402 sessions and passes them to custom packages, eliminating the undeclared cross-package import.
- All 5 FSM skills pass `fsm-specs` and `docstrings` validation.
- All skills pass `handlers` and `dialogues` validation (with `funds_manager` excluded).
- **Re-audit (2026-03-16):** All previously fixed findings confirmed resolved. Three new findings added (H3b, M-NEW-1, M-NEW-2). CLI tools could not be re-run (missing `aea` module in environment). C1-C4 re-checked clean. All test checks (T1-T6) passed. No shared mutable references, no operator precedence bugs, no dead timeouts found.
