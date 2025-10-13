# Protocol Test Audit Report (Optimus)

**Repository:** `https://github.com/valory-xyz/optimus`  
**Branch:** `audit/integration-tests`
**commit:** `2caee3f95bb8caffb2e52775436adb823ae0aec8`

Reviewed files:
- `https://github.com/valory-xyz/optimus/tree/audit/integration-tests/tests/integration/protocols`

## 1) Executive Summary

Current tests labeled as “integration” are mock-driven and don’t hit real EVM paths. They validate high-level behavior but miss:
- Rounding/overflow realities in AMM math (Δx, Δy, L, proportionality).
- ABI layout mismatches & revert data.
- Non-standard ERC-20 (included USDT) behavior, events, gas & deadline semantics, L2 specifics.

**Action:** keep mocks for unit tests; add fork-based integration tests with Anvil.

## 2) Mathematical Background — What to Test Explicitly. Low priority

### 2.1 Uniswap v3

Let \( P = (\sqrt{P})^2/2^{192} \) and for a price move from \(P_a\) to \(P_b\):
\[ \Delta x = \frac{L(\sqrt{P_b} - \sqrt{P_a})}{\sqrt{P_b}\sqrt{P_a}} \quad ; \quad \Delta y = L(\sqrt{P_b} - \sqrt{P_a}) \]

**Add property tests for:**
- Rounding direction in `LiquidityAmounts.getAmountsForLiquidity` (up vs down).
- Boundary cases: `tickLower == currentTick`, `tickUpper == currentTick + 1`.
- Overflow safety for `sqrtP^2` and Q96 scaling.

### 2.2 Balancer (Weighted/Stable/Composable)

For proportional joins/exits:
\[ \frac{\text{BPT}_{in}}{\text{BPT}_{supply}} \approx \frac{\text{amount}_i}{\text{balance}_i} \]

**Add checks:**
- Token sorting alignment (`assets` ↔ `amounts`) per `getPoolTokens()`.
- Compare `queryJoin` vs `joinPool` results within rounding tolerance.
- Verify relevant events (`PoolBalanceChanged`).

### 2.3 Velodrome

**Strengthen tests:**
- Assert structure: `reserve0`, `reserve1`, `stable`, `totalSupply`, etc.
- Validate numeric sanity (positivity, expected ranges).

---

## 3) Methodology Issues & Inconsistencies (What to Fix). High priority
You can't emulate all the behavior of real protocols using mock-contracts. At best, you can only emulate "successful" ("happy") paths. That's a dead end. <br>

1. **Mock ABIs drift from real ABIs** (e.g., Uniswap PM `positions()` length/order). - **I highly recommend not trying to create mocks for such complex DeFi/protocol**.
2. **Checksum** is faked via `addr.lower()` — replace with EIP‑55 (`Web3.to_checksum_address`).  
3. **`tx_hash` type** inconsistent (bytes vs hex string) — standardize to `"0x..."` string for sent tx, none for encode only.  
4. **Vault return shape** tuple vs dict — unify to named dict `{tokens, balances, lastChangeBlock}`.  
5. **No event assertions** — always assert events & args for integration tests.  
6. **No L2 specifics** — add fork tests per network (Optimism/Base/Arbitrum) with fixed block numbers.

---

## 4) Target Architecture

| Layer | Purpose | Tooling |
|---|---|---|
| Unit / Component | Pure logic & adapters (mocks) | `pytest`, mocks only |
| Integration | Real contracts on fork | `anvil` + `pytest` + `web3.py` |
| E2E Scenarios | Cross‑protocol flows | Same as Integration + fixtures |

Key techniques: fixed fork block, `evm_snapshot/evm_revert`, impersonation, event assertions, deterministic deadlines.

---

## 5) Fork Example (Python + Anvil)

### 5.1 Start a forked node

```bash
# Ethereum mainnet example (replace YOUR_KEY)
anvil --fork-url https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY       --fork-block-number 20850000       --chain-id 1 --port 8545 --no-rate-limit
```
I highly recommend using archive nodes (quicknode as example).

### 5.2 Install Python deps

```bash
pip install pytest web3
```

### 5.3 Place the two files next to your tests

- `conftest.py` — shared web3 + snapshot fixtures  
- `test_uniswap_v3_fork_e2e.py` — end‑to‑end mint → decrease on Uniswap v3

(Download links are below.)

### 5.4 Run

```bash
pytest -q test_uniswap_v3_fork_e2e.py
```

**What it does:** connects to the fork, impersonates a whale, approves DAI/WETH to the NonfungiblePositionManager, mints a narrow v3 position around current tick, asserts `IncreaseLiquidity` event and positive liquidity, then decreases half the liquidity.

---

## 6) Minimal Checklist for Real Integration Tests

- ✅ Use fixed fork block & `evm_snapshot`/`evm_revert` between tests.  
- ✅ Always assert emitted events and argument values.  
- ✅ Validate `assets`/`amounts` ordering for Balancer joins/exits.  
- ✅ Cover fee-on-transfer / no-return ERC-20 edge cases.  
- ✅ Standardize adapters: `encode` returns calldata only; `send` returns `{tx_hash, receipt}` as hex string + dict.  
- ✅ Replace fake checksum with `Web3.to_checksum_address`.  
- ✅ Add L2 forks (Optimism/Base/Arbitrum) with per‑network specifics (gas, base fee, deadlines).

---

## 7) Next Steps (Suggested PR scope)

1. Unify ABI adapters & return shapes across Balancer/Uniswap/Velodrome.  
2. Introduce `tests/integration_fork/` with fixtures and at least one per‑protocol fork test.  
3. Update CI: separate `unit` (mocks) vs `fork` jobs.  
4. Add real event assertions & revert reason checks.
