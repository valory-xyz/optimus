[WD] Safe API Proxy Routing and Per-Cycle Caching — Detailed Plan

Created: 2026-06-03
Owner: Divya Nautiyal
Status: WIP, awaiting implementation
Parent doc: [WD] Basius Launch
Sibling: safe_api_optimization.md (PRs 1.1 to 1.4, already shipped)

Purpose

This is the implementation plan for the next PR in the Safe-API workstream. The previous PRs (1.1 to 1.4) brought the per-cycle Safe API call count down via caching and proxy-safe pagination. This PR routes the agent at our reverse proxy at rpc-gate.autonolas.tech and adds the remaining caching needed to fit a 100 agent fleet inside the proxy quota. Production team has the proxy running with one shared API key behind it. Mode is excluded because it does not have a Safe Transaction Service.

Context

The production team has stood up a reverse proxy in front of api.safe.global at https://rpc-gate.autonolas.tech/safe/tx-service. One API key sits inside the proxy. Optimism uses slug "oeth" and Base uses slug "base" (Safe's new chain identifiers). The quota on the key gives us a fixed monthly call budget per chain. With 44 Optimus agents on Optimism today plus ~20 Basius agents planned on Base, running ~12 hours a day on average, we need to land inside that budget while keeping the agent's view of its own state fresh.

Math snapshot (50 to 60 agents, 12h/day, 6 min FSM cycle)

Per agent per day, after this PR:
  Balances:           ~8 calls (3h TTL plus on-tx invalidate plus RPC change-detect invalidate)
  Incoming transfers: ~5 calls (3h TTL plus RPC change-detect invalidate)
  Outgoing transfers: ~7 calls (3h TTL plus on-tx invalidate)
  Total Safe API:     ~20 calls per agent per day

50 agents: ~30k/month. ~40% headroom against a 50k budget. Comfortable for the current fleet, room for growth to ~60. At 100 agents this approach is tight (~60k/month) and would need a longer TTL or a second key.

The five changes

1. Route Safe API through the proxy.

Today the three Optimism fetchers in fetch_strategies.py talk to safe-transaction-optimism.safe.global directly. After this PR they go through the proxy. Two new env-overridable params:
  safe_api_base_url: default https://rpc-gate.autonolas.tech/safe/tx-service
  safe_api_chain_slugs: default {"optimism": "oeth", "base": "base"}

Each fetcher constructs its URL as f"{base}/{slugs[chain]}/api/{version}/safes/{addr}/{endpoint}/". Both params are env-overridable so production can swap proxies, change slugs, or disable the proxy entirely without redeploying.

2. Stronger 429 handling in _request_with_retries.

If the proxy returns 429 and the response has a Retry-After header, honour it. Cap the honoured wait at 30s so we don't blow the FSM round budget.

If Retry-After is absent, retry up to 4 attempts with exponential backoff plus jitter. Sleep schedule: ~5s, ~10s, ~15s, each plus a small random offset (so 50 agents that all 429 at the same wall-clock second don't all retry at the same second). Total wait budget ~30s.

After max retries, return the existing fetch-failed signal. The next FetchStrategies cycle will retry. The existing fetch_failed handling already skips persistence and returns the previous-cycle data.

3. Use Safe API's USD value field instead of re-fetching from CoinGecko.

The current code in _calculate_safe_balances_value (fetch_strategies.py:2265) ignores the fiat_balance and fiat_conversion fields Safe API returns and instead calls _fetch_token_price (CoinGecko, paid via x402) for each token to compute USD value. After this PR we use Safe's USD field when present and fall back to CoinGecko only when Safe returns null (exotic tokens). Cuts CoinGecko spend on the whitelisted-token path.

4. Safe balances cache with 3h TTL + per-cycle RPC change-detect.

Every FetchStrategies cycle, the agent reads the safe's balances over RPC via a multicall on balanceOf for each whitelisted token plus eth_getBalance for native ETH. This is cheap, uses the agent's own RPC quota, and doesn't touch the Safe budget. Multicall is already used in this codebase for portfolio computation so no new infrastructure.

Stores the last-known balances in kv keyed by chain (last_safe_balances_<chain>). Compares this cycle's reads to last-known:

  Cached + no change detected: serve cached Safe API result, no Safe API call.
  Cached + any whitelisted balance went up (external inflow): invalidate balances cache and incoming-transfers cache, update last-known. Next cycle re-fetches Safe API and ROI plus balances reflect the new state.
  TTL expired (3h since last refresh): refresh Safe API regardless. Safety floor that catches non-whitelisted token arrivals that the RPC change-detect would miss (we don't query their balanceOf).

The RPC change-detect is the primary freshness mechanism. The 3h TTL is a safety floor.

5. Invalidate caches after every successful agent transaction.

Every transaction the agent makes (swap, enter pool, exit pool, withdrawal, approval, reversion) goes through PostTxSettlementRound. We hook into that round and, on successful settlement, reset the balances cache timestamp and outgoing-transfers cache timestamp to 0. Next FetchStrategies cycle re-fetches and the agent sees its own action immediately.

By hooking into PostTxSettlement we catch every tx type in one place. No risk of forgetting an action.

End-state freshness

  Event                                      Visible after
  Agent swap / enter / exit / withdrawal     ~6 min (one FSM cycle, on-tx invalidate)
  External funding to whitelisted token      ~6 min (RPC change-detect triggers refresh)
  External funding to non-whitelisted token  Up to 3h (TTL safety floor catches it)
  Nothing changed                            Cache served, no Safe API calls until TTL expires

For ROI display purposes this means an agent's own actions and any whitelisted token inflow show up next cycle. Only exotic non-whitelisted token arrivals can sit stale for up to 3h.

Out of scope

  Mode chain. No Safe Transaction Service available. We currently use a Mode block-explorer API there and that path stays unchanged.
  Long-term ROI off Safe API entirely (eth_getLogs based). Plan doc appendix already filed this as a post-launch follow-up; not in scope for this PR.
  Second API key or higher Safe plan. Production team can revisit if fleet grows past ~60 agents on the current setup.
  Webhook subscription for instant external-funding detection. The 6-min cycle delay via RPC change-detect is sufficient for ROI display.

Files touched

  packages/valory/skills/liquidity_trader_abci/models.py
    Add safe_api_base_url, safe_api_chain_slugs, BALANCES_CACHE_TTL_SECONDS, plus expose existing WITHDRAWAL_CACHE_TTL_SECONDS and INITIAL_INVESTMENT_CACHE_TTL_SECONDS as env-overridable params if not already.

  packages/valory/agents/optimus/aea-config.yaml
  packages/valory/services/optimus/service.yaml
  packages/valory/skills/liquidity_trader_abci/skill.yaml
    Wire the new params and their defaults.

  packages/valory/skills/liquidity_trader_abci/behaviours/fetch_strategies.py
    Construct fetcher URLs from base + slug. Add balances cache with 3h TTL. Add RPC multicall change-detect at top of cycle. Use Safe's USD field with CoinGecko fallback for null.

  packages/valory/skills/liquidity_trader_abci/behaviours/post_tx_settlement.py
    Invalidate balances + outgoing-transfers caches on successful settlement.

  packages/valory/skills/liquidity_trader_abci/behaviours/base.py
    Strengthen _request_with_retries: honour Retry-After (capped at 30s), exponential backoff with jitter as fallback, up to 4 attempts.

  packages/valory/skills/liquidity_trader_abci/tests/behaviours/test_fetch_strategies.py
    Tests for the new URL construction, the balances cache, the RPC change-detect logic (detect external inflow, ignore expected balance changes after a known tx), and the Safe USD field usage with fallback.

  packages/valory/skills/liquidity_trader_abci/tests/behaviours/test_post_tx_settlement.py
    Tests for the cache-invalidation hook.

  packages/valory/skills/liquidity_trader_abci/tests/behaviours/test_base.py (or wherever _request_with_retries tests live)
    Tests for Retry-After handling, jitter shape, max-attempt cap.

Implementation order

  1. Models, yaml wiring, env defaults. Verify no existing behaviour breaks.
  2. URL construction switch in fetcher functions. Run tests against the proxy.
  3. Strengthen _request_with_retries.
  4. Balances cache and RPC change-detect logic.
  5. PostTxSettlement invalidation hook.
  6. Use Safe USD field with CoinGecko fallback.
  7. Tests for every step above.
  8. autonomy packages lock, full lint suite, run local FSM test if possible.

PR description framing

  Title: route Safe API through proxy and add balances caching
  No mention of API plans, tiers, quotas, or cost in commit messages, code comments, or PR description. Comments framed around "routes Safe API through our infra" and "reduces per-cycle Safe API calls so the agent stays inside the configured budget".

Test plan

Unit tests (run in the test suite, fail the PR if broken):

  URL construction
    Given base_url + slug map + chain, fetcher builds f"{base}/{slugs[chain]}/api/v1/safes/{addr}/..." for v1 endpoints and same with /api/v2/ for balances.
    A chain missing from the slug map raises a clear error (not a silent KeyError).
    Env override of safe_api_base_url and safe_api_chain_slugs is picked up.

  _request_with_retries 429 handling
    Response with Retry-After header in seconds: agent sleeps exactly that long (clamped to 30s upper bound).
    Response with Retry-After as an HTTP date: parsed correctly.
    Response with no Retry-After: agent retries 4 times with backoff schedule ~5s, ~10s, ~15s (plus jitter, asserted as a range not an exact value).
    After max retries the existing fetch-failed signal is returned.
    Jitter: across two simulated retry runs from the same state the actual sleeps differ (not zero-jitter).

  Balances cache
    Cold cache: fetcher hits Safe API, stores result with current timestamp.
    Cache age < 3h, no RPC change detected, no tx invalidation: serves cached, no Safe API call.
    Cache age >= 3h: refreshes regardless.
    Stored shape matches what _calculate_safe_balances_value expects to consume (no migration shape issues).

  RPC change-detect
    Same balances as last-known: cache served, no invalidation.
    One whitelisted token balance went up: balances cache + incoming-transfers cache both get their timestamps reset to 0 in kv. Last-known balances updated.
    One token balance went down without a recent agent tx: log a warning but do not invalidate (anomaly).
    One token balance went down with a recent agent tx: treated as expected, no warning.

  Safe USD field usage
    Token with non-null fiat_balance: USD value taken directly from Safe response, no _fetch_token_price call.
    Token with null fiat_balance: falls back to _fetch_token_price (CoinGecko).
    Mix of present and null: each token routed correctly.

  PostTxSettlement invalidation
    On successful tx settlement: balances cache timestamp + outgoing-transfers cache timestamp both reset to 0 in kv.
    On unsuccessful settlement (already failed before this hook): no invalidation.
    Invalidation fires for every tx submitter type (swap, enter, exit, withdrawal, reversion, approval). One parametrized test covering each.

  Integration-style test in test_fetch_strategies
    Drive two consecutive FetchStrategies cycles against mocked Safe API + mocked multicall.
    Cycle 1: cold, makes 3 Safe API calls (balances + incoming + outgoing).
    Cycle 2 with no balance change and no tx: makes 0 Safe API calls (RPC multicall only).
    Cycle 2 after a simulated external USDC inflow on the multicall mock: balances + incoming-transfers caches are refreshed.

Live verification after merge (not automated, manual):

  Deploy the updated package to a test agent on Optimism with the existing funded safe (0x5a4B31942d37d564e5cEf4C82340E43fe66686b2).
  Watch agent log for the first two FetchStrategies cycles:
    Cycle 1: confirm URLs hit rpc-gate.autonolas.tech, not safe-transaction-optimism.safe.global.
    Cycle 2: confirm "balances cache hit" / "incoming-transfers cache hit" / "outgoing-transfers cache hit" log lines and zero Safe API calls.
  Submit one bridge or swap from Pearl. Confirm the next FetchStrategies cycle re-fetches balances and outgoing (cache-miss log line within ~6 min of the tx).
  Manually send a small amount of USDC to the agent safe from another address. Confirm next FetchStrategies cycle logs "external inflow detected" and re-fetches balances + incoming.
  Watch rate-limit headers in the agent log over an hour. Confirm x-ratelimit-remaining decreases steadily, not in bursts.

Approval and tracking

  Approved by: pending
  Implementation start: pending
  Target PR: to be created from hack/launch-base
