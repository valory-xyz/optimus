[WD] Basius Launch

Created: May 26, 2026
Current Version: 1.0.0
Approved for external use by: N/A
Status: WIP
Owner: Divya Nautiyal Dhairya Patel
Contributors: 
Other stakeholders:
Approvers: David Minarsch


Context
An old branch (hack/launch-base) has most of the scaffolding for Optimus on Base — chain configuration, Aerodrome support, the token whitelist, and balance and transfer fetching. The branch is 8 months stale, ships with several temporary hacks left in to keep one debugging session moving, and has no tests. It also uses the same Safe API balance-fetching architecture as the production Optimism agent. SafeGlobal has since introduced rate limits across all its endpoints, so that architecture has to change regardless of Basius — the existing Optimism agent is exposed to the same problem.
Decisions to lock first
Balance-fetch architecture (resolved 27 May). For MVP we keep Safe API as the primary source and proxy it through our infra with an authenticated free-tier key. Production team owns the proxy; agent side only needs to swap the configured base URL. Self-hosting Safe Transaction Service stays on the table for later. PR #269's full re-architecture is deferred. The in-agent fixes captured in workstream 1 bring per-cycle Safe API traffic down by roughly 30x and that, plus the proxy's free-tier ceiling, gives us comfortable headroom without a bigger migration. eth_getLogs-based ROI is the natural next step post-launch.
Allowed slippage for swaps — Currently bumped to 90% as a temporary hack. Production value needs to be agreed (previously 8%). Thin Base stables may need a higher cap, a per-chain value, or a trimmed whitelist.
Funding token scope — On Optimism the agent is funded with USDC initially. Do we keep the USDC-only convention for Basius, or any other token?
Workstreams
Three parallel workstreams. Each task below is sized to be one PR. Intra-workstream ordering is noted where it matters. Cross-workstream dependencies: WS1 PR 1.5 (per-chain Safe API params) is a prerequisite for WS3 since Base needs the new param shape; the proxy itself is a production-team task and should be ready before WS1 PR 1.5 merges so end-to-end testing can use the proxy URL.

1. Safe API call optimization (free-tier compatible)

Goal: bring per-cycle Safe API traffic from O(N transfer-history pages) down to O(1) in steady state. Combined with the proxy on Safe's authenticated free tier, this gives comfortable headroom without the eth_getLogs migration before launch. After these fixes, steady-state cost drops from roughly 12 + 12N calls/hour to 12 + 2 calls/hour on a 5-minute cycle (~30x reduction with a 3-page transfer history). Same fixes apply to the existing Optimism agent.

PR 1.1 — Add cache and early-stop to the USDC withdrawal fetcher. _track_erc20_transfers_optimism today walks the full /transfers/ history on every FetchStrategies cycle. Mirror the existing incoming-transfer pattern: add a withdrawals dict to funding_events.json keyed by date, total_withdrawals plus last_withdrawals_calculated_timestamp in kv_store, and the same consecutive_existing early-stop the sibling fetcher already uses. Day-T cycle 2 onwards then costs 0 withdrawal calls instead of N.

PR 1.2 — Switch transfer dedup from date-keyed to transferId-keyed across all transfer fetchers. Today, once a date is in funding_events.json, any later transfer on the same day is silently dropped at fetch_strategies.py:3890 and equivalents. A 10 AM funding plus a 4 PM top-up loses the top-up forever. Dedup by transferId (or transactionHash + log index as fallback) so same-day events land correctly.

PR 1.3 — Shorten the initial-investment kv cache TTL. Today calculate_initial_investment_value_from_funding_events returns the cached total whenever last_initial_value_calculated_timestamp falls on the same calendar day. Change to a ~30 minute window so genuine same-day events get picked up without holding the cache for 24 hours.

PR 1.4 — Make pagination proxy-safe. Three sites (fetch_strategies.py:4012, 4598, 4746) do transfers_url = response_json.get("next"). That URL is absolute, pointing back at safe-transaction-*.safe.global, so under the proxy page 1 goes through us and page 2 onwards bypasses us with no API key. Strip the host from next and re-prefix with safe_api_v1_url, or paginate by offset ourselves.

PR 1.5 — Wire per-chain Safe API URLs in models.py, skill.yaml, and service.yaml. Add Base. Make both base and v1 URLs env-overridable so production can point them at the proxy. Coord with production team on the proxy URL value outside the PR.

2. Cleanup and other fixes

Goal: remove the Optimus-specific ETH-reversion logic that Basius doesn't need, plus the pre-launch fixes that aren't Base-specific. The ETH-reversion path exists because Optimus was being drip-funded with dust ETH and we built logic to both send it back (revert) and subtract it from ROI. Basius is USDC-funded only, so both pieces of logic come out, and we rely on funding_events.json as the raw record of what was received. Order matters: PRs 2.1 → 2.2 → 2.3 should land in sequence so the ROI calc doesn't break between commits.

PR 2.1 — Remove ETH reversion transaction generation. Strip the FSM path that builds the "send ETH back to the funder" transaction.

PR 2.2 — Remove ETH reversion subtraction from ROI math. Anywhere reversion_amount is netted out of the portfolio total, drop the subtraction. Rely on funding_events.json instead.

PR 2.3 — Delete the ETH reversion fetch and cache layer: _track_eth_transfers_and_reversions, _fetch_outgoing_transfers_until_date_optimism, the ETHER_TRANSFER branch inside _fetch_optimism_transfers_safeglobal, the funding_events.json keys optimism_reversion_info and optimism_reversion_transfer_count, and the period-0 caller _check_and_create_eth_revert_transactions at fetch_strategies.py:148.

PR 2.4 — Fix the cached-opportunity edge case. If the agent fails to invest in a given period, stale opportunity data is left in the cache and trips later cycles. Invalidate the opportunity cache on failure and on period rollover. This is a common workflow across all BabyDegen agents, so the fix benefits the wider family, not just Basius.

PR 2.5 — Re-enable the slippage cap safety check.

PR 2.6 — Re-enable the profitability gate safety check.

3. Base-specific launch wiring

Goal: everything that exists only because we're shipping on Base. PR 3.5 depends on PR 3.4.

PR 3.1 — Verify Aerodrome contract addresses on Base. Compare against current on-chain deployments. The branch is 8 months stale, so any address that moved (upgrade, redeploy, migration) needs to be updated.

3.1 audit findings (2026-06-04)

Verified correct on Base (matches official Aerodrome / Sugar deployments and has bytecode on-chain):
- v2 Router 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43
- Voter 0x16613524e02ad97eDfeF371bC883F2F5d6C480A5
- AERO 0x940181a94A35A4569E4529A3CDfB74e38FD98631
- OLAS 0x54330d28ca3357F294334BDC454a032e7f353416
- WHITELISTED_ASSETS["base"]: USDC, oUSDT, axlUSDC, msUSD, frxUSD — all have code on chain. Whitelist composition itself decided by PR 3.4 calibration.

Bugs to fix in the WS3 PR:

1. Base NFPM placeholder. optimus_abci/skill.yaml:245 sets velodrome_non_fungible_position_manager_contract_addresses["base"] to 0x79e1bEf100000000000000000000000000000000, which has no bytecode on Base. The wrapper skill overrides liquidity_trader_abci, so this is the value used at runtime. Any Base CL pool entry crashes at the first contract call. Replace with the live Aerodrome Slipstream V3 NFPM 0xe1f8cd9AC4e4A65F54f38a5CdAfCA44f6dD68b53 (verified: factory() returns the official Slipstream V3 PoolFactory 0xf8f2eB4940CFE7d13603DDDD87f123820Fc061Ef).

2. NFPM disagreement across skill files. liquidity_trader_abci/skill.yaml:324 also defines the same param with Base = 0x827922686190790b37229fd06084350E74485b72 (the older Slipstream V1 NFPM, factory() returns 0x5e7BB104d84c7CB9B682AaC2F3d509f5F406809A). Wrapper wins at runtime so today this value is dead, but two skill files disagreeing on the same key is its own bug. Align both files on the V3 NFPM from fix 1.

3. Stale Sugar / Slipstream-helper contracts. Current addresses have bytecode but are older deployments than what velodrome-finance/sugar/deployments/base.env ships today:
   - LpSugar Base (velodrome_pools_search.py:75): ours 0x27fc745390d1f4BaF8D184FBd97748340f786634, latest 0x69dD9db6d8f8E7d83887A704f447b1a584b599A1.
   - RewardsSugar Base (liquidity_trader_abci/skill.yaml:330 + velodrome_pools_search.py:82): ours 0xD4aD2EeeB3314d54212A92f4cBBE684195dEfe3E, latest 0x1b121EfDaF4ABb8785a315C51D29BCE0552A7678.
   - Slipstream helper Base (liquidity_trader_abci/skill.yaml:328): ours 0x0AD09A66af0154a84e86F761313d02d0abB6edd5, latest 0x9c62ab10577fB3C20A22E231b7703Ed6D456CC7a.
   Bump alongside fix 1, with an ABI compatibility check before flipping in case the V3 tuple layouts changed.

PR 3.2 — Register Aerodrome as a recognised protocol in the agent's strategy map. Wire available_strategies["base"] to include Aerodrome strategies.

PR 3.3 — Re-enable the Aerodrome slippage protection safety check (the fourth commented-out check, Aerodrome-specific so it lives in this workstream).

PR 3.4 — Calibrate the whitelist on thin Base stables. Run real-size LiFi quotes for axlUSDC, msUSD, and frxUSD, observe actual slippage at realistic position sizes, decide which to keep vs trim. Output: updated whitelist config.

3.4 calibration findings (2026-06-04)

Ran USDC <-> {axlUSDC, msUSD, frxUSD} via LiFi /v1/quote on Base at $100, $500, $1000, $5000 from agent safe 0xA0fF35Bfbd3C42E3aFE29255742C7558498f5544. Aggregator routes (SushiSwap aggregator, Nordstern, Fly, Kyberswap) — same routing layer the agent uses at trade time, so observed slippage is representative of what production will see.

Worst single leg, across all sizes:
- axlUSDC: 0.31% (forward at $5000)
- msUSD: 0.42% (reverse at every size — asymmetric, cheap in ~0.10%, expensive out ~0.42%)
- frxUSD: 0.28% (reverse at $5000)

All three pass any reasonable threshold up to ~1%. None are dangerously thin at $100-$5000 position sizes. Round-trip cost (in+out) is ~0.5% for each. Keep all three in WHITELISTED_ASSETS["base"]; no trim.

PR 3.5 — Set the production slippage value. Currently 0.09 (was hacked up to 0.90). Final value depends on calibration in PR 3.4. Could be a single global value, per-chain, or per-token.

3.5 decision (2026-06-04)

`slippage_tolerance` is a single global param (models.py:410), not chain-keyed, so Base inherits whatever Optimism's deployment env sets it to. Decision: keep that shape — match Optimism. No code change in WS3 PR for slippage. If/when Optimism gets retuned (the doc-mentioned 0.09 looks loose vs the 0.42% worst observed on Base, but that's an Optimism-side decision and would need its own calibration), Base follows for free.

PR 3.6 — Tune Base-specific gas and profitability thresholds. Existing values were calibrated for Optimism and may pass trivially or block valid trades on Base.

PR 3.7 — End-to-end validation against a real funded Base safe. Not strictly a PR, more an integration sign-off task before launch.
Launch blockers
Safe API rate limits: proxy through our infra (production team) plus the in-agent cache and pagination fixes in workstream 1.
Four commented-out safety checks need to be revived and validated. Three of them sit in workstream 2 (slippage cap, profitability gate) and workstream 3 (Aerodrome slippage protection).
Cached-opportunity edge case (workstream 2 PR 2.4). Common workflow across all BabyDegen agents, worth fixing before Basius users hit it.
Whitelist calibration and production slippage value need to be agreed (workstream 3 PRs 3.4 and 3.5).
Appendix
PR #269 reference
Closed 5 May 2026. Proposed TTL-cached SafeGlobal responses with an on-chain multicall fallback and a degraded trading mode. Closed for being an architectural change rather than a quick fix. Resolution captured in workstream 1: proxy plus in-agent cache and pagination fixes for MVP; eth_getLogs-based ROI as the post-launch follow-up.
