# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the behaviour for checking is staking kpi is met for the 'liquidity_trader_abci' skill."""

from statistics import median
from typing import Generator, NamedTuple, Optional, Type

from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.funds_manager.behaviours import (
    GET_FUNDS_STATUS_METHOD_NAME,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    ETHER_VALUE,
    LiquidityTraderBaseBehaviour,
    SAFE_TX_GAS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckStakingKPIMetPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.check_staking_kpi_met import (
    CheckStakingKPIMetRound,
    Event,
)
from packages.valory.skills.liquidity_trader_abci.utils.activity_goal import (
    ACTIVITY_GOAL_KEY,
    KV_ACTIVITY_GOAL_LAST_COUNTED_PERIOD,
    KV_ACTIVITY_GOAL_LAST_MET_AT,
    KV_ACTIVITY_GOAL_PERIOD_START,
    KV_ACTIVITY_GOAL_PROGRESS,
    KV_ACTIVITY_GOAL_TARGET,
    build_activity_goal_block,
    is_non_negative_int,
    merge_agent_performance,
    should_stand_by,
    stamp_last_met_at,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)

_RECENT_GAS_RECORDS_TO_CONSIDER = 5


class _FundingSignal(NamedTuple):
    """EOA balance versus recent real-tx cost on the staking chain (both wei).

    Named fields rather than a bare tuple so the gate comparison reads as
    ``eoa_balance < recent_real_tx_cost`` and cannot be silently inverted by
    swapping element order (this package is exempt from mypy in ``tox.ini``).
    """

    eoa_balance: int
    recent_real_tx_cost: int


class CheckStakingKPIMetBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that checks if the staking KPI has been met and makes vanity transactions if necessary."""

    # pylint-disable too-many-ancestors
    matching_round: Type[AbstractRound] = CheckStakingKPIMetRound

    def async_act(self) -> Generator:  # type: ignore[override]
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            investing_paused = yield from self._read_investing_paused()
            if investing_paused:
                self.context.logger.info(
                    "Investing paused due to withdrawal request. Transitioning to WithdrawFunds round."
                )
                payload = CheckStakingKPIMetPayload(
                    sender=self.context.agent_address,
                    tx_submitter=self.matching_round.auto_round_id(),
                    tx_hash=None,
                    safe_contract_address=None,
                    chain_id=None,
                    is_staking_kpi_met=None,
                    event=Event.WITHDRAWAL_INITIATED.value,
                )
                yield from self.send_a2a_transaction(payload)
                yield from self.wait_until_round_end()
                self.set_done()
                return

            vanity_tx_hex = None
            is_activity_target_met = None
            activity_target = None
            activity_completed = None

            multisig = self.params.safe_contract_addresses.get(
                self.params.staking_chain
            )
            # Single on-chain read: the KPI verdict and the activity-counter
            # delta come from the same snapshot, so the /healthcheck signal can
            # never disagree with the KPI decision.
            is_staking_kpi_met, multisig_nonces_since_last_cp = (
                yield from self._is_staking_kpi_met()
            )

            # ``is_staking_kpi_met`` is ``None`` whenever the verdict cannot be
            # computed — service not STAKED, ``min_num_of_safe_tx_required``
            # unknown, or a transient nonce-read failure. When it is not ``None``
            # the delta is populated too; compute the activity-target signal for
            # the /healthcheck endpoint, on the new staking regime only (``None``
            # on the old regime / undetermined — see §5.4).
            # A non-None verdict guarantees ``multisig_nonces_since_last_cp`` is an
            # integer (``_is_staking_kpi_met`` returns ``(None, None)`` otherwise),
            # so it is safe to use directly below without re-guarding for ``None``.
            is_new_regime = None
            if is_staking_kpi_met is not None:
                is_new_regime = yield from self._is_new_staking_regime()
                if is_new_regime:
                    activity_target = self.params.activity_target
                    activity_completed = multisig_nonces_since_last_cp
                    is_activity_target_met = (
                        multisig_nonces_since_last_cp >= activity_target
                    )

            if is_staking_kpi_met is None:
                # Most commonly the service is simply not staked, which is a
                # normal steady state — keep it quiet. A genuine read failure on
                # a staked service is logged at ERROR inside ``_is_staking_kpi_met``.
                self.context.logger.info(
                    "Staking KPI undetermined (service not staked or counter "
                    "unavailable); skipping activity tx this period."
                )
            elif is_staking_kpi_met is True:
                self.context.logger.info("KPI already met for the day!")
            else:
                is_period_threshold_exceeded = (
                    self.synchronized_data.period_count  # type: ignore[operator]
                    - self.synchronized_data.period_number_at_last_cp
                    >= self.params.staking_threshold_period
                )

                if is_period_threshold_exceeded:
                    # A False verdict guarantees ``_is_staking_kpi_met`` saw a
                    # non-None ``min_num_of_safe_tx_required`` and an integer nonce
                    # delta, so both are safe to use directly here.
                    min_num_of_safe_tx_required = (
                        self.synchronized_data.min_num_of_safe_tx_required
                    )
                    num_of_tx_left_to_meet_kpi = (
                        min_num_of_safe_tx_required - multisig_nonces_since_last_cp
                    )
                    if num_of_tx_left_to_meet_kpi > 0:
                        self.context.logger.info(
                            f"Number of tx left to meet KPI: {num_of_tx_left_to_meet_kpi}"
                        )
                        if is_new_regime is None:
                            # Regime undetermined (transient VERSION read);
                            # do not fire a vanity tx and risk ticking the wrong
                            # counter — retry next period.
                            self.context.logger.warning(
                                "Staking regime undetermined this period; "
                                "deferring activity tx until it resolves"
                            )
                        elif is_new_regime:
                            # This regime's counter is marketplace requests, and
                            # the only ones the agent makes are its paid API
                            # calls on the facilitator route. Nothing is sent
                            # purely to tick it. With that route off no amount of
                            # work moves the counter, so the shortfall is a
                            # misconfiguration rather than a quiet day, and the
                            # service is heading for eviction either way.
                            if not self.params.use_mech_facilitator:
                                self.context.logger.error(
                                    "Staking KPI is short by "
                                    f"{num_of_tx_left_to_meet_kpi} and cannot be "
                                    "met: this staking regime counts mech "
                                    "marketplace requests and the agent makes "
                                    "none while use_mech_facilitator is off. "
                                    "Enable it, or stake against an activity "
                                    "checker that counts Safe nonces."
                                )
                            else:
                                self.context.logger.warning(
                                    "Staking KPI is short by "
                                    f"{num_of_tx_left_to_meet_kpi}; this regime "
                                    "counts the agent's paid API calls, and "
                                    "settlement is batched so recent ones may "
                                    "not be counted yet. Nothing is sent to make "
                                    "up the difference."
                                )
                        else:
                            # Old regime: keep the existing vanity Safe tx, gated
                            # by the EOA-funded check so vanity activity does not
                            # hide a funding alert behind a green staking KPI.
                            signal = self._real_tx_cost_vs_balance(
                                chain=self.params.staking_chain  # type: ignore[arg-type]
                            )
                            if (
                                signal is not None
                                and signal.eoa_balance < signal.recent_real_tx_cost
                            ):
                                self.context.logger.warning(
                                    f"vanity tx suppressed: agent EOA balance "
                                    f"{signal.eoa_balance} wei on "
                                    f"{self.params.staking_chain} is below the "
                                    f"recent real-tx cost "
                                    f"{signal.recent_real_tx_cost} wei; fund EOA "
                                    f"to restore staking activity"
                                )
                            else:
                                self.context.logger.info("Preparing vanity tx..")
                                vanity_tx_hex = yield from self._prepare_vanity_tx(
                                    chain=self.params.staking_chain  # type: ignore[arg-type]
                                )
                                self.context.logger.info(f"tx hash: {vanity_tx_hex}")

            is_activity_goal_met = None
            if is_staking_kpi_met is not None:
                is_activity_goal_met = yield from self._track_activity_goal(
                    is_staking_kpi_met,
                    is_activity_target_met,
                    settling=vanity_tx_hex is not None,
                )

            tx_submitter = self.matching_round.auto_round_id()
            payload = CheckStakingKPIMetPayload(
                self.context.agent_address,
                tx_submitter,
                vanity_tx_hex,
                multisig,
                self.params.staking_chain,
                is_staking_kpi_met,
                is_activity_target_met=is_activity_target_met,
                activity_target=activity_target,
                activity_completed=activity_completed,
                is_activity_goal_met=is_activity_goal_met,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _track_activity_goal(
        self,
        is_staking_kpi_met: bool,
        is_activity_target_met: Optional[bool],
        settling: bool,
    ) -> Generator[None, None, Optional[bool]]:
        """Count this period toward the rounds goal and publish the block.

        Progress belongs to the staking epoch starting at ``tsCheckpoint`` and
        restarts from zero when that moves. A period is counted once, however
        many times this round runs in it, and only if it does not stand by.
        The verdict returned is taken before counting, so the period that
        reaches the goal still works and the next one stands by.

        :param is_staking_kpi_met: the on-chain KPI verdict.
        :param is_activity_target_met: the new-regime activity-target verdict.
        :param settling: whether this period settles a vanity tx first.
        :yield: the contract and KV store requests.
        :return: whether the goal was met before this period, or ``None`` when
            the epoch or the stored state cannot be read.
        """
        ts_checkpoint = yield from self._get_ts_checkpoint(
            chain=self.params.staking_chain
        )
        if not is_non_negative_int(ts_checkpoint):
            self.context.logger.warning(
                f"Cannot read tsCheckpoint ({ts_checkpoint!r}); activity goal "
                "not updated this period."
            )
            return None

        state = yield from self._read_activity_goal_state()
        if state is None:
            self.context.logger.warning(
                "KV store unreachable; activity goal not updated this period."
            )
            return None

        target = state[KV_ACTIVITY_GOAL_TARGET]
        if target is None:
            target = self.params.activity_goal_target
        progress = state[KV_ACTIVITY_GOAL_PROGRESS] or 0
        last_counted_period = state[KV_ACTIVITY_GOAL_LAST_COUNTED_PERIOD]
        if state[KV_ACTIVITY_GOAL_PERIOD_START] != ts_checkpoint:
            progress, last_counted_period = 0, None

        is_activity_goal_met = progress >= target
        period = self.synchronized_data.period_count
        stands_by = not settling and should_stand_by(
            is_staking_kpi_met, is_activity_target_met, is_activity_goal_met
        )
        if stands_by:
            self.context.logger.info(
                f"Activity goal ({progress}/{target} rounds) and staking target "
                "met; standing by until the next epoch."
            )
        elif last_counted_period != period:
            progress += 1
            last_counted_period = period

        now = self._get_current_timestamp()
        last_met_at = stamp_last_met_at(
            progress,
            target,
            ts_checkpoint,
            state[KV_ACTIVITY_GOAL_LAST_MET_AT],
            now,
        )
        written = yield from self._write_activity_goal_state(
            {
                KV_ACTIVITY_GOAL_PROGRESS: progress,
                KV_ACTIVITY_GOAL_PERIOD_START: ts_checkpoint,
                KV_ACTIVITY_GOAL_LAST_MET_AT: last_met_at,
                KV_ACTIVITY_GOAL_LAST_COUNTED_PERIOD: last_counted_period,
            }
        )
        if not written:
            # Publishing progress the KV store does not hold would make it go
            # backwards on the next period.
            self.context.logger.error(
                "Failed to persist the activity goal; not publishing it."
            )
            return is_activity_goal_met

        block = build_activity_goal_block(
            target, progress, ts_checkpoint, last_met_at, now
        )
        try:
            merge_agent_performance(
                self.agent_performance_filepath, {ACTIVITY_GOAL_KEY: block}
            )
        except OSError as e:
            self.context.logger.error(f"Failed to publish the activity goal: {e}")
        return is_activity_goal_met

    def _real_tx_cost_vs_balance(self, chain: str) -> Optional[_FundingSignal]:
        """Read the agent EOA balance and recent real-tx cost on ``chain``.

        Balance comes from the funds_manager shared-state hook (the bound
        ``get_funds_status`` method that also backs ``/funds-status``); the hook
        returns a ``FundRequirements`` instance whose ``get_response_body()`` is
        read here. Cost is the median of the last few ``GasCostTracker`` records,
        which ``post_tx_settlement`` populates from settled non-vanity tx
        receipts.

        The gate is designed to fail open: any of the several distinct paths
        where a signal cannot be read returns ``None`` so a transient lookup
        error never silently kills the staking KPI. Routine empty paths (no
        chain mapping, no records yet, hook not registered, balance absent) stay
        quiet; the unexpected ones (malformed gas record, hook raising) emit a
        WARNING so a permanently dead gate is not mistaken for a healthy boot.

        Latency note: the hook runs synchronous Multicall RPCs (one per
        configured chain, even though only ``chain`` is needed) and the shipped
        ``funds_manager`` builds its ``Web3`` provider with no explicit HTTP
        timeout, so a slow/hanging endpoint can block the cooperative
        ``async_act`` loop until the round timer fires. This risk is accepted
        because the gate only runs on the rare KPI-behind path (KPI unmet,
        threshold period exceeded, and txs still owed). Bounding the call with a
        timeout — or having ``funds_manager`` cache its last poll so this reads a
        scalar — is left as a ``funds_manager`` follow-up.

        :param chain: chain name as used in ``chain_to_chain_id_mapping`` and
            ``safe_contract_addresses`` (the staking chain).
        :return: a ``_FundingSignal`` of ``(eoa_balance, recent_real_tx_cost)``
            in wei, or ``None`` when either signal is unavailable.
        """
        chain_id = self.params.chain_to_chain_id_mapping.get(chain)
        if not chain_id:
            return None
        records = self.gas_cost_tracker.data.get(str(chain_id), [])
        if not records:
            return None
        try:
            recent_costs = [
                int(r["gas_used"]) * int(r["gas_price"])
                for r in records[-_RECENT_GAS_RECORDS_TO_CONSIDER:]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            self.context.logger.warning(
                f"vanity-tx gate: malformed gas-cost record on {chain} "
                f"({type(exc).__name__}: {exc}); failing open (vanity tx allowed)"
            )
            return None
        cost = int(median(recent_costs))

        hook = self.context.shared_state.get(GET_FUNDS_STATUS_METHOD_NAME)
        if hook is None:
            return None
        try:
            response = hook().get_response_body()
        except Exception as exc:  # pylint: disable=broad-except
            self.context.logger.warning(
                f"vanity-tx gate: funds-status hook raised "
                f"{type(exc).__name__}({exc}); failing open (vanity tx allowed)"
            )
            return None
        balance_str = (
            response.get(chain, {})
            .get(self.context.agent_address, {})
            .get(ZERO_ADDRESS, {})
            .get("balance")
        )
        if balance_str is None:
            return None
        try:
            balance = int(balance_str)
        except (TypeError, ValueError):
            return None
        return _FundingSignal(balance, cost)

    def _prepare_vanity_tx(self, chain: str) -> Generator[None, None, Optional[str]]:
        self.context.logger.info(f"Preparing vanity transaction for chain: {chain}")

        safe_address = self.params.safe_contract_addresses.get(chain)
        self.context.logger.debug(f"Safe address for chain {chain}: {safe_address}")

        tx_data = b"0x"
        self.context.logger.debug(f"Transaction data: {tx_data}")  # type: ignore[str-bytes-safe]

        try:
            safe_tx_hash = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=safe_address,
                contract_public_id=GnosisSafeContract.contract_id,
                contract_callable="get_raw_safe_transaction_hash",
                data_key="tx_hash",
                to_address=ZERO_ADDRESS,
                value=ETHER_VALUE,
                data=tx_data,
                operation=SafeOperation.CALL.value,
                safe_tx_gas=SAFE_TX_GAS,
                chain_id=chain,
            )
        except Exception as e:
            self.context.logger.error(f"Exception during contract interaction: {e}")
            return None

        if safe_tx_hash is None:
            self.context.logger.error("Error preparing vanity tx: safe_tx_hash is None")
            return None

        self.context.logger.debug(f"Safe transaction hash: {safe_tx_hash}")

        try:
            tx_hash = hash_payload_to_hex(
                safe_tx_hash=safe_tx_hash[2:],
                ether_value=ETHER_VALUE,
                safe_tx_gas=SAFE_TX_GAS,
                operation=SafeOperation.CALL.value,
                to_address=ZERO_ADDRESS,
                data=tx_data,
            )
        except Exception as e:
            self.context.logger.error(f"Exception during hash payload conversion: {e}")
            return None

        self.context.logger.info(f"Vanity transaction hash: {tx_hash}")

        return tx_hash
