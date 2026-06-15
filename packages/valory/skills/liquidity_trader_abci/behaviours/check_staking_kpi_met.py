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
            is_staking_kpi_met = yield from self._is_staking_kpi_met()
            if is_staking_kpi_met is None:
                self.context.logger.error("Error checking if staking KPI is met.")
            elif is_staking_kpi_met is True:
                self.context.logger.info("KPI already met for the day!")
            else:
                is_period_threshold_exceeded = (
                    self.synchronized_data.period_count  # type: ignore[operator]
                    - self.synchronized_data.period_number_at_last_cp
                    >= self.params.staking_threshold_period
                )

                if is_period_threshold_exceeded:
                    min_num_of_safe_tx_required = (
                        self.synchronized_data.min_num_of_safe_tx_required
                    )
                    multisig_nonces_since_last_cp = (
                        yield from self._get_multisig_nonces_since_last_cp(
                            chain=self.params.staking_chain,  # type: ignore[arg-type]
                            multisig=self.params.safe_contract_addresses.get(
                                self.params.staking_chain
                            ),
                        )
                    )
                    if (
                        multisig_nonces_since_last_cp is not None
                        and min_num_of_safe_tx_required is not None
                    ):
                        num_of_tx_left_to_meet_kpi = (
                            min_num_of_safe_tx_required - multisig_nonces_since_last_cp
                        )
                        if num_of_tx_left_to_meet_kpi > 0:
                            self.context.logger.info(
                                f"Number of tx left to meet KPI: {num_of_tx_left_to_meet_kpi}"
                            )
                            signal = self._real_tx_cost_vs_balance(
                                chain=self.params.staking_chain  # type: ignore[arg-type]
                            )
                            if (
                                signal is not None
                                and signal.eoa_balance < signal.recent_real_tx_cost
                            ):
                                # Padding the multisig nonce with vanity txs
                                # while the EOA cannot fund a real on-chain
                                # action hides the funding alert behind a
                                # green staking KPI.
                                self.context.logger.warning(
                                    f"vanity-tx suppressed: agent EOA balance "
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

            tx_submitter = self.matching_round.auto_round_id()
            payload = CheckStakingKPIMetPayload(
                self.context.agent_address,
                tx_submitter,
                vanity_tx_hex,
                self.params.safe_contract_addresses.get(self.params.staking_chain),
                self.params.staking_chain,
                is_staking_kpi_met,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

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
