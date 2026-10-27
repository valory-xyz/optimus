# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""This module contains the behaviour that tops up the Agent Signer from the Agent Safe."""

from typing import Dict, Generator, NamedTuple, Optional

from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.funds_manager.behaviours import (
    GET_AGENT_TOPUP_METHOD_NAME,
    GET_FUNDS_STATUS_METHOD_NAME,
)
from packages.valory.skills.funds_manager.models import FundRequirements
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
    SAFE_TX_GAS,
)
from packages.valory.skills.liquidity_trader_abci.payloads import AgentTopUpPayload
from packages.valory.skills.liquidity_trader_abci.states.agent_top_up import (
    AgentTopUpRound,
)
from packages.valory.skills.liquidity_trader_abci.utils.agent_topup import (
    executed_withdrawal_count,
    native_requirement,
    paid_call_chain,
    signer_pays_topup_gas,
    withdrawal_agent_need,
    x402_agent_need_extra,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)

AGENT_TOPUP_CHECK_INTERVAL_SECONDS = 3600
TOPUP_TX_DATA = b""


class TopUpMove(NamedTuple):
    """A native transfer from the Agent Safe to the Agent Signer."""

    chain: str
    wei: int


class AgentTopUpBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that moves native token from the Agent Safe to the Agent Signer."""

    matching_round = AgentTopUpRound

    def async_act(self) -> Generator:  # type: ignore[override]
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            tx_hash, safe_address, chain = None, None, None
            self._record_unsettled_topups()
            move = yield from self._get_topup_move()
            if move is not None:
                tx_hash = yield from self._prepare_topup_tx(move.chain, move.wei)
                if tx_hash is not None:
                    chain = move.chain
                    safe_address = self.params.safe_contract_addresses.get(chain)
                    self.shared_state.topup_in_flight_chains.add(chain)
                    self._close_topup_check()
                else:
                    self.shared_state.topup_failed_chains.add(move.chain)

            payload = AgentTopUpPayload(
                sender=self.context.agent_address,
                tx_submitter=self.matching_round.auto_round_id(),
                tx_hash=tx_hash,
                safe_contract_address=safe_address,
                chain_id=chain,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    def _record_unsettled_topups(self) -> None:
        """Count a top-up tx that never reached PostTxSettlement as failed."""
        # A retry of this round never sent the tx its first visit prepared.
        if self.round_sequence.last_round_id == self.matching_round.auto_round_id():
            return
        state = self.shared_state
        state.topup_failed_chains |= state.topup_in_flight_chains
        state.topup_in_flight_chains.clear()

    def _is_topup_check_due(self) -> bool:
        """Open the gate at start, then at most once per check interval."""
        last_check = self.shared_state.last_topup_check
        # A retry of this round re-checks: the visit that closed the gate sent
        # a payload that never reached consensus.
        return (
            self.round_sequence.last_round_id == self.matching_round.auto_round_id()
            or last_check is None
            or self._get_current_timestamp() - last_check
            >= AGENT_TOPUP_CHECK_INTERVAL_SECONDS
        )

    def _close_topup_check(self) -> None:
        """Close the gate once a check reaches a decision."""
        self.shared_state.last_topup_check = self._get_current_timestamp()

    def _get_topup_move(self) -> Generator[None, None, Optional[TopUpMove]]:
        """Return the move to make, or None when nothing moves."""
        if not self._is_topup_check_due():
            return None

        get_funds_status = self.context.shared_state.get(GET_FUNDS_STATUS_METHOD_NAME)
        get_agent_topup = self.context.shared_state.get(GET_AGENT_TOPUP_METHOD_NAME)
        if get_funds_status is None or get_agent_topup is None:
            self.context.logger.warning(
                "Agent top-up skipped: funds_manager hooks are not registered."
            )
            self._close_topup_check()
            return None

        try:
            funds = get_funds_status()
        except Exception as exc:  # pylint: disable=broad-except
            self.context.logger.warning(
                f"Agent top-up skipped: reading the funds status raised "
                f"{type(exc).__name__}({exc})."
            )
            return None

        agent_need_extra = x402_agent_need_extra(
            funds,
            paid_call_chain(
                self.coingecko.use_mech_facilitator,
                self.coingecko.mech_chain,
                self.params.target_investment_chains,
            ),
            self.context.agent_address,
            int(self.shared_state.x402_eth_deficit or 0),
        )
        agent_need_override = None
        investing_paused = yield from self._read_investing_paused_or_none()
        if investing_paused is None:
            self.context.logger.warning(
                "Agent top-up deferred: whether investing is paused could not "
                "be read."
            )
            return None
        if investing_paused:
            agent_need_override = yield from self._get_withdrawal_need(
                funds, agent_need_extra
            )
            if agent_need_override is None:
                return None

        try:
            topups = get_agent_topup(
                funds,
                agent_need_extra=agent_need_extra,
                agent_need_override=agent_need_override,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.context.logger.warning(
                f"Agent top-up skipped: the top-up rule raised "
                f"{type(exc).__name__}({exc})."
            )
            return None
        undecided = False
        for chain, topup in topups.items():
            if topup.move <= 0:
                self.shared_state.topup_failed_chains.discard(chain)
                continue
            pays_gas = signer_pays_topup_gas(funds, chain, self.context.agent_address)
            if not pays_gas:
                reason = (
                    "cannot pay the gas"
                    if pays_gas is False
                    else "native balance could not be read"
                )
                self.context.logger.warning(
                    f"Agent top-up skipped on {chain}: the Agent Signer {reason}."
                )
                undecided = True
                continue
            self.context.logger.info(
                f"Agent top-up: moving {topup.move} wei from the Agent Safe "
                f"to the Agent Signer on {chain}."
            )
            return TopUpMove(chain, topup.move)
        if undecided:
            return None
        if not self._native_balances_read(funds):
            self.context.logger.warning(
                "Agent top-up skipped: a native balance could not be read."
            )
            return None
        self._close_topup_check()
        return None

    def _native_balances_read(self, funds: FundRequirements) -> bool:
        """Return whether every Agent Signer and Agent Safe native balance was read."""
        for chain in funds:
            accounts = (
                self.context.agent_address,
                self.params.safe_contract_addresses.get(chain, ""),
            )
            for account in accounts:
                native = native_requirement(funds, chain, account)
                if native is not None and native[1].balance is None:
                    return False
        return True

    def _get_withdrawal_need(
        self, funds: FundRequirements, agent_need_extra: Dict[str, int]
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        """Return the Agent Signer need in withdrawal mode, as /funds-status reports it.

        :param funds: the status returned by ``get_funds_status()``.
        :param agent_need_extra: the x402 extra added to the Agent Signer need.
        :yield: the KV store request.
        :return: ``{trading chain: need}``, or None when it is unknown.
        """
        chain = self.params.target_investment_chains[0]
        withdrawal_data = yield from self._read_kv(("withdrawal_transaction_hashes",))
        need = withdrawal_agent_need(
            funds,
            chain,
            self.context.agent_address,
            agent_need_extra,
            len(self.synchronized_data.withdrawal_actions),
            executed_withdrawal_count(withdrawal_data, self.context.logger),
        )
        if need is None:
            self.context.logger.info(
                "Agent top-up deferred: the withdrawal actions are not prepared "
                "yet, or the executed ones could not be counted."
            )
            return None
        return {chain: need}

    def _prepare_topup_tx(
        self, chain: str, amount: int
    ) -> Generator[None, None, Optional[str]]:
        """Build the Agent Safe -> Agent Signer transfer: one plain CALL, no data."""
        safe_address = self.params.safe_contract_addresses.get(chain)
        agent_address = self.context.agent_address
        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=agent_address,
            value=amount,
            data=TOPUP_TX_DATA,
            operation=SafeOperation.CALL.value,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=chain,
        )
        if safe_tx_hash is None:
            self.context.logger.error(
                f"Agent top-up: could not get the Safe tx hash on {chain}."
            )
            return None

        return hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash[2:],
            ether_value=amount,
            safe_tx_gas=SAFE_TX_GAS,
            to_address=agent_address,
            data=TOPUP_TX_DATA,
            operation=SafeOperation.CALL.value,
        )
