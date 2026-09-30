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

"""Keep the marketplace pre-deposit funded so paid API calls can be served."""

import json
from typing import Generator, List, Optional, Tuple, Type

from packages.valory.contracts.balance_tracker.contract import BalanceTrackerContract
from packages.valory.contracts.erc20.contract import ERC20TokenContract
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.mech_marketplace.contract import (
    MechMarketplaceContract,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    ETHER_VALUE,
    HTTP_OK,
    LiquidityTraderBaseBehaviour,
    SAFE_TX_GAS,
    ZERO_ADDRESS,
)
from packages.valory.skills.liquidity_trader_abci.payloads import (
    CheckMechPreDepositPayload,
)
from packages.valory.skills.liquidity_trader_abci.states.base import Event
from packages.valory.skills.liquidity_trader_abci.states.check_mech_pre_deposit import (
    CheckMechPreDepositRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


class CheckMechPreDepositBehaviour(LiquidityTraderBaseBehaviour):
    """Top the marketplace pre-deposit up from the Safe when it runs low.

    Paid API calls are debited from the Safe's pre-deposit held by the
    marketplace balance tracker, not from the Safe's own token balance, and
    nothing else in this agent moves funds between the two. Without this the
    Safe can hold the payment token indefinitely while every paid call is
    refused for want of a deposit.
    """

    matching_round: Type[AbstractRound] = CheckMechPreDepositRound

    def async_act(self) -> Generator:
        """Check the pre-deposit and prepare a top-up when it is short.

        :yield: the contract reads and the consensus round.
        """
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # A deposit is one-way for the agent's purposes: the marketplace
            # spends it on deliveries and nothing here withdraws it. While a
            # withdrawal is pending the payment token has to stay in the Safe.
            investing_paused = yield from self._read_investing_paused()
            if investing_paused:
                self.context.logger.info(
                    "Withdrawal requested; leaving the mech pre-deposit alone."
                )
                payload = CheckMechPreDepositPayload(
                    sender=self.context.agent_address,
                    tx_submitter=self.matching_round.auto_round_id(),
                    tx_hash=None,
                    safe_contract_address=None,
                    chain_id=None,
                    event=Event.WITHDRAWAL_INITIATED.value,
                )
            else:
                tx_hash = yield from self._prepare_top_up()
                payload = CheckMechPreDepositPayload(
                    sender=self.context.agent_address,
                    tx_submitter=self.matching_round.auto_round_id(),
                    tx_hash=tx_hash,
                    safe_contract_address=self._safe_address,
                    chain_id=self._chain,
                )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()

    @property
    def _chain(self) -> str:
        """Return the chain the marketplace is on."""
        return str(self.coingecko.mech_chain)

    @property
    def _safe_address(self) -> Optional[str]:
        """Return the Safe that pays, on the mech chain."""
        return self.params.safe_contract_addresses.get(self._chain)

    def _prepare_top_up(self) -> Generator[None, None, Optional[str]]:
        """Return a Safe tx that tops the pre-deposit up, or ``None``.

        :yield: the reads.
        :return: the settleable transaction hash, or ``None`` when the
            pre-deposit already covers the floor or nothing could be read.
        """
        if not self.coingecko.use_mech_facilitator:
            return None

        safe = self._safe_address
        if not safe:
            self.context.logger.warning(
                f"No Safe configured for {self._chain}; cannot fund the mech "
                "pre-deposit."
            )
            return None

        info = yield from self._read_requester_info()
        if info is None:
            return None

        try:
            deposited = int(info["balance"])
            marketplace = str(info["marketplace_address"])
            payment_type = str(info["payment_type"])
        except (KeyError, TypeError, ValueError) as exc:
            self.context.logger.warning(
                f"Facilitator requester info was missing what the deposit "
                f"needs: {exc}"
            )
            return None

        floor = self.coingecko.mech_pre_deposit_floor
        if deposited >= floor:
            self.context.logger.debug(
                f"Mech pre-deposit {deposited} is at or above the floor "
                f"{floor}; nothing to do."
            )
            return None

        amount = min(
            self.coingecko.mech_pre_deposit_target - deposited,
            self.coingecko.mech_pre_deposit_cap,
        )
        if amount <= 0:
            return None

        tracker = yield from self._resolve_balance_tracker(marketplace, payment_type)
        if tracker is None:
            return None

        self.context.logger.info(
            f"Mech pre-deposit {deposited} is below the floor {floor}; "
            f"depositing {amount} into {tracker}."
        )
        return (yield from self._build_deposit_tx(tracker, safe, amount))

    def _read_requester_info(self) -> Generator[None, None, Optional[dict]]:
        """Read what the facilitator reports for this Safe.

        :yield: the HTTP call.
        :return: the decoded body, or ``None``.

        Taken from the facilitator rather than from configuration, because it
        decides which marketplace and mech it serves and a configured address
        can drift from that silently. The call is the framework's own helper,
        so it yields rather than blocking the agent.
        """
        base = str(self.coingecko.mech_facilitator_base_url).rstrip("/")
        url = f"{base}/mech/{self._chain}/requester/{self._safe_address}"
        response = yield from self.get_http_response(method="GET", url=url)
        if response is None or response.status_code not in HTTP_OK:
            self.context.logger.warning(
                "Could not read the facilitator's requester info; skipping the "
                "pre-deposit check this period."
            )
            return None
        try:
            return dict(json.loads(response.body))
        except (ValueError, TypeError) as exc:
            self.context.logger.warning(
                f"Facilitator requester info was not readable: {exc}"
            )
            return None

    def _resolve_balance_tracker(
        self, marketplace: str, payment_type: str
    ) -> Generator[None, None, Optional[str]]:
        """Resolve the balance tracker that holds the pre-deposit.

        :param marketplace: the marketplace the facilitator reports.
        :param payment_type: the mech's payment type.
        :yield: the contract read.
        :return: the tracker address, or ``None``.

        Trackers are keyed by payment type, so this resolves the one the
        facilitator debits for this mech rather than assuming a single pot.
        """
        tracker = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=marketplace,
            contract_public_id=MechMarketplaceContract.contract_id,
            contract_callable="get_balance_tracker_for_mech_type",
            data_key="data",
            mech_type=payment_type,
            chain_id=self._chain,
        )
        if not tracker or tracker == ZERO_ADDRESS:
            self.context.logger.warning(
                f"The marketplace reports no balance tracker for payment type "
                f"{payment_type!r}; skipping the pre-deposit check."
            )
            return None
        return str(tracker)

    def _build_deposit_tx(
        self, tracker: str, safe: str, amount: int
    ) -> Generator[None, None, Optional[str]]:
        """Build the Safe transaction that deposits ``amount`` for ``safe``.

        :param tracker: the balance tracker address.
        :param safe: the requester Safe, which is also the sender.
        :param amount: how much to deposit, in the token's base units.
        :yield: the contract reads.
        :return: the settleable transaction hash, or ``None``.

        A native tracker takes the deposit as transaction value. A token one
        needs an allowance first, so the approve and the deposit go out as one
        multisend: a deposit that lands without its approval reverts, and an
        approval that lands without its deposit leaves the allowance standing.

        The deposit never exceeds what the Safe holds. The shortfall is
        computed from what the tracker reports, which says nothing about
        whether the Safe has been funded yet, and a deposit above the balance
        reverts on settlement instead of being refused here.
        """
        token = yield from self._read_tracker_token(tracker)

        available = yield from self._safe_balance(token, safe)
        if available is None:
            return None
        if available <= 0:
            self.context.logger.warning(
                f"The Safe holds none of the mech payment token on "
                f"{self._chain}; cannot fund the pre-deposit."
            )
            return None
        amount = min(amount, available)

        deposit_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=tracker,
            contract_public_id=BalanceTrackerContract.contract_id,
            contract_callable="build_deposit_for_data",
            data_key="data",
            account=safe,
            amount=amount,
            chain_id=self._chain,
        )
        if deposit_data is None:
            self.context.logger.error("Could not encode the pre-deposit call.")
            return None

        if token is None:
            # Native tracker: the deposit carries its value directly.
            return (
                yield from self._safe_tx(
                    to_address=tracker,
                    data=deposit_data,
                    value=amount,
                )
            )

        approve_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=token,
            contract_public_id=ERC20TokenContract.contract_id,
            contract_callable="build_approval_tx",
            data_key="data",
            spender=tracker,
            amount=amount,
            chain_id=self._chain,
        )
        if approve_data is None:
            self.context.logger.error("Could not encode the pre-deposit approval.")
            return None

        return (
            yield from self._multisend_tx(
                [(token, approve_data), (tracker, deposit_data)]
            )
        )

    def _safe_balance(
        self, token: Optional[str], safe: str
    ) -> Generator[None, None, Optional[int]]:
        """Return what the Safe holds of the tracker's payment asset.

        :param token: the tracker's token, or ``None`` for a native tracker.
        :param safe: the Safe that pays.
        :yield: the balance read.
        :return: the balance in base units, or ``None`` when it is unreadable.
        """
        if token is None:
            balance = yield from self._get_native_balance(self._chain, safe)
        else:
            balance = yield from self._get_token_balance(self._chain, safe, token)

        if balance is None:
            self.context.logger.warning(
                "Could not read the Safe's payment-token balance; skipping the "
                "pre-deposit check this period."
            )
        return balance

    def _read_tracker_token(self, tracker: str) -> Generator[None, None, Optional[str]]:
        """Return the token a tracker takes, or ``None`` when it is native.

        :param tracker: the balance tracker address.
        :yield: the contract read.
        :return: the token address, or ``None`` for a native tracker.

        A native tracker has no ``token()``, so the read reverting is the
        signal rather than an error.
        """
        token = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=tracker,
            contract_public_id=BalanceTrackerContract.contract_id,
            contract_callable="get_token_address",
            data_key="token_address",
            chain_id=self._chain,
        )
        if not token or token == ZERO_ADDRESS:
            return None
        return str(token)

    def _multisend_tx(
        self, calls: List[Tuple[str, bytes]]
    ) -> Generator[None, None, Optional[str]]:
        """Bundle ``calls`` into one Safe transaction through MultiSend.

        :param calls: the target address and calldata of each call.
        :yield: the contract reads.
        :return: the settleable transaction hash, or ``None``.
        """
        multisend_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.multisend_contract_addresses[self._chain],
            contract_public_id=MultiSendContract.contract_id,
            contract_callable="get_tx_data",
            data_key="data",
            multi_send_txs=[
                {
                    "operation": MultiSendOperation.CALL,
                    "to": to_address,
                    "value": ETHER_VALUE,
                    "data": data,
                }
                for to_address, data in calls
            ],
            chain_id=self._chain,
        )
        if multisend_data is None:
            self.context.logger.error("Could not build the pre-deposit multisend.")
            return None

        return (
            yield from self._safe_tx(
                to_address=self.params.multisend_contract_addresses[self._chain],
                data=bytes.fromhex(str(multisend_data)[2:]),
                value=ETHER_VALUE,
                operation=SafeOperation.DELEGATE_CALL.value,
            )
        )

    def _safe_tx(
        self,
        to_address: str,
        data: bytes,
        value: int,
        operation: int = SafeOperation.CALL.value,
    ) -> Generator[None, None, Optional[str]]:
        """Hash a Safe transaction into the form the settlement skill takes.

        :param to_address: the call target.
        :param data: the calldata.
        :param value: the native value to send.
        :param operation: CALL, or DELEGATE_CALL for a multisend.
        :yield: the contract read.
        :return: the settleable transaction hash, or ``None``.
        """
        safe_tx_hash = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self._safe_address,
            contract_public_id=GnosisSafeContract.contract_id,
            contract_callable="get_raw_safe_transaction_hash",
            data_key="tx_hash",
            to_address=to_address,
            value=value,
            data=data,
            operation=operation,
            safe_tx_gas=SAFE_TX_GAS,
            chain_id=self._chain,
        )
        if safe_tx_hash is None:
            self.context.logger.error("Could not hash the pre-deposit Safe tx.")
            return None

        return hash_payload_to_hex(
            safe_tx_hash=safe_tx_hash[2:],
            ether_value=value,
            safe_tx_gas=SAFE_TX_GAS,
            operation=operation,
            to_address=to_address,
            data=data,
        )
