# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This module contains the behaviour for preparing swap(s) instructions."""

import json
import traceback
from typing import Any, Callable, Dict, Generator, List, Optional, Sized, Tuple, cast

from packages.eightballer.connections.dcxt import PUBLIC_ID as DCXT_ID
from packages.eightballer.protocols.orders.custom_types import (
    Order,
    OrderSide,
    OrderType,
)
from packages.eightballer.protocols.orders.message import OrdersMessage
from packages.valory.contracts.gnosis_safe.contract import GnosisSafeContract
from packages.valory.protocols.contract_api.message import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import (
    BaseTxPayload,
    LEDGER_API_ADDRESS,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogue,
    ContractApiDialogues,
)
from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.abstract_round_abci.models import Requests
from packages.valory.skills.strategy_evaluator_abci.behaviours.base import (
    StrategyEvaluatorBaseBehaviour,
)
from packages.valory.skills.strategy_evaluator_abci.payloads import (
    TransactionHashPayload,
)
from packages.valory.skills.strategy_evaluator_abci.states.prepare_swap import (
    PrepareEvmSwapRound,
    PrepareSwapRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)
from packages.valory.skills.transaction_settlement_abci.rounds import TX_HASH_LENGTH


SAFE_GAS = 0


class PrepareSwapBehaviour(StrategyEvaluatorBaseBehaviour):
    """A behaviour in which the agents execute the selected strategy and decide on the swap(s)."""

    matching_round = PrepareSwapRound

    def __init__(self, **kwargs: Any):
        """Initialize the swap-preparation behaviour."""
        super().__init__(**kwargs)
        self.incomplete = False

    def setup(self) -> None:
        """Initialize the behaviour."""
        self.context.swap_quotes.reset_retries()
        self.context.swap_instructions.reset_retries()

    def build_quote(
        self, quote_data: Dict[str, str]
    ) -> Generator[None, None, Optional[dict]]:
        """Build the quote."""
        response = yield from self._get_response(self.context.swap_quotes, quote_data)
        return response

    def build_instructions(self, quote: dict) -> Generator[None, None, Optional[dict]]:
        """Build the instructions."""
        content = {
            "quoteResponse": quote,
            "userPublicKey": self.context.agent_address,
        }
        response = yield from self._get_response(
            self.context.swap_instructions,
            dynamic_parameters={},
            content=content,
        )
        return response

    def build_swap_tx(
        self, quote_data: Dict[str, str]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Build instructions for a swap transaction."""
        quote = yield from self.build_quote(quote_data)
        if quote is None:
            return None
        instructions = yield from self.build_instructions(quote)
        return instructions

    def prepare_instructions(
        self, orders: List[Dict[str, str]]
    ) -> Generator[None, None, Tuple[List[Dict[str, Any]], bool]]:
        """Prepare the instructions for a Swap transaction."""
        instructions = []
        for quote_data in orders:
            swap_instruction = yield from self.build_swap_tx(quote_data)
            if swap_instruction is None:
                self.incomplete = True
            else:
                instructions.append(swap_instruction)

        return instructions, self.incomplete

    def async_act(self) -> Generator:
        """Do the action."""
        yield from self.get_process_store_act(
            self.synchronized_data.backtested_orders_hash,
            self.prepare_instructions,
            str(self.swap_instructions_filepath),
        )


class PrepareEvmSwapBehaviour(StrategyEvaluatorBaseBehaviour):
    """A behaviour in which the agents execute the selected strategy and decide on the swap(s)."""

    matching_round = PrepareEvmSwapRound

    def __init__(self, **kwargs: Any):
        """Initialize the swap-preparation behaviour."""
        super().__init__(**kwargs)
        self.incomplete = False
        self._performative_to_dialogue_class = {
            OrdersMessage.Performative.CREATE_ORDER: self.context.orders_dialogues,
        }

    def setup(self) -> None:
        """Initialize the behaviour."""
        self.context.swap_quotes.reset_retries()
        self.context.swap_instructions.reset_retries()

    def build_quote(
        self, quote_data: Dict[str, str]
    ) -> Generator[None, None, Optional[dict]]:
        """Build the quote."""
        response = yield from self._get_response(self.context.swap_quotes, quote_data)
        return response

    def build_instructions(self, quote: dict) -> Generator[None, None, Optional[dict]]:
        """Build the instructions."""
        content = {
            "quoteResponse": quote,
            "userPublicKey": self.context.agent_address,
        }
        response = yield from self._get_response(
            self.context.swap_instructions,
            dynamic_parameters={},
            content=content,
        )
        return response

    def build_swap_tx(
        self, quote_data: Dict[str, str]
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Build instructions for a swap transaction."""
        quote = yield from self.build_quote(quote_data)
        if quote is None:
            return None
        instructions = yield from self.build_instructions(quote)
        return instructions

    def prepare_transactions(
        self, orders: List[Dict[str, str]]
    ) -> Generator[None, None, Tuple[List[Dict[str, Any]], bool]]:
        """Prepare the instructions for a Swap transaction."""
        instructions = []
        for quote_data in orders:
            symbol = f'{quote_data["inputMint"]}/{quote_data["outputMint"]}'
            # We assume for now that we are only sending to the one exchange
            ledger_id: str = self.params.ledger_ids[0]
            exchange_ids = self.params.exchange_ids[ledger_id]
            if len(exchange_ids) != 1:
                self.context.logger.error(
                    f"Expected exactly one exchange id, got {exchange_ids}."
                )
                raise ValueError(
                    f"Expected exactly one exchange id, got {exchange_ids}."
                )
            exchange_id = f"{exchange_ids[0]}_{ledger_id}"

            order = Order(
                exchange_id=exchange_id,
                symbol=symbol,
                amount=self.params.trade_size_in_base_token,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                data=json.dumps(
                    {
                        "safe_contract_address": self.synchronized_data.safe_contract_address,
                    }
                ),
            )

            result = yield from self.get_dcxt_response(
                protocol_performative=OrdersMessage.Performative.CREATE_ORDER,  # type: ignore
                order=order,
            )
            call_data = result.order.data
            try:
                can_create_hash = yield from self._build_safe_tx_hash(
                    vault_address=call_data["vault_address"],
                    chain_id=call_data["chain_id"],
                    call_data=bytes.fromhex(call_data["data"][2:]),
                )
            except Exception as e:
                can_create_hash = False
                self.context.logger.error(
                    f"Error building safe tx hash: {traceback.format_exc()} with error {e}"
                )

            if call_data is None:
                self.incomplete = not can_create_hash
            else:
                instructions.append(call_data)

        return instructions, self.incomplete

    def _build_safe_tx_hash(
        self,
        vault_address: str,
        chain_id: int,
        call_data: bytes,
    ) -> Any:
        """Prepares and returns the safe tx hash for a multisend tx."""
        self.context.logger.info(
            f"Building safe tx hash: safe={self.synchronized_data.safe_contract_address}\n"
            + f"vault={vault_address}\n"
            + f"chain_id={chain_id}\n"
            + f"call_data={call_data.hex()}"
        )

        ledger_id: str = self.params.ledger_ids[0]
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=vault_address,
            value=0,
            data=call_data,
            safe_tx_gas=SAFE_GAS,
            ledger_id="ethereum",
            chain_id=ledger_id,
        )

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                "Couldn't get safe tx hash. Expected response performative "
                f"{ContractApiMessage.Performative.RAW_TRANSACTION.value}, "  # type: ignore
                f"received {response_msg.performative.value}: {response_msg}."
            )
            return False

        tx_hash = response_msg.raw_transaction.body.get("tx_hash", None)
        if tx_hash is None or len(tx_hash) != TX_HASH_LENGTH:
            self.context.logger.error(
                "Something went wrong while trying to get the buy transaction's hash. "
                f"Invalid hash {tx_hash!r} was returned."
            )
            return False

        safe_tx_hash = tx_hash[2:]
        self.context.logger.info(f"Hash of the Safe transaction: {safe_tx_hash}")
        # temp hack:
        payload_string = hash_payload_to_hex(
            safe_tx_hash, 0, SAFE_GAS, vault_address, call_data
        )
        self.safe_tx_hash = safe_tx_hash
        self.payload_string = payload_string
        self.call_data = call_data
        return True

    def async_act(self) -> Generator:
        """Do the action."""
        yield from self.get_process_store_act(
            self.synchronized_data.backtested_orders_hash,
            self.prepare_transactions,
            str(self.swap_instructions_filepath),
        )

    def get_dcxt_response(
        self,
        protocol_performative: OrdersMessage.Performative,
        **kwargs: Any,
    ) -> Generator[None, None, Any]:
        """Get a ccxt response."""
        if protocol_performative not in self._performative_to_dialogue_class:
            raise ValueError(
                f"Unsupported protocol performative {protocol_performative}."
            )
        dialogue_class = self._performative_to_dialogue_class[protocol_performative]

        msg, dialogue = dialogue_class.create(
            counterparty=str(DCXT_ID),
            performative=protocol_performative,
            **kwargs,
        )
        msg._sender = str(self.context.skill_id)  # pylint: disable=protected-access
        response = yield from self._do_request(msg, dialogue)
        return response

    def get_process_store_act(
        self,
        hash_: Optional[str],
        process_fn: Callable[[Any], Generator[None, None, Tuple[Sized, bool]]],
        store_filepath: str,
    ) -> Generator:
        """An async act method for getting some data, processing them, and storing the result.

        1. Get some data using the given hash.
        2. Process them using the given fn.
        3. Send them to IPFS using the given filepath as intermediate storage.

        :param hash_: the hash of the data to process.
        :param process_fn: the function to process the data.
        :param store_filepath: path to the file to store the processed data.
        :yield: None
        """
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            data = yield from self.get_from_ipfs(hash_, SupportedFiletype.JSON)
            sender = self.context.agent_address
            yield from self.get_ipfs_hash_payload_content(
                data, process_fn, store_filepath
            )

            payload = TransactionHashPayload(
                sender,
                tx_hash=self.payload_string,
            )

        yield from self.finish_behaviour(payload)

    def finish_behaviour(self, payload: BaseTxPayload) -> Generator:
        """Finish the behaviour."""
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_contract_api_response(
        self,
        performative: ContractApiMessage.Performative,
        contract_address: Optional[str],
        contract_id: str,
        contract_callable: str,
        ledger_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[None, None, ContractApiMessage]:
        """
        Request contract safe transaction hash

        Happy-path full flow of the messages.

        AbstractRoundAbci skill -> (ContractApiMessage | ContractApiMessage.Performative) -> Ledger connection (contract dispatcher)
        Ledger connection (contract dispatcher) -> (ContractApiMessage | ContractApiMessage.Performative) -> AbstractRoundAbci skill

        :param performative: the message performative
        :param contract_address: the contract address
        :param contract_id: the contract id
        :param contract_callable: the callable to call on the contract
        :param ledger_id: the ledger id, if not specified, the default ledger id is used
        :param kwargs: keyword argument for the contract api request
        :return: the contract api response
        :yields: the contract api response
        """
        contract_api_dialogues = cast(
            ContractApiDialogues, self.context.contract_api_dialogues
        )
        kwargs = {
            "performative": performative,
            "counterparty": LEDGER_API_ADDRESS,
            "ledger_id": ledger_id or self.context.default_ledger_id,
            "contract_id": contract_id,
            "callable": contract_callable,
            "kwargs": ContractApiMessage.Kwargs(kwargs),
        }
        if contract_address is not None:
            kwargs["contract_address"] = contract_address
        contract_api_msg, contract_api_dialogue = contract_api_dialogues.create(
            **kwargs
        )
        contract_api_dialogue = cast(
            ContractApiDialogue,
            contract_api_dialogue,
        )
        contract_api_dialogue.terms = self._get_default_terms()
        request_nonce = self._get_request_nonce_from_dialogue(contract_api_dialogue)
        cast(Requests, self.context.requests).request_id_to_callback[
            request_nonce
        ] = self.get_callback_request()
        self.context.outbox.put_message(message=contract_api_msg)
        response = yield from self.wait_for_message()
        return response
