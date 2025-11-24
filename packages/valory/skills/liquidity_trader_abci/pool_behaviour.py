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

"""This package contains the implemenatation of the PoolBehaviour interface."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Optional, cast

from aea.configurations.data_types import PublicId
from aea_ledger_ethereum.ethereum import EthereumCrypto
from eth_account import Account

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.behaviours import BaseBehaviour
from packages.valory.skills.liquidity_trader_abci.models import Coingecko


WaitableConditionType = Generator[None, None, Any]


class PoolBehaviour(BaseBehaviour, ABC):
    """PoolBehaviour"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize `PoolBehaviour`."""
        super().__init__(**kwargs)

    @abstractmethod
    def _get_tokens(self) -> Dict[str, str]:
        """Get the pool tokens"""
        pass

    @abstractmethod
    def enter(self, **kwargs: Any) -> Generator[None, None, str]:
        """Enter pool"""
        pass

    @abstractmethod
    def exit(self, **kwargs: Any) -> None:
        """Exit pool"""
        pass

    @property
    def eoa_account(self) -> Optional[Account]:
        """Get EOA account from encrypted private key using password from command line."""
        default_ledger = self.context.default_ledger_id
        eoa_file = Path(self.context.data_dir) / f"{default_ledger}_private_key.txt"

        # Get password from command line arguments
        password = self._get_password_from_args()

        if password is None:
            # No password provided, try to read as plain private key
            try:
                with eoa_file.open("r") as f:
                    private_key = f.read().strip()

                account = Account.from_key(private_key=private_key)
                return account

            except Exception as e:
                self.context.logger.error(
                    f"Failed to read as plain private key. Error: {e}"
                )
                return None

        # Password provided, try to decrypt encrypted keyfile
        try:
            crypto = EthereumCrypto(private_key_path=str(eoa_file), password=password)
            private_key = crypto.private_key
            return Account.from_key(private_key)
        except Exception as e:
            self.context.logger.error(
                f"Failed to decrypt private key with password: {e}"
            )
            return None

    def _get_password_from_args(self) -> Optional[str]:
        """Extract password from command line arguments."""
        args = sys.argv
        try:
            password_index = args.index("--password")
            if password_index + 1 < len(args):
                return args[password_index + 1]
        except ValueError:
            pass

        for arg in args:
            if arg.startswith("--password="):
                return arg.split("=", 1)[1]

        return None

    @property
    def coingecko(self) -> Coingecko:
        """Return the Coingecko."""
        return cast(Coingecko, self.context.coingecko)

    def default_error(
        self, contract_id: str, contract_callable: str, response_msg: ContractApiMessage
    ) -> None:
        """Return a default contract interaction error message."""
        self.context.logger.error(
            f"Could not successfully interact with the {contract_id} contract "
            f"using {contract_callable!r}: {response_msg}"
        )

    def contract_interaction_error(
        self, contract_id: str, contract_callable: str, response_msg: ContractApiMessage
    ) -> None:
        """Return a contract interaction error message."""
        # contracts can only return one message, i.e., multiple levels cannot exist.
        for level in ("info", "warning", "error"):
            msg = response_msg.raw_transaction.body.get(level, None)
            logger = getattr(self.context.logger, level)
            if msg is not None:
                logger(msg)
                return

        self.default_error(contract_id, contract_callable, response_msg)

    def contract_interact(
        self,
        performative: ContractApiMessage.Performative,
        contract_address: str,
        contract_public_id: PublicId,
        contract_callable: str,
        data_key: str,
        **kwargs: Any,
    ) -> WaitableConditionType:
        """Interact with a contract."""
        contract_id = str(contract_public_id)

        self.context.logger.info(
            f"Interacting with contract {contract_id} at address {contract_address}\n"
            f"Calling method {contract_callable} with parameters: {kwargs}"
        )

        response_msg = yield from self.get_contract_api_response(
            performative,
            contract_address,
            contract_id,
            contract_callable,
            **kwargs,
        )

        self.context.logger.info(f"Contract response: {response_msg}")

        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.default_error(contract_id, contract_callable, response_msg)
            return None

        data = response_msg.raw_transaction.body.get(data_key, None)
        if data is None:
            self.contract_interaction_error(
                contract_id, contract_callable, response_msg
            )
            return None

        return data

    def async_act(self) -> Generator[Any, None, None]:
        """Async act"""
        pass
