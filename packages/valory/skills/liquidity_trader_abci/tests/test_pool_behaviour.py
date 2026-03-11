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

"""Test the pool_behaviour.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import sys
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

from packages.valory.protocols.contract_api.message import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.pool_behaviour import (
    PoolBehaviour,
)


class ConcretePoolBehaviour(PoolBehaviour):
    """Concrete implementation of PoolBehaviour for testing."""

    matching_round = MagicMock(spec=AbstractRound)

    def _get_tokens(self) -> Dict[str, str]:
        return {"token0": "0xaddr0", "token1": "0xaddr1"}

    def enter(self, **kwargs: Any) -> Generator[None, None, str]:
        yield
        return "tx_hash"

    def exit(self, **kwargs: Any) -> None:
        pass


def test_import() -> None:
    """Test that the pool_behaviour module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.pool_behaviour  # noqa


class TestPoolBehaviour:
    """Test PoolBehaviour class."""

    def _make_behaviour(self) -> ConcretePoolBehaviour:
        """Create a ConcretePoolBehaviour instance without calling __init__."""
        obj = object.__new__(ConcretePoolBehaviour)
        obj.__dict__["_context"] = MagicMock()
        return obj

    def test_coingecko_property(self) -> None:
        """Test coingecko property returns context.coingecko."""
        obj = self._make_behaviour()
        mock_cg = MagicMock()
        obj.context.coingecko = mock_cg
        assert obj.coingecko is mock_cg

    def test_default_error(self) -> None:
        """Test default_error logs error message."""
        obj = self._make_behaviour()
        response_msg = MagicMock()
        obj.default_error("contract_id", "callable", response_msg)
        obj.context.logger.error.assert_called_once()

    def test_contract_interaction_error_info(self) -> None:
        """Test contract_interaction_error logs info level message."""
        obj = self._make_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body.get.side_effect = (
            lambda k, default=None: "info message" if k == "info" else None
        )
        obj.contract_interaction_error("contract_id", "callable", response_msg)
        obj.context.logger.info.assert_called_once_with("info message")

    def test_contract_interaction_error_warning(self) -> None:
        """Test contract_interaction_error logs warning level message."""
        obj = self._make_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body.get.side_effect = (
            lambda k, default=None: "warn message" if k == "warning" else None
        )
        obj.contract_interaction_error("contract_id", "callable", response_msg)
        obj.context.logger.warning.assert_called_once_with("warn message")

    def test_contract_interaction_error_error(self) -> None:
        """Test contract_interaction_error logs error level message."""
        obj = self._make_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body.get.side_effect = (
            lambda k, default=None: "err message" if k == "error" else None
        )
        obj.contract_interaction_error("contract_id", "callable", response_msg)
        obj.context.logger.error.assert_called_once_with("err message")

    def test_contract_interaction_error_no_message(self) -> None:
        """Test contract_interaction_error falls through to default_error."""
        obj = self._make_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body.get.return_value = None
        obj.contract_interaction_error("contract_id", "callable", response_msg)
        # default_error should be called
        obj.context.logger.error.assert_called_once()

    def test_get_password_from_args_with_positional(self) -> None:
        """Test _get_password_from_args with --password <value>."""
        obj = self._make_behaviour()
        with patch.object(sys, "argv", ["script", "--password", "mypass"]):
            result = obj._get_password_from_args()
        assert result == "mypass"

    def test_get_password_from_args_with_equals(self) -> None:
        """Test _get_password_from_args with --password=value."""
        obj = self._make_behaviour()
        with patch.object(sys, "argv", ["script", "--password=mypass"]):
            result = obj._get_password_from_args()
        assert result == "mypass"

    def test_get_password_from_args_no_password(self) -> None:
        """Test _get_password_from_args returns None when no password."""
        obj = self._make_behaviour()
        with patch.object(sys, "argv", ["script", "--other"]):
            result = obj._get_password_from_args()
        assert result is None

    def test_get_password_from_args_password_at_end(self) -> None:
        """Test _get_password_from_args when --password is last arg."""
        obj = self._make_behaviour()
        with patch.object(sys, "argv", ["script", "--password"]):
            result = obj._get_password_from_args()
        assert result is None

    def test_eoa_account_no_password_success(self) -> None:
        """Test eoa_account reads plain private key when no password."""
        obj = self._make_behaviour()
        obj.context.default_ledger_id = "ethereum"
        obj.context.data_dir = "/tmp/test_data"

        fake_key = "0x" + "a" * 64
        with patch.object(
            PoolBehaviour, "_get_password_from_args", return_value=None
        ), patch.object(
            Path, "open", mock_open(read_data=fake_key)
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.pool_behaviour.Account"
        ) as mock_account:
            mock_account.from_key.return_value = MagicMock()
            result = obj.eoa_account
        assert result is not None
        mock_account.from_key.assert_called_once_with(private_key=fake_key)

    def test_eoa_account_no_password_failure(self) -> None:
        """Test eoa_account returns None when reading plain key fails."""
        obj = self._make_behaviour()
        obj.context.default_ledger_id = "ethereum"
        obj.context.data_dir = "/tmp/test_data"

        with patch.object(
            PoolBehaviour, "_get_password_from_args", return_value=None
        ), patch("builtins.open", side_effect=Exception("file not found")):
            result = obj.eoa_account
        assert result is None
        obj.context.logger.error.assert_called_once()

    def test_eoa_account_with_password_success(self) -> None:
        """Test eoa_account decrypts key with password."""
        obj = self._make_behaviour()
        obj.context.default_ledger_id = "ethereum"
        obj.context.data_dir = "/tmp/test_data"

        with patch.object(
            PoolBehaviour, "_get_password_from_args", return_value="mypass"
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.pool_behaviour.EthereumCrypto"
        ) as mock_crypto_cls, patch(
            "packages.valory.skills.liquidity_trader_abci.pool_behaviour.Account"
        ) as mock_account:
            mock_crypto = MagicMock()
            mock_crypto.private_key = "0x" + "b" * 64
            mock_crypto_cls.return_value = mock_crypto
            mock_account.from_key.return_value = MagicMock()
            result = obj.eoa_account
        assert result is not None

    def test_eoa_account_with_password_failure(self) -> None:
        """Test eoa_account returns None when decryption fails."""
        obj = self._make_behaviour()
        obj.context.default_ledger_id = "ethereum"
        obj.context.data_dir = "/tmp/test_data"

        with patch.object(
            PoolBehaviour, "_get_password_from_args", return_value="badpass"
        ), patch(
            "packages.valory.skills.liquidity_trader_abci.pool_behaviour.EthereumCrypto",
            side_effect=Exception("bad password"),
        ):
            result = obj.eoa_account
        assert result is None
        obj.context.logger.error.assert_called_once()

    def test_contract_interact_success(self) -> None:
        """Test contract_interact returns data on success."""
        obj = self._make_behaviour()

        mock_response = MagicMock()
        mock_response.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        mock_response.raw_transaction.body.get.return_value = "some_data"

        def fake_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response

        obj.get_contract_api_response = fake_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = obj.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0xaddr",
            contract_public_id=PublicId("valory", "test", "0.1.0"),
            contract_callable="some_method",
            data_key="result",
        )
        # Advance through the generator
        try:
            next(gen)
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == "some_data"

    def test_contract_interact_bad_performative(self) -> None:
        """Test contract_interact returns None on bad performative."""
        obj = self._make_behaviour()

        mock_response = MagicMock()
        mock_response.performative = ContractApiMessage.Performative.ERROR

        def fake_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response

        obj.get_contract_api_response = fake_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = obj.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0xaddr",
            contract_public_id=PublicId("valory", "test", "0.1.0"),
            contract_callable="some_method",
            data_key="result",
        )
        try:
            next(gen)
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_contract_interact_missing_data_key(self) -> None:
        """Test contract_interact returns None when data_key not in response."""
        obj = self._make_behaviour()

        mock_response = MagicMock()
        mock_response.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        mock_response.raw_transaction.body.get.return_value = None

        def fake_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response

        obj.get_contract_api_response = fake_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = obj.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0xaddr",
            contract_public_id=PublicId("valory", "test", "0.1.0"),
            contract_callable="some_method",
            data_key="result",
        )
        try:
            next(gen)
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_init(self) -> None:
        """Test __init__ calls super().__init__."""
        with patch.object(PoolBehaviour.__bases__[0], "__init__", return_value=None):
            obj = ConcretePoolBehaviour(name="test", skill_context=MagicMock())
        assert isinstance(obj, ConcretePoolBehaviour)

    def test_async_act(self) -> None:
        """Test async_act returns a generator that does nothing."""
        obj = self._make_behaviour()
        gen = obj.async_act()
        assert gen is None or hasattr(gen, "__next__")
