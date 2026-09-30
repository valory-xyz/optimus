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

"""Tests for behaviours/check_mech_pre_deposit.py."""

# pylint: skip-file

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.valory.contracts.gnosis_safe.contract import SafeOperation
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.liquidity_trader_abci.behaviours.check_mech_pre_deposit import (
    CheckMechPreDepositBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.models import Coingecko
from packages.valory.skills.liquidity_trader_abci.states.base import Event
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    skill_input_hex_to_payload,
)

_AGENT = "0xagent"
_CHAIN = "optimism"
_SAFE = "0x" + "5a" * 20
_MARKETPLACE = "0x" + "11" * 20
_TRACKER = "0x" + "22" * 20
_TOKEN = "0x" + "33" * 20
_MULTISEND = "0x" + "44" * 20
_ZERO = "0x" + "00" * 20
_SAFE_TX_HASH = "0x" + "ab" * 32
_PAYMENT_TYPE = "0x6406bb5f" + "31" * 28

# Distinct calldata per call so a test can tell the approve from the deposit.
_DEPOSIT_DATA = bytes.fromhex("2f4f21e2" + "aa" * 8)
_APPROVE_DATA = bytes.fromhex("095ea7b3" + "bb" * 8)
_MULTISEND_DATA = "0x8d80ff0a" + "cc" * 8

_FLOOR = 500_000
_TARGET = 2_000_000
_CAP = 5_000_000

_DEPOSIT_CALL = "build_deposit_for_data"
_APPROVE_CALL = "build_approval_tx"
_TRACKER_CALL = "get_balance_tracker_for_mech_type"
_TOKEN_CALL = "get_token_address"
_MULTISEND_CALL = "get_tx_data"
_SAFE_HASH_CALL = "get_raw_safe_transaction_hash"
_BALANCE_CALL = "check_balance"

# The key each contract callable actually puts its result under. A wrong
# ``data_key`` reads ``None`` in production while a name-keyed stub would still
# answer, so the stub checks it.
_DATA_KEYS = {
    _TRACKER_CALL: "data",
    _DEPOSIT_CALL: "data",
    _APPROVE_CALL: "data",
    _TOKEN_CALL: "token_address",
    _BALANCE_CALL: "token",
    _MULTISEND_CALL: "data",
    _SAFE_HASH_CALL: "tx_hash",
}

# Enough for any top-up these tests ask for, so the balance clamp only bites
# where a test sets it deliberately.
_AMPLE_BALANCE = 100_000_000


def _requester_info(
    balance: int,
    marketplace: str = _MARKETPLACE,
    payment_type: str = _PAYMENT_TYPE,
) -> bytes:
    """Build the facilitator's requester-info body.

    :param balance: what the balance tracker holds for this Safe.
    :param marketplace: the marketplace the facilitator serves.
    :param payment_type: the mech's payment type.
    :return: the encoded response body.
    """
    return json.dumps(
        {
            "balance": str(balance),
            "marketplace_address": marketplace,
            "payment_type": payment_type,
            "next_nonce": 7,
        }
    ).encode()


def _token_answers(**overrides: Any) -> Dict[str, Any]:
    """Answers for the token-paid tracker path, overridable per test.

    :param overrides: contract callables to answer differently.
    :return: the callable-to-answer mapping for ``_ContractStub``.
    """
    answers: Dict[str, Any] = {
        _TRACKER_CALL: _TRACKER,
        _DEPOSIT_CALL: _DEPOSIT_DATA,
        _TOKEN_CALL: _TOKEN,
        _BALANCE_CALL: _AMPLE_BALANCE,
        _APPROVE_CALL: _APPROVE_DATA,
        _MULTISEND_CALL: _MULTISEND_DATA,
        _SAFE_HASH_CALL: _SAFE_TX_HASH,
    }
    answers.update(overrides)
    return answers


def _native_answers(**overrides: Any) -> Dict[str, Any]:
    """Answers for a native-paid tracker, which reports no token.

    :param overrides: contract callables to answer differently.
    :return: the callable-to-answer mapping for ``_ContractStub``.
    """
    answers = _token_answers(**{_TOKEN_CALL: _ZERO})
    answers.update(overrides)
    return answers


class _ContractStub:
    """Answer ``contract_interact`` by callable name and record every call.

    Keyed by name rather than by call order so the stub does not encode a
    sequence the production code is free to change. An unexpected callable
    raises, which is how a test asserts that a read never happens.
    """

    def __init__(self, answers: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the stub.

        :param answers: callable name to the value ``contract_interact`` returns.
        """
        self.answers = answers or {}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> Any:
        """Record the call and return its answer.

        :param kwargs: the keyword arguments the behaviour passed.
        :yield: once, as a real contract read would.
        :return: the configured answer.
        """
        name = kwargs["contract_callable"]
        self.calls.append((name, kwargs))
        if name not in self.answers:
            raise AssertionError(f"unexpected contract read: {name}")
        assert kwargs["data_key"] == _DATA_KEYS[name], (
            f"{name} returns its result under {_DATA_KEYS[name]!r}, "
            f"not {kwargs['data_key']!r}"
        )
        yield
        return self.answers[name]

    @property
    def names(self) -> List[str]:
        """Return the callables invoked, in order.

        :return: the callable names.
        """
        return [name for name, _ in self.calls]

    def kwargs_for(self, name: str) -> Dict[str, Any]:
        """Return the kwargs of the single call to ``name``.

        :param name: the contract callable.
        :return: that call's keyword arguments.
        """
        matches = [kw for called, kw in self.calls if called == name]
        assert len(matches) == 1, f"{name} called {len(matches)} times"
        return matches[0]


class _HttpStub:
    """Answer ``get_http_response`` with a fixed response and record the URLs."""

    def __init__(self, status_code: Optional[int], body: bytes = b"{}") -> None:
        """Initialise the stub.

        :param status_code: the status to report, or ``None`` for no response.
        :param body: the response body.
        """
        self.status_code = status_code
        self.body = body
        self.urls: List[str] = []

    def __call__(self, **kwargs: Any) -> Any:
        """Record the URL and return the configured response.

        :param kwargs: the keyword arguments the behaviour passed.
        :yield: once, as a real HTTP call would.
        :return: the response, or ``None``.
        """
        self.urls.append(kwargs["url"])
        yield
        if self.status_code is None:
            return None
        return SimpleNamespace(status_code=self.status_code, body=self.body)


def _params(**overrides: Any) -> MagicMock:
    """Build a stub of the skill params the behaviour reads.

    :param overrides: params attributes to set differently.
    :return: the params stub.
    """
    params = MagicMock()
    params.safe_contract_addresses = {_CHAIN: _SAFE}
    params.multisend_contract_addresses = {_CHAIN: _MULTISEND}
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


def _coingecko(**overrides: Any) -> Coingecko:
    """Build the real Coingecko model that carries the mech configuration.

    The mech settings live on this model rather than on the skill params, so
    the tests bind to the real attribute names: moving or renaming one breaks
    them instead of passing against a permissive mock.

    :param overrides: model kwargs to set differently.
    :return: the configured model.
    """
    kwargs: Dict[str, Any] = {
        "token_price_endpoint": "/api/v3/simple/token_price",
        "coin_price_endpoint": "/api/v3/simple/price",
        "api_key": None,
        "rate_limited_code": 429,
        "historical_price_endpoint": "/api/v3/coins/{coin_id}/history",
        "historical_market_data_endpoint": "/api/v3/coins/{coin_id}/market_chart",
        "chain_to_platform_id_mapping": json.dumps({_CHAIN: "optimistic-ethereum"}),
        "requests_per_minute": 30,
        "credits": 10000,
        "use_x402": True,
        "network_selector": _CHAIN,
        "mech_chain": _CHAIN,
        "coingecko_server_base_url": "https://api.coingecko.com",
        "coingecko_x402_server_base_url": "https://x402.example/{chain}",
        "coin_from_address_endpoint": "/api/v3/coins/{platform}/contract/{address}",
        "use_mech_facilitator": True,
        "mech_facilitator_base_url": "https://facilitator.example/",
        "mech_max_delivery_rate": 100_000,
        "mech_pre_deposit_floor": _FLOOR,
        "mech_pre_deposit_target": _TARGET,
        "mech_pre_deposit_cap": _CAP,
    }
    kwargs.update(overrides)
    return Coingecko(name="coingecko", skill_context=MagicMock(), **kwargs)


class _LedgerStub:
    """Answer ``get_ledger_api_response`` with a native balance."""

    def __init__(self, balance: Optional[int]) -> None:
        """Initialise the stub.

        :param balance: the balance to report, or ``None`` to fail the read.
        """
        self.balance = balance
        self.accounts: List[str] = []

    def __call__(self, **kwargs: Any) -> Any:
        """Record the account and answer the read.

        :param kwargs: the keyword arguments the behaviour passed.
        :yield: once, as a real ledger read would.
        :return: the ledger API message.
        """
        self.accounts.append(kwargs["account"])
        yield
        if self.balance is None:
            return SimpleNamespace(
                performative=LedgerApiMessage.Performative.ERROR, state=None
            )
        return SimpleNamespace(
            performative=LedgerApiMessage.Performative.STATE,
            state=SimpleNamespace(body={"get_balance_result": self.balance}),
        )


def _make_behaviour(
    contracts: Optional[_ContractStub] = None,
    http: Optional[_HttpStub] = None,
    investing_paused: bool = False,
    coingecko: Optional[Coingecko] = None,
    native_balance: Optional[int] = _AMPLE_BALANCE,
) -> CheckMechPreDepositBehaviour:
    """Create the behaviour without ``__init__`` and stub only its boundaries.

    Everything between the HTTP read and the settled payload runs for real, so
    a mutation anywhere in the deposit arithmetic or the transaction shape
    shows up in the decoded payload.

    :param contracts: the contract-read stub.
    :param http: the facilitator-read stub.
    :param investing_paused: what the withdrawal gate reads.
    :param coingecko: the mech configuration model.
    :param native_balance: what the Safe holds natively, or ``None`` to fail
        that read.
    :return: the behaviour under test.
    """
    obj = object.__new__(CheckMechPreDepositBehaviour)
    obj.__dict__["_context"] = MagicMock()
    obj.context.coingecko = coingecko or _coingecko()

    def _paused() -> Any:
        yield
        return investing_paused

    obj._read_investing_paused = _paused
    obj.contract_interact = contracts or _ContractStub()
    obj.get_http_response = http or _HttpStub(200, _requester_info(_TARGET))
    obj.get_ledger_api_response = _LedgerStub(native_balance)
    return obj


def _drive(gen: Any) -> Any:
    """Drive a generator to completion.

    :param gen: the generator.
    :return: its return value.
    """
    value = None
    while True:
        try:
            value = gen.send(value)
        except StopIteration as exc:
            return exc.value


def _run(obj: CheckMechPreDepositBehaviour, params: MagicMock) -> Any:
    """Drive ``async_act`` with ``params`` patched and return the payload sent.

    :param obj: the behaviour under test.
    :param params: the params stub.
    :return: the single payload passed to ``send_a2a_transaction``.
    """
    sent: List[Any] = []

    with patch.object(
        type(obj), "params", new_callable=PropertyMock, return_value=params
    ):
        obj.context.benchmark_tool.measure.return_value = MagicMock()
        obj.context.agent_address = _AGENT

        def fake_send(payload: Any) -> Any:
            """Record the payload instead of broadcasting it.

            :param payload: the payload the behaviour built.
            :yield: once, as the real send would.
            """
            sent.append(payload)
            yield

        def fake_wait(*args: Any, **kwargs: Any) -> Any:
            """Return immediately instead of waiting for the round.

            :param args: ignored.
            :param kwargs: ignored.
            :yield: once.
            """
            yield

        obj.send_a2a_transaction = fake_send
        obj.wait_until_round_end = fake_wait
        obj.set_done = MagicMock()
        _drive(obj.async_act())

    assert len(sent) == 1
    obj.set_done.assert_called_once()
    return sent[0]


class TestNoTopUpNeeded:
    """Cases where the behaviour must settle nothing."""

    def test_disabled_facilitator_reads_nothing_at_all(self) -> None:
        """With the facilitator off there is no deposit to keep, so no reads."""
        http = _HttpStub(200, _requester_info(0))
        contracts = _ContractStub()
        obj = _make_behaviour(
            contracts, http, coingecko=_coingecko(use_mech_facilitator=False)
        )

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert http.urls == []
        assert contracts.names == []

    def test_pre_deposit_at_the_floor_is_left_alone(self) -> None:
        """At exactly the floor the deposit still covers the next call."""
        contracts = _ContractStub()
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(_FLOOR)))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == []

    def test_a_target_below_the_floor_deposits_nothing(self) -> None:
        """A target under the floor would ask for a non-positive deposit."""
        contracts = _ContractStub()
        obj = _make_behaviour(
            contracts,
            _HttpStub(200, _requester_info(_FLOOR - 1)),
            coingecko=_coingecko(mech_pre_deposit_target=_FLOOR - 1),
        )

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == []

    def test_missing_safe_for_the_mech_chain_reads_nothing(self) -> None:
        """Without a Safe on the mech chain there is nobody to deposit for."""
        http = _HttpStub(200, _requester_info(0))
        obj = _make_behaviour(_ContractStub(), http)

        payload = _run(obj, _params(safe_contract_addresses={"base": _SAFE}))

        assert payload.tx_hash is None
        assert http.urls == []
        obj.context.logger.warning.assert_called()

    def test_withdrawal_request_leaves_the_deposit_alone(self) -> None:
        """A pending withdrawal must not push the Safe's token into the tracker."""
        http = _HttpStub(200, _requester_info(0))
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, http, investing_paused=True)

        payload = _run(obj, _params())

        assert payload.event == Event.WITHDRAWAL_INITIATED.value
        assert payload.tx_hash is None
        assert payload.safe_contract_address is None
        assert http.urls == []
        assert contracts.names == []


class TestFacilitatorReadFailures:
    """An unreadable facilitator must skip the period, not deposit blind."""

    def test_the_requester_url_names_the_chain_and_the_safe(self) -> None:
        """The facilitator is asked about the Safe that pays, on the mech chain."""
        http = _HttpStub(200, _requester_info(_TARGET))
        obj = _make_behaviour(_ContractStub(), http)

        _run(obj, _params())

        assert http.urls == [
            f"https://facilitator.example/mech/{_CHAIN}/requester/{_SAFE}"
        ]

    @pytest.mark.parametrize("status_code", [None, 404, 500, 502])
    def test_a_failed_read_settles_nothing(self, status_code: Optional[int]) -> None:
        """No response, or a non-OK one, skips the check for this period.

        :param status_code: the status the facilitator returns.
        """
        contracts = _ContractStub()
        obj = _make_behaviour(contracts, _HttpStub(status_code, b"{}"))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == []
        obj.context.logger.warning.assert_called()

    @pytest.mark.parametrize(
        "body",
        [
            b"not json",
            b"",
            b"[1, 2, 3]",
        ],
    )
    def test_an_unparseable_body_settles_nothing(self, body: bytes) -> None:
        """A body that is not a JSON object cannot say what the deposit is.

        :param body: the malformed response body.
        """
        contracts = _ContractStub()
        obj = _make_behaviour(contracts, _HttpStub(200, body))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == []
        obj.context.logger.warning.assert_called()

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps({"marketplace_address": _MARKETPLACE}).encode(),
            json.dumps({"balance": "10", "payment_type": _PAYMENT_TYPE}).encode(),
            json.dumps(
                {
                    "balance": "not a number",
                    "marketplace_address": _MARKETPLACE,
                    "payment_type": _PAYMENT_TYPE,
                }
            ).encode(),
            json.dumps(
                {
                    "balance": None,
                    "marketplace_address": _MARKETPLACE,
                    "payment_type": _PAYMENT_TYPE,
                }
            ).encode(),
        ],
    )
    def test_requester_info_missing_a_field_settles_nothing(self, body: bytes) -> None:
        """Every field the deposit needs has to be present and numeric.

        :param body: a response body missing or corrupting one field.
        """
        contracts = _ContractStub()
        obj = _make_behaviour(contracts, _HttpStub(200, body))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == []
        obj.context.logger.warning.assert_called()

    @pytest.mark.parametrize("tracker", [_ZERO, "", None])
    def test_an_unregistered_payment_type_settles_nothing(self, tracker: Any) -> None:
        """No tracker for this payment type means nowhere to send the deposit.

        :param tracker: what the marketplace reports for the payment type.
        """
        contracts = _ContractStub(_token_answers(**{_TRACKER_CALL: tracker}))
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert contracts.names == [_TRACKER_CALL]
        obj.context.logger.warning.assert_called()

    def test_the_tracker_is_resolved_from_the_reported_payment_type(self) -> None:
        """Trackers are keyed by payment type, so the read must pass it through."""
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        _run(obj, _params())

        kwargs = contracts.kwargs_for(_TRACKER_CALL)
        assert kwargs["contract_address"] == _MARKETPLACE
        assert kwargs["mech_type"] == _PAYMENT_TYPE
        assert kwargs["chain_id"] == _CHAIN


class TestTokenPaidTopUp:
    """A token-paid tracker needs the allowance and the deposit in one tx."""

    def test_a_short_deposit_settles_an_approve_and_deposit_multisend(self) -> None:
        """The multisend is the Safe's target, delegate-called, with no value."""
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.safe_contract_address == _SAFE
        assert payload.chain_id == _CHAIN
        decoded = skill_input_hex_to_payload(payload.tx_hash)
        assert decoded["to_address"] == _MULTISEND
        assert decoded["operation"] == SafeOperation.DELEGATE_CALL.value
        assert decoded["ether_value"] == 0
        assert decoded["data"] == bytes.fromhex(_MULTISEND_DATA[2:])
        assert decoded["safe_tx_hash"] == _SAFE_TX_HASH[2:]
        # The hash the Safe signed has to be over the same gas terms the
        # settlement skill will submit, or the signature does not verify.
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["safe_tx_gas"] == (
            decoded["safe_tx_gas"]
        )
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["value"] == decoded["ether_value"]
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["to_address"] == (
            decoded["to_address"]
        )
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["operation"] == (
            decoded["operation"]
        )
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["data"] == decoded["data"]
        assert contracts.kwargs_for(_SAFE_HASH_CALL)["contract_address"] == _SAFE

    def test_the_approve_precedes_the_deposit_and_matches_its_amount(self) -> None:
        """A deposit without its allowance reverts, and a stale allowance lingers."""
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        _run(obj, _params())

        calls = contracts.kwargs_for(_MULTISEND_CALL)["multi_send_txs"]
        assert [call["to"] for call in calls] == [_TOKEN, _TRACKER]
        assert [call["data"] for call in calls] == [_APPROVE_DATA, _DEPOSIT_DATA]
        assert [call["value"] for call in calls] == [0, 0]
        assert contracts.kwargs_for(_APPROVE_CALL)["spender"] == _TRACKER
        assert (
            contracts.kwargs_for(_APPROVE_CALL)["amount"]
            == contracts.kwargs_for(_DEPOSIT_CALL)["amount"]
        )

    def test_the_deposit_credits_the_safe_rather_than_the_sender(self) -> None:
        """The marketplace debits the requester Safe, so it has to be the account."""
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        _run(obj, _params())

        assert contracts.kwargs_for(_DEPOSIT_CALL)["account"] == _SAFE
        assert contracts.kwargs_for(_DEPOSIT_CALL)["contract_address"] == _TRACKER

    @pytest.mark.parametrize(
        ("deposited", "cap", "expected"),
        [
            (0, _CAP, _TARGET),
            (_FLOOR - 1, _CAP, _TARGET - (_FLOOR - 1)),
            (0, 100_000, 100_000),
            (_FLOOR - 1, 100_000, 100_000),
            (0, _TARGET, _TARGET),
        ],
    )
    def test_the_top_up_fills_to_target_but_never_exceeds_the_cap(
        self, deposited: int, cap: int, expected: int
    ) -> None:
        """One cycle deposits at most the cap, and never overshoots the target.

        :param deposited: what the tracker already holds.
        :param cap: the per-cycle ceiling.
        :param expected: the amount the deposit must carry.
        """
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(
            contracts,
            _HttpStub(200, _requester_info(deposited)),
            coingecko=_coingecko(mech_pre_deposit_cap=cap),
        )

        _run(obj, _params())

        assert contracts.kwargs_for(_DEPOSIT_CALL)["amount"] == expected

    @pytest.mark.parametrize(
        "failing", [_DEPOSIT_CALL, _APPROVE_CALL, _MULTISEND_CALL, _SAFE_HASH_CALL]
    )
    def test_any_encoding_failure_settles_nothing(self, failing: str) -> None:
        """A half-built transaction must never reach settlement.

        :param failing: the contract read that returns nothing.
        """
        contracts = _ContractStub(_token_answers(**{failing: None}))
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        obj.context.logger.error.assert_called()


class TestNativePaidTopUp:
    """A native-paid tracker takes the deposit as transaction value."""

    def test_a_native_tracker_sends_value_instead_of_approving(self) -> None:
        """There is nothing to approve, and the value has to carry the amount."""
        contracts = _ContractStub(_native_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        decoded = skill_input_hex_to_payload(payload.tx_hash)
        assert decoded["to_address"] == _TRACKER
        assert decoded["operation"] == SafeOperation.CALL.value
        assert decoded["ether_value"] == _TARGET
        assert decoded["data"] == _DEPOSIT_DATA
        assert _APPROVE_CALL not in contracts.names
        assert _MULTISEND_CALL not in contracts.names

    def test_a_native_top_up_is_capped_the_same_way(self) -> None:
        """The value sent is the capped amount, not the full shortfall."""
        contracts = _ContractStub(_native_answers())
        obj = _make_behaviour(
            contracts,
            _HttpStub(200, _requester_info(0)),
            coingecko=_coingecko(mech_pre_deposit_cap=100_000),
        )

        payload = _run(obj, _params())

        decoded = skill_input_hex_to_payload(payload.tx_hash)
        assert decoded["ether_value"] == 100_000
        assert contracts.kwargs_for(_DEPOSIT_CALL)["amount"] == 100_000

    def test_a_native_safe_hash_failure_settles_nothing(self) -> None:
        """The native path guards its Safe hash too.

        Covers ``_safe_tx`` reached without the multisend in between.
        """
        contracts = _ContractStub(_native_answers(**{_SAFE_HASH_CALL: None}))
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        obj.context.logger.error.assert_called()


class TestSafeBalanceBound:
    """The deposit is bounded by what the Safe actually holds.

    The shortfall comes from the tracker, which says nothing about whether the
    Safe has been funded. A deposit above the Safe's balance reverts when it
    settles, so it has to be refused or trimmed before it gets that far.
    """

    def test_a_token_deposit_is_trimmed_to_the_safe_balance(self) -> None:
        """A Safe holding less than the shortfall deposits what it has."""
        held = 120_000
        contracts = _ContractStub(_token_answers(**{_BALANCE_CALL: held}))
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.tx_hash is not None
        assert contracts.kwargs_for(_DEPOSIT_CALL)["amount"] == held
        assert contracts.kwargs_for(_APPROVE_CALL)["amount"] == held

    def test_the_token_balance_is_read_for_the_safe_and_the_tracker_token(self) -> None:
        """The balance that matters is the Safe's, in the tracker's own token."""
        contracts = _ContractStub(_token_answers())
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        _run(obj, _params())

        kwargs = contracts.kwargs_for(_BALANCE_CALL)
        assert kwargs["account"] == _SAFE
        assert kwargs["contract_address"] == _TOKEN
        assert kwargs["chain_id"] == _CHAIN

    @pytest.mark.parametrize("held", [0, None])
    def test_no_token_balance_settles_nothing(self, held: Any) -> None:
        """An empty or unreadable Safe balance skips rather than deposits.

        :param held: what the token balance read reports.
        """
        contracts = _ContractStub(_token_answers(**{_BALANCE_CALL: held}))
        obj = _make_behaviour(contracts, _HttpStub(200, _requester_info(0)))

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert _DEPOSIT_CALL not in contracts.names
        obj.context.logger.warning.assert_called()

    def test_a_native_deposit_is_trimmed_to_the_safe_balance(self) -> None:
        """The value sent can never exceed what the Safe holds natively."""
        held = 90_000
        contracts = _ContractStub(_native_answers())
        obj = _make_behaviour(
            contracts, _HttpStub(200, _requester_info(0)), native_balance=held
        )

        payload = _run(obj, _params())

        decoded = skill_input_hex_to_payload(payload.tx_hash)
        assert decoded["ether_value"] == held
        assert contracts.kwargs_for(_DEPOSIT_CALL)["amount"] == held
        assert obj.get_ledger_api_response.accounts == [_SAFE]
        assert _BALANCE_CALL not in contracts.names

    @pytest.mark.parametrize("held", [0, None])
    def test_no_native_balance_settles_nothing(self, held: Any) -> None:
        """An empty or unreadable native balance skips rather than deposits.

        :param held: what the native balance read reports.
        """
        contracts = _ContractStub(_native_answers())
        obj = _make_behaviour(
            contracts, _HttpStub(200, _requester_info(0)), native_balance=held
        )

        payload = _run(obj, _params())

        assert payload.tx_hash is None
        assert _DEPOSIT_CALL not in contracts.names
        obj.context.logger.warning.assert_called()
