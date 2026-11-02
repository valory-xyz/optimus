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

"""Tests for utils/multicall.py."""

from eth_abi import encode
from eth_utils import keccak

from packages.valory.skills.liquidity_trader_abci.utils.multicall import (
    Read,
    decode_results,
    encode_read,
)

TARGET = "0x4857DB576eE0e45F10525eC59527BfD10beeC60b"
OWNER = "0x7fe908a35C335df9AAeDAcC83230fb8a8e944080"


def test_encode_read_prefixes_the_selector_and_allows_failure() -> None:
    """A read becomes (target, selector + abi-encoded args, True)."""
    read = Read(TARGET, "tokenOfOwnerByIndex(address,uint256)", (OWNER, 3), ["uint256"])

    target, call_data, allow_failure = encode_read(read)

    assert target == TARGET
    assert call_data == keccak(text="tokenOfOwnerByIndex(address,uint256)")[
        :4
    ] + encode(["address", "uint256"], [OWNER, 3])
    assert allow_failure is True


def test_encode_read_without_arguments() -> None:
    """A read with no arguments is just the selector."""
    _, call_data, _ = encode_read(Read(TARGET, "factory()", (), ["address"]))
    assert call_data == keccak(text="factory()")[:4]


def test_decode_results_decodes_each_answer() -> None:
    """Each answer is decoded with its read's output types."""
    reads = [
        Read(TARGET, "balanceOf(address)", (OWNER,), ["uint256"]),
        Read(TARGET, "stakedValues(address)", (OWNER,), ["uint256[]"]),
    ]
    results = [
        (True, encode(["uint256"], [2])),
        (True, encode(["uint256[]"], [[57869, 57870]])),
    ]
    assert decode_results(reads, results) == [(2,), ((57869, 57870),)]


def test_decode_results_rejects_a_failed_or_undecodable_read() -> None:
    """One failed, empty or missing answer fails the whole batch."""
    read = Read(TARGET, "factory()", (), ["address"])
    assert decode_results([read], [(False, b"")]) is None
    assert decode_results([read], [(True, b"")]) is None
    assert decode_results([read, read], [(True, encode(["address"], [OWNER]))]) is None
