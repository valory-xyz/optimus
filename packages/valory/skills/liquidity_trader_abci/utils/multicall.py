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

"""Encoding and decoding of the reads batched through Multicall3."""

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

from eth_abi import decode, encode
from eth_utils import keccak

MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"


class Read(NamedTuple):
    """A contract read: ``signature`` is the flat function signature, e.g. ``balanceOf(address)``."""

    target: str
    signature: str
    args: Tuple[Any, ...]
    output_types: List[str]


def _arg_types(signature: str) -> List[str]:
    """Return the argument types of a flat function signature."""
    inner = signature[signature.index("(") + 1 : -1]
    return [arg_type for arg_type in inner.split(",") if arg_type]


def encode_read(read: Read) -> Tuple[str, bytes, bool]:
    """Return ``read`` as a Multicall3 ``(target, call_data, allow_failure)`` call."""
    call_data = keccak(text=read.signature)[:4] + encode(
        _arg_types(read.signature), list(read.args)
    )
    return read.target, call_data, True


def decode_results(
    reads: Sequence[Read], results: Sequence[Tuple[bool, bytes]]
) -> Optional[List[Tuple[Any, ...]]]:
    """Decode the answers to ``reads``; None when any read failed or is undecodable."""
    if len(results) != len(reads):
        return None
    decoded = []
    for read, (success, data) in zip(reads, results):
        if not success:
            return None
        try:
            decoded.append(decode(read.output_types, data))
        except Exception:  # pylint: disable=broad-except
            return None
    return decoded
