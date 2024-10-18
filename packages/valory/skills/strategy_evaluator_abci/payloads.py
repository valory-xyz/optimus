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

"""This module contains the transaction payloads for the strategy evaluator."""

from dataclasses import dataclass
from typing import Optional

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload


@dataclass(frozen=True)
class IPFSHashPayload(BaseTxPayload):
    """Represents a transaction payload for an IPFS hash."""

    ipfs_hash: Optional[str]
    incomplete: Optional[bool]


@dataclass(frozen=True)
class SendSwapProxyPayload(BaseTxPayload):
    """Represents a transaction payload for attempting a swap transaction via the proxy server."""

    tx_id: Optional[str]


@dataclass(frozen=True)
class SendSwapPayload(BaseTxPayload):
    """Represents a transaction payload for preparing the instruction for a swap transaction."""

    # `instructions` is a serialized `List[Dict[str, Any]]`
    instructions: Optional[str]


@dataclass(frozen=True)
class TransactionHashPayload(BaseTxPayload):
    """Represent a transaction payload of type 'tx_hash'."""

    tx_hash: Optional[str]
