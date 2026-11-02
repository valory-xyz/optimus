# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2026 Valory AG
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
"""Wrapper for Velodrome CL Pool Manager contract interface."""

import logging
from typing import Dict, List

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi

PUBLIC_ID = PublicId.from_str("valory/velodrome_non_fungible_position_manager:0.1.0")

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

_logger = logging.getLogger(__name__)


def _to_topic(value: int) -> str:
    """Return ``value`` as a 32-byte log topic."""
    return "0x" + format(value, "064x")


class VelodromeNonFungiblePositionManagerContract(Contract):
    """Velodrome CL Pool Manager contract wrapper."""

    contract_id = PUBLIC_ID

    @classmethod
    def _encode_call(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        method_name: str,
        args: tuple,
    ) -> JSONLike:
        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encode_abi(method_name, args=args)
        return dict(tx_hash=bytes.fromhex(data[2:]))

    # ------------------- Methods -------------------

    @classmethod
    def mint(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token0: str,
        token1: str,
        tick_spacing: int,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int,
        sqrt_price_x96: int,
    ) -> JSONLike:
        """Prepare encoded tx for mint."""
        params = (
            token0,
            token1,
            tick_spacing,
            tick_lower,
            tick_upper,
            amount0_desired,
            amount1_desired,
            amount0_min,
            amount1_min,
            recipient,
            deadline,
            sqrt_price_x96,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "mint",
            (params,),
        )

    @classmethod
    def decrease_liquidity(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        deadline: int,
    ) -> JSONLike:
        """Prepare encoded tx for decreaseLiquidity."""
        params = (
            token_id,
            liquidity,
            amount0_min,
            amount1_min,
            deadline,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "decreaseLiquidity",
            (params,),
        )

    @classmethod
    def burn(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare encoded tx for burn."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "burn",
            (token_id,),
        )

    @classmethod
    def collect(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
        recipient: str,
        amount0_max: int,
        amount1_max: int,
    ) -> JSONLike:
        """Prepare encoded tx for collect."""
        params = (
            token_id,
            recipient,
            amount0_max,
            amount1_max,
        )
        return cls._encode_call(
            ledger_api,
            contract_address,
            "collect",
            (params,),
        )

    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get tokens information from a position."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.positions(token_id).call()
        return dict(
            token0=result[2],
            token1=result[3],
            tick_spacing=result[4],
            tick_lower=result[5],
            tick_upper=result[6],
            liquidity=result[7],
        )

    @classmethod
    def balanceOf(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
    ) -> JSONLike:
        """Get the number of NFT positions owned by an address"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        balance = contract_instance.functions.balanceOf(owner).call()
        return dict(balance=balance)

    @classmethod
    def ownerOf(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get the owner of a specific NFT position"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        owner = contract_instance.functions.ownerOf(token_id).call()
        return dict(owner=owner)

    @classmethod
    def get_position(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get the position info"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        position = contract_instance.functions.positions(token_id).call()
        if not position:
            return dict(data={})
        return dict(
            data={
                "nonce": position[0],
                "operator": position[1],
                "token0": position[2],
                "token1": position[3],
                "tickSpacing": position[4],
                "tickLower": position[5],
                "tickUpper": position[6],
                "liquidity": position[7],
                "feeGrowthInside0LastX128": position[8],
                "feeGrowthInside1LastX128": position[9],
                "tokensOwed0": position[10],
                "tokensOwed1": position[11],
            }
        )

    @classmethod
    def approve(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        to: str,
        token_id: int,
    ) -> JSONLike:
        """Prepare encoded tx for approving a specific NFT token."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "approve",
            (to, token_id),
        )

    @classmethod
    def set_approval_for_all(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        operator: str,
        approved: bool,
    ) -> JSONLike:
        """Prepare encoded tx for setting approval for all NFTs."""
        return cls._encode_call(
            ledger_api,
            contract_address,
            "setApprovalForAll",
            (operator, approved),
        )

    @classmethod
    def is_approved_for_all(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
        operator: str,
    ) -> JSONLike:
        """Check if operator is approved for all NFTs of owner."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        is_approved = contract_instance.functions.isApprovedForAll(
            owner, operator
        ).call()
        return dict(is_approved=is_approved)

    @classmethod
    def get_approved(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get the approved address for a specific NFT token."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        approved = contract_instance.functions.getApproved(token_id).call()
        return dict(approved=approved)

    @classmethod
    def token_of_owner_by_index(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
        index: int,
    ) -> JSONLike:
        """Get token ID owned by owner at a given index."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        token_id = contract_instance.functions.tokenOfOwnerByIndex(owner, index).call()
        return dict(token_id=token_id)

    @classmethod
    def get_all_token_ids_for_owner(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
    ) -> JSONLike:
        """Get all token IDs owned by an address."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        balance = contract_instance.functions.balanceOf(owner).call()

        token_ids = []
        for i in range(balance):
            try:
                token_id = contract_instance.functions.tokenOfOwnerByIndex(
                    owner, i
                ).call()
                token_ids.append(token_id)
            except Exception:  # nosec B112 — token enumeration: skip non-existent IDs
                # Skip if token doesn't exist or error occurs
                continue

        return dict(token_ids=token_ids, count=len(token_ids))

    @classmethod
    def get_mint_tx_hashes(  # pylint: disable=too-many-arguments
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
        token_ids: List[int],
        to_block: int,
        window_blocks: int,
        min_window_blocks: int,
        lookback_blocks: int,
    ) -> JSONLike:
        """Find the tx that minted each of ``token_ids`` to ``owner``, walking back from ``to_block``.

        A window the RPC rejects is halved and retried down to ``min_window_blocks``.
        Ids whose mint is not found within ``lookback_blocks`` are left out.

        :param ledger_api: the ledger API.
        :param contract_address: the position manager address.
        :param owner: the address the NFTs were minted to.
        :param token_ids: the NFT ids to look for.
        :param to_block: the block the search starts from.
        :param window_blocks: the block range of one eth_getLogs call.
        :param min_window_blocks: the smallest range tried after rejections.
        :param lookback_blocks: how many blocks back from to_block to search.
        :return: the mint tx hash keyed by token id.
        """
        remaining = set(token_ids)
        tx_hashes: Dict[int, str] = {}
        topics = [
            TRANSFER_TOPIC,
            _to_topic(0),
            _to_topic(int(owner, 16)),
            [_to_topic(token_id) for token_id in sorted(remaining)],
        ]
        address = ledger_api.api.to_checksum_address(contract_address)
        lowest_block = max(0, to_block - lookback_blocks + 1)
        window = window_blocks
        high = to_block
        while remaining and high >= lowest_block:
            low = max(lowest_block, high - window + 1)
            try:
                logs = ledger_api.api.eth.get_logs(
                    {
                        "address": address,
                        "fromBlock": low,
                        "toBlock": high,
                        "topics": topics,
                    }
                )
            except Exception as e:  # pylint: disable=broad-except
                if window // 2 < min_window_blocks:
                    _logger.warning(
                        f"eth_getLogs failed for blocks {low}-{high} at the "
                        f"smallest window; giving up the mint search: {e}"
                    )
                    break
                window //= 2
                continue
            for log in logs:
                token_id = int.from_bytes(bytes(log["topics"][3]), "big")
                if token_id in remaining:
                    remaining.discard(token_id)
                    tx_hashes[token_id] = "0x" + bytes(log["transactionHash"]).hex()
            high = low - 1
        return dict(tx_hashes=tx_hashes)
