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
"""This class contains a wrapper for Pool contract interface."""

from typing import Any

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea_ledger_ethereum import EthereumApi


# TickMath implementation
class TickMath:
    """Implementation of Uniswap V3 TickMath library."""

    # Constants from the Uniswap V3 TickMath library
    MIN_TICK = -887272
    MAX_TICK = 887272
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

    @staticmethod
    def getSqrtRatioAtTick(tick: Any) -> Any:
        """
        Calculates sqrt(1.0001^tick) * 2^96.

        Args:
            tick: The tick for which to compute the sqrt ratio

        Returns:
            The sqrt ratio as a Q64.96 value
        """
        # Validate tick range
        if tick < TickMath.MIN_TICK or tick > TickMath.MAX_TICK:
            raise ValueError("Tick out of range")

        absTick = abs(tick)

        # Initialize ratio with Q32.128 number 0x100000000000000000000000000000000
        ratio = 0x100000000000000000000000000000000

        # Apply the bit operations from the original Solidity implementation
        if (absTick & 0x1) != 0:
            ratio = (ratio * 0xFFFCB933BD6FAD37AA2D162D1A594001) >> 128
        if (absTick & 0x2) != 0:
            ratio = (ratio * 0xFFF97272373D413259A46990580E213A) >> 128
        if (absTick & 0x4) != 0:
            ratio = (ratio * 0xFFF2E50F5F656932EF12357CF3C7FDCC) >> 128
        if (absTick & 0x8) != 0:
            ratio = (ratio * 0xFFE5CACA7E10E4E61C3624EAA0941CD0) >> 128
        if (absTick & 0x10) != 0:
            ratio = (ratio * 0xFFCB9843D60F6159C9DB58835C926644) >> 128
        if (absTick & 0x20) != 0:
            ratio = (ratio * 0xFF973B41FA98C081472E6896DFB254C0) >> 128
        if (absTick & 0x40) != 0:
            ratio = (ratio * 0xFF2EA16466C96A3843EC78B326B52861) >> 128
        if (absTick & 0x80) != 0:
            ratio = (ratio * 0xFE5DEE046A99A2A811C461F1969C3053) >> 128
        if (absTick & 0x100) != 0:
            ratio = (ratio * 0xFCBE86C7900A88AEDCFFC83B479AA3A4) >> 128
        if (absTick & 0x200) != 0:
            ratio = (ratio * 0xF987A7253AC413176F2B074CF7815E54) >> 128
        if (absTick & 0x400) != 0:
            ratio = (ratio * 0xF3392B0822B70005940C7A398E4B70F3) >> 128
        if (absTick & 0x800) != 0:
            ratio = (ratio * 0xE7159475A2C29B7443B29C7FA6E889D9) >> 128
        if (absTick & 0x1000) != 0:
            ratio = (ratio * 0xD097F3BDFD2022B8845AD8F792AA5825) >> 128
        if (absTick & 0x2000) != 0:
            ratio = (ratio * 0xA9F746462D870FDF8A65DC1F90E061E5) >> 128
        if (absTick & 0x4000) != 0:
            ratio = (ratio * 0x70D869A156D2A1B890BB3DF62BAF32F7) >> 128
        if (absTick & 0x8000) != 0:
            ratio = (ratio * 0x31BE135F97D08FD981231505542FCFA6) >> 128
        if (absTick & 0x10000) != 0:
            ratio = (ratio * 0x9AA508B5B7A84E1C677DE54F3E99BC9) >> 128
        if (absTick & 0x20000) != 0:
            ratio = (ratio * 0x5D6AF8DEDB81196699C329225EE604) >> 128
        if (absTick & 0x40000) != 0:
            ratio = (ratio * 0x2216E584F5FA1EA926041BEDFE98) >> 128
        if (absTick & 0x80000) != 0:
            ratio = (ratio * 0x48A170391F7DC42444E8FA2) >> 128

        # If tick is negative, invert the ratio
        if tick < 0:
            ratio = (2**256 - 1) // ratio

        # Ensure the ratio is within valid bounds
        if ratio > TickMath.MAX_SQRT_RATIO:
            return TickMath.MAX_SQRT_RATIO

        return ratio


# LiquidityAmounts implementation
class LiquidityAmounts:
    """Implementation of Uniswap V3 LiquidityAmounts library."""

    @staticmethod
    def getAmountsForLiquidity(
        sqrtRatioX96: Any, sqrtRatioAX96: Any, sqrtRatioBX96: Any, liquidity: Any
    ) -> Any:
        """
        Computes the token0 and token1 amounts for a given liquidity amount.

        Args:
            sqrtRatioX96: The current sqrt price as a Q64.96
            sqrtRatioAX96: The lower sqrt price as a Q64.96
            sqrtRatioBX96: The upper sqrt price as a Q64.96
            liquidity: The liquidity amount

        Returns:
            A tuple of (amount0, amount1) representing the token amounts
        """
        if sqrtRatioAX96 > sqrtRatioBX96:
            sqrtRatioAX96, sqrtRatioBX96 = sqrtRatioBX96, sqrtRatioAX96

        amount0 = 0
        amount1 = 0

        # Calculate amount0
        if sqrtRatioX96 <= sqrtRatioAX96:
            # Current price is below the range, all liquidity is in token0
            amount0 = LiquidityAmounts._getAmount0ForLiquidity(
                sqrtRatioAX96, sqrtRatioBX96, liquidity
            )
        elif sqrtRatioX96 < sqrtRatioBX96:
            # Current price is within the range
            amount0 = LiquidityAmounts._getAmount0ForLiquidity(
                sqrtRatioX96, sqrtRatioBX96, liquidity
            )

        # Calculate amount1
        if sqrtRatioX96 >= sqrtRatioBX96:
            # Current price is above the range, all liquidity is in token1
            amount1 = LiquidityAmounts._getAmount1ForLiquidity(
                sqrtRatioAX96, sqrtRatioBX96, liquidity
            )
        elif sqrtRatioX96 > sqrtRatioAX96:
            # Current price is within the range
            amount1 = LiquidityAmounts._getAmount1ForLiquidity(
                sqrtRatioAX96, sqrtRatioX96, liquidity
            )

        return amount0, amount1

    @staticmethod
    def _getAmount0ForLiquidity(
        sqrtRatioAX96: Any, sqrtRatioBX96: Any, liquidity: Any
    ) -> Any:
        """Calculate amount0 from liquidity and sqrt-price bounds.

        Formula: amount0 = liquidity * (sqrtRatioBX96 - sqrtRatioAX96)
        / (sqrtRatioAX96 * sqrtRatioBX96)
        """
        # Multiply by 2^96 first to maintain precision
        numerator = liquidity * (sqrtRatioBX96 - sqrtRatioAX96) * (2**96)
        denominator = sqrtRatioBX96 * sqrtRatioAX96

        return numerator // denominator

    @staticmethod
    def _getAmount1ForLiquidity(
        sqrtRatioAX96: Any, sqrtRatioBX96: Any, liquidity: Any
    ) -> Any:
        """Calculate amount1 from liquidity and sqrt-price bounds.

        Formula: amount1 = liquidity * (sqrtRatioBX96 - sqrtRatioAX96)
        """
        return liquidity * (sqrtRatioBX96 - sqrtRatioAX96) // (2**96)


PUBLIC_ID = PublicId.from_str("valory/uniswap_v3_pool:0.1.0")


class UniswapV3PoolContract(Contract):
    """The Uniswap V3 Pool contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_pool_tokens(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the pool tokens."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        token0 = contract_instance.functions.token0().call()
        token1 = contract_instance.functions.token1().call()
        return dict(tokens=[token0, token1])

    @classmethod
    def get_pool_fee(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the fee."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        fee = contract_instance.functions.fee().call()
        return dict(data=fee)

    @classmethod
    def get_tick_spacing(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the tick spacing."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        tick_spacing = contract_instance.functions.tickSpacing().call()
        return dict(data=tick_spacing)

    @classmethod
    def slot0(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the current state of the pool from slot0."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        result = contract_instance.functions.slot0().call()
        return dict(
            slot0={
                "sqrt_price_x96": result[0],
                "tick": result[1],
                "observation_index": result[2],
                "observation_cardinality": result[3],
                "observation_cardinality_next": result[4],
                "fee_protocol": result[5],
                "unlocked": result[6],
            }
        )

    @classmethod
    def get_reserves_and_balances(
        cls, ledger_api: EthereumApi, contract_address: str, position: Any
    ) -> JSONLike:
        """
        Get token amounts for a Uniswap V3 position.

        This implements the calculation as per Uniswap V3 documentation:
        1. Get position info from NonfungiblePositionManager
        2. Get pool information including current price
        3. Calculate amounts using LiquidityAmounts formula

        Args:
            ledger_api: The ledger API
            contract_address: The pool contract address
            your_address: The user's address
            token_id: The NFT token ID representing the position
            position_manager_address: The address of the NonfungiblePositionManager contract

        Returns:
            A dictionary containing the calculated token amounts and position details
        """

        # Extract position details
        tick_lower = position["tickLower"]
        tick_upper = position["tickUpper"]
        liquidity = position["liquidity"]
        tokens_owed0 = position["tokensOwed0"]
        tokens_owed1 = position["tokensOwed1"]

        # Step 2: Get the pool contract
        pool_instance = cls.get_instance(ledger_api, contract_address)

        # Step 3: Call slot0() to get current price and tick
        slot0_data = pool_instance.functions.slot0().call()
        sqrt_price_x96 = slot0_data[0]
        current_tick = slot0_data[1]

        # Step 4: Use TickMath to get sqrt ratios at ticks

        # Calculate sqrtRatioA and sqrtRatioB
        sqrt_ratio_a_x96 = TickMath.getSqrtRatioAtTick(tick_lower)
        sqrt_ratio_b_x96 = TickMath.getSqrtRatioAtTick(tick_upper)

        # Ensure sqrtRatioA <= sqrtRatioB
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        # Step 5: Use LiquidityAmounts to calculate token amounts

        # Calculate amounts
        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_price_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity
        )

        # Add uncollected fees
        amount0 += tokens_owed0
        amount1 += tokens_owed1

        return dict(
            data={
                "current_token0_qty": amount0,
                "current_token1_qty": amount1,
                "liquidity": liquidity,
                "tick_lower": tick_lower,
                "tick_upper": tick_upper,
                "current_tick": current_tick,
                "sqrt_price_x96": sqrt_price_x96,
                "tokens_owed0": tokens_owed0,
                "tokens_owed1": tokens_owed1,
            }
        )
