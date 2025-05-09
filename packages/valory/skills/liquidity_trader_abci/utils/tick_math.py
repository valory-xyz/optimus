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

"""Implementation of TickMath and LiquidityAmounts utilities for concentrated liquidity calculations."""

from typing import Tuple


# TickMath implementation
class TickMath:
    """Implementation of Uniswap V3 TickMath library."""

    # Constants from the Uniswap V3 TickMath library
    MIN_TICK = -887272
    MAX_TICK = 887272
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

    @staticmethod
    def getSqrtRatioAtTick(tick: int) -> int:
        """Convert a tick value to a sqrtPriceX96 value for Uniswap/Velodrome calculations."""
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
        if ratio < TickMath.MIN_SQRT_RATIO:
            return TickMath.MIN_SQRT_RATIO
        if ratio > TickMath.MAX_SQRT_RATIO:
            return TickMath.MAX_SQRT_RATIO

        return ratio


# LiquidityAmounts implementation
class LiquidityAmounts:
    """Implementation of Uniswap V3 LiquidityAmounts library."""

    @staticmethod
    def getAmountsForLiquidity(
        sqrtRatioX96: int, sqrtRatioAX96: int, sqrtRatioBX96: int, liquidity: int
    ) -> Tuple[int, int]:
        """Calculate token amounts from liquidity and price range boundaries for a position."""
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
        sqrtRatioAX96: int, sqrtRatioBX96: int, liquidity: int
    ) -> int:
        """Calculate the amount of token0 for a position given liquidity and price range."""
        # Multiply by 2^96 first to maintain precision
        numerator = liquidity * (sqrtRatioBX96 - sqrtRatioAX96) * (2**96)
        denominator = sqrtRatioBX96 * sqrtRatioAX96

        return numerator // denominator

    @staticmethod
    def _getAmount1ForLiquidity(
        sqrtRatioAX96: int, sqrtRatioBX96: int, liquidity: int
    ) -> int:
        """Calculate the amount of token1 for a position given liquidity and price range."""
        return liquidity * (sqrtRatioBX96 - sqrtRatioAX96) // (2**96)
