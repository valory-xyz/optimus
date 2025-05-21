# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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
    def getSqrtRatioAtTick(tick):
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
            ratio = (ratio * 0xfffcb933bd6fad37aa2d162d1a594001) >> 128
        if (absTick & 0x2) != 0:
            ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128
        if (absTick & 0x4) != 0:
            ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128
        if (absTick & 0x8) != 0:
            ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128
        if (absTick & 0x10) != 0:
            ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128
        if (absTick & 0x20) != 0:
            ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128
        if (absTick & 0x40) != 0:
            ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128
        if (absTick & 0x80) != 0:
            ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128
        if (absTick & 0x100) != 0:
            ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128
        if (absTick & 0x200) != 0:
            ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128
        if (absTick & 0x400) != 0:
            ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128
        if (absTick & 0x800) != 0:
            ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128
        if (absTick & 0x1000) != 0:
            ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128
        if (absTick & 0x2000) != 0:
            ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128
        if (absTick & 0x4000) != 0:
            ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128
        if (absTick & 0x8000) != 0:
            ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128
        if (absTick & 0x10000) != 0:
            ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128
        if (absTick & 0x20000) != 0:
            ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128
        if (absTick & 0x40000) != 0:
            ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128
        if (absTick & 0x80000) != 0:
            ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128
        
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
    def getAmountsForLiquidity(sqrtRatioX96, sqrtRatioAX96, sqrtRatioBX96, liquidity):
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
    def _getAmount0ForLiquidity(sqrtRatioAX96, sqrtRatioBX96, liquidity):
        """
        Calculates amount0 based on the formula:
        amount0 = liquidity * (sqrtRatioBX96 - sqrtRatioAX96) / (sqrtRatioAX96 * sqrtRatioBX96)
        """
        # Multiply by 2^96 first to maintain precision
        numerator = liquidity * (sqrtRatioBX96 - sqrtRatioAX96) * (2**96)
        denominator = sqrtRatioBX96 * sqrtRatioAX96
        
        return numerator // denominator
    
    @staticmethod
    def _getAmount1ForLiquidity(sqrtRatioAX96, sqrtRatioBX96, liquidity):
        """
        Calculates amount1 based on the formula:
        amount1 = liquidity * (sqrtRatioBX96 - sqrtRatioAX96)
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
        """get the pool tokens."""
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
        """get the fee."""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        fee = contract_instance.functions.fee().call()
        return dict(data=fee)

    @classmethod
    def get_tick_spacing(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """get the tick spacing."""
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
        return dict(slot0={
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
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        position
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
        token0 = position["token0"]
        token1 = position["token1"]
        fee = position["fee"]
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
            sqrt_price_x96, 
            sqrt_ratio_a_x96, 
            sqrt_ratio_b_x96, 
            liquidity
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
