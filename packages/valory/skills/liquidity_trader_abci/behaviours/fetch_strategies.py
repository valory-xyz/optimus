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

"""This module contains the behaviour for fetching the strategies to execute for 'liquidity_trader_abci' skill."""

import json
from decimal import Context, Decimal, getcontext
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    Type,
    List,
    Tuple,
    Union,
)
from eth_utils import to_checksum_address
from packages.valory.contracts.balancer_vault.contract import VaultContract
from packages.valory.contracts.balancer_weighted_pool.contract import (
    WeightedPoolContract,
)
from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.sturdy_yearn_v3_vault.contract import (
    YearnV3VaultContract,
)
from packages.valory.contracts.uniswap_v3_non_fungible_position_manager.contract import (
    UniswapV3NonfungiblePositionManagerContract,
)
from packages.valory.contracts.uniswap_v3_pool.contract import UniswapV3PoolContract
from packages.valory.contracts.velodrome_cl_pool.contract import VelodromeCLPoolContract
from packages.valory.contracts.velodrome_non_fungible_position_manager.contract import (
    VelodromeNonFungiblePositionManagerContract,
)
from packages.valory.contracts.velodrome_pool.contract import VelodromePoolContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.liquidity_trader_abci.rounds import (
    FetchStrategiesPayload,
    FetchStrategiesRound,
)
from packages.valory.skills.liquidity_trader_abci.utils.tick_math import (
    LiquidityAmounts,
    TickMath,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
    PositionStatus,
    DexType,
    TradingType
)

class FetchStrategiesBehaviour(LiquidityTraderBaseBehaviour):
    """Behaviour that gets the balances of the assets of agent safes."""

    matching_round: Type[AbstractRound] = FetchStrategiesRound
    strategies = None

    def async_act(self) -> Generator:
        """Async act"""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            db_data = yield from self._read_kv(
                keys=("selected_protocols", "trading_type")
            )

            selected_protocols = db_data.get("selected_protocols", None)
            if selected_protocols is None:
                serialized_protocols = []
            else:
                serialized_protocols = json.loads(selected_protocols)

            trading_type = db_data.get("trading_type", None)

            if not serialized_protocols:
                serialized_protocols = []
                for chain in self.params.target_investment_chains:
                    chain_strategies = self.params.available_strategies.get(chain, [])
                    serialized_protocols.extend(chain_strategies)

            if not trading_type:
                trading_type = TradingType.BALANCED.value

            self.context.logger.info(
                f"Reading values from kv store... Selected protocols: {serialized_protocols}, Trading type: {trading_type}"
            )
            self.shared_state.trading_type = trading_type
            self.shared_state.selected_protocols = selected_protocols

            # Check if we need to recalculate the portfolio
            if self.should_recalculate_portfolio(self.portfolio_data):
                yield from self.calculate_user_share_values()
                # Store the updated portfolio data
                self.store_portfolio_data()

            payload = FetchStrategiesPayload(
                sender=sender,
                content=json.dumps(
                    {
                        "selected_protocols": serialized_protocols,
                        "trading_type": trading_type,
                    },
                    sort_keys=True,
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def should_recalculate_portfolio(self, last_portfolio_data: Dict) -> bool:
        """Determine if the portfolio should be recalculated."""
        return self._is_period_due() or self._have_positions_changed(
            last_portfolio_data
        )

    def _is_period_due(self) -> bool:
        """Check if the period count indicates a recalculation is due."""
        return self.synchronized_data.period_count % 10 == 0

    def _have_positions_changed(self, last_portfolio_data: Dict) -> bool:
        """Check if the positions have changed since the last calculation."""
        current_positions = self.current_positions
        last_positions = last_portfolio_data.get("allocations", [])

        if len(current_positions) != len(last_positions):
            return True

        # Create a set of current position identifiers i.e. pool_id with their status
        current_position_status = {
            (position.get("pool_address"), position.get("status"))
            for position in current_positions
        }

        # Create a set of last position identifiers i.e. pool_id with their status
        last_position_status = {
            (position.get("id"), "open") for position in last_positions
        }

        # Check if there are any differences in the sets
        if current_position_status != last_position_status:
            return True

        return False

    def calculate_user_share_values(self) -> Generator[None, None, None]:
        """Calculate the value of shares for the user based on open pools."""
        total_user_share_value_usd = Decimal(0)
        allocations = []
        individual_shares = []
        portfolio_breakdown = []

        for position in self.current_positions:
            if position.get("status") == PositionStatus.OPEN.value:
                dex_type = position.get("dex_type")
                chain = position.get("chain")
                pool_id = (
                    position.get("pool_id")
                    if dex_type == DexType.BALANCER.value
                    else position.get("pool_address")
                )
                assets = (
                    [position.get("token0_symbol"), position.get("token1_symbol")]
                    if dex_type in [DexType.BALANCER.value, DexType.UNISWAP_V3.value, DexType.VELODROME.value]
                    else [position.get("token0_symbol")]
                )
                apr = position.get("apr")

                # Calculate user share value
                user_address = self.params.safe_contract_addresses.get(chain)
                if dex_type == DexType.BALANCER.value:
                    pool_address = position.get("pool_address")
                    user_balances = yield from self.get_user_share_value_balancer(
                        user_address,
                        pool_id,
                        pool_address,
                        chain,
                    )
                    details = yield from self._get_balancer_pool_name(
                        pool_address, chain
                    )
                elif dex_type == DexType.UNISWAP_V3.value:
                    pool_address = position.get("pool_address")
                    token_id = position.get("token_id")
                    user_balances = yield from self.get_user_share_value_uniswap(
                        pool_address, token_id, chain, position
                    )
                    details = f"Uniswap V3 Pool - {position.get('token0_symbol')}/{position.get('token1_symbol')}"
                elif dex_type == DexType.STURDY.value:
                    aggregator_address = position.get("pool_address")
                    asset_address = position.get("token0")
                    user_balances = yield from self.get_user_share_value_sturdy(
                        user_address, aggregator_address, asset_address, chain
                    )
                    details = yield from self._get_aggregator_name(
                        aggregator_address, chain
                    )
                elif dex_type == DexType.VELODROME.value:
                    pool_address = position.get("pool_address")
                    token_id = position.get("token_id")
                    user_balances = yield from self.get_user_share_value_velodrome(
                        user_address, pool_address, token_id, chain, position
                    )
                    # For Velodrome pools, we'll use a simple description for now
                    details = "Velodrome " + (
                        "CL Pool" if position.get("is_cl_pool") else "Pool"
                    )

                user_share = Decimal(0)

                for asset in assets:
                    if (
                        dex_type == DexType.BALANCER.value
                        or dex_type == DexType.UNISWAP_V3.value
                        or dex_type == DexType.VELODROME.value
                    ):
                        token0_address = position.get("token0")
                        token1_address = position.get("token1")
                        asset_addresses = [token0_address, token1_address]
                        asset_address = asset_addresses[assets.index(asset)]
                    elif dex_type == DexType.STURDY.value:
                        asset_address = position.get("token0")
                    else:
                        self.context.logger.error(f"Unsupported DEX type: {dex_type}")
                        continue

                    asset_balance = user_balances.get(asset_address)
                    if asset_balance is None:
                        self.context.logger.error(
                            f"Could not find balance for asset {asset}"
                        )
                        continue

                    asset_price = yield from self._fetch_token_price(
                        asset_address, chain
                    )
                    if asset_price is None:
                        self.context.logger.error(
                            f"Could not fetch price for asset {asset}"
                        )
                        continue

                    asset_price = Decimal(str(asset_price))

                    asset_value_usd = asset_balance * asset_price
                    user_share += asset_value_usd
                    # Check if the asset already exists in the portfolio_breakdown
                    existing_asset = next(
                        (
                            entry
                            for entry in portfolio_breakdown
                            if entry["address"] == asset_address
                        ),
                        None,
                    )
                    if existing_asset:
                        # Add the balance to the existing entry
                        existing_asset["balance"] = float(asset_balance)
                        existing_asset["value_usd"] = asset_value_usd
                    else:
                        # Create a new entry for the asset
                        portfolio_breakdown.append(
                            {
                                "asset": asset,
                                "address": asset_address,
                                "balance": float(asset_balance),
                                "price": asset_price,
                                "value_usd": asset_value_usd,
                            }
                        )

                total_user_share_value_usd += user_share
                individual_shares.append(
                    (
                        user_share,
                        dex_type,
                        chain,
                        pool_id,
                        assets,
                        apr,
                        details,
                        user_address,
                        user_balances,
                    )
                )

        # Remove closed positions from allocations
        allocations = [
            allocation
            for allocation in allocations
            if allocation["id"] != pool_id
            or allocation["type"] != dex_type
            or allocation["status"] != PositionStatus.CLOSED.value
        ]

        portfolio_breakdown = [
            entry for entry in portfolio_breakdown if entry["value_usd"] > 0
        ]

        total_user_share_value_usd = sum(
            Decimal(str(entry["value_usd"])) for entry in portfolio_breakdown
        )
        if total_user_share_value_usd > 0:
            # Calculate the ratio of each asset in the portfolio
            total_ratio = sum(
                Decimal(str(entry["value_usd"])) / total_user_share_value_usd
                for entry in portfolio_breakdown
                if total_user_share_value_usd > 0
            )
            for entry in portfolio_breakdown:
                if total_user_share_value_usd > 0:
                    entry["ratio"] = round(
                        Decimal(str(entry["value_usd"]))
                        / total_user_share_value_usd
                        / total_ratio,
                        6,
                    )
                    entry["value_usd"] = float(entry["value_usd"])
                    entry["balance"] = float(entry["balance"])
                else:
                    entry["ratio"] = 0.0
                    entry["value_usd"] = float(entry["value_usd"])
                    entry["balance"] = float(entry["balance"])

            # Calculate ratios and build allocations
            total_ratio = sum(
                float(user_share / total_user_share_value_usd) * 100
                for user_share, _, _, _, _, _, _, _, _ in individual_shares
                if total_user_share_value_usd > 0
            )
            for (
                user_share,
                dex_type,
                chain,
                pool_id,
                assets,
                apr,
                details,
                user_address,
                _,
            ) in individual_shares:
                if total_user_share_value_usd > 0:
                    ratio = round(
                        float(user_share / total_user_share_value_usd)
                        * 100
                        * 100
                        / total_ratio,
                        2,
                    )
                else:
                    ratio = 0.0

                allocations.append(
                    {
                        "chain": chain,
                        "type": dex_type,
                        "id": pool_id,
                        "assets": assets,
                        "apr": round(apr, 2),
                        "details": details,
                        "ratio": float(ratio),
                        "address": user_address,
                    }
                )

        # Store the calculated portfolio value and breakdown
        self.portfolio_data = {
            "portfolio_value": float(total_user_share_value_usd),
            "allocations": [
                {
                    "chain": allocation["chain"],
                    "type": allocation["type"],
                    "id": allocation["id"],
                    "assets": allocation["assets"],
                    "apr": float(allocation["apr"]),
                    "details": allocation["details"],
                    "ratio": float(allocation["ratio"]),
                    "address": allocation["address"],
                }
                for allocation in allocations
            ],
            "portfolio_breakdown": [
                {
                    "asset": entry["asset"],
                    "address": entry["address"],
                    "balance": float(entry["balance"]),
                    "price": float(entry["price"]),
                    "value_usd": float(entry["value_usd"]),
                    "ratio": float(entry["ratio"]),
                }
                for entry in portfolio_breakdown
            ],
            "address": self.params.safe_contract_addresses.get(
                self.params.target_investment_chains[0]
            ),
        }

    def get_user_share_value_velodrome(
        self, user_address: str, pool_address: str, token_id: int, chain: str, position
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Velodrome pool."""
        token0_address = position.get("token0")
        token1_address = position.get("token1")
        is_cl_pool = position.get("is_cl_pool", False)

        if not token0_address or not token1_address:
            self.context.logger.error("Token addresses not found")
            return {}

        if is_cl_pool:
            result = yield from self._get_user_share_value_velodrome_cl(
                pool_address,
                chain,
                position,
                token0_address,
                token1_address,
            )
        else:
            result = yield from self._get_user_share_value_velodrome_non_cl(
                user_address,
                pool_address,
                chain,
                position,
                token0_address,
                token1_address,
            )
        return result

    def _get_token_decimals_pair(self, chain, token0_address, token1_address):
        token0_decimals = yield from self._get_token_decimals(chain, token0_address)
        token1_decimals = yield from self._get_token_decimals(chain, token1_address)
        if token0_decimals is None or token1_decimals is None:
            self.context.logger.error("Failed to get token decimals")
            return None, None
        return token0_decimals, token1_decimals

    def _adjust_for_decimals(self, amount, decimals):
        return Decimal(str(amount)) / Decimal(10**decimals)

    def _calculate_cl_position_value(
        self,
        pool_address: str,
        chain: str,
        position: Dict[str, Any],
        token0_address: str,
        token1_address: str,
        position_manager_address: str,
        contract_id: Any,
        get_position_callable: str = "get_position",
        position_data_key: str = "data",
        slot0_contract_id: Any = None,
    ) -> Generator[None, None, Dict[str, Decimal]]:
        """
        Calculate concentrated liquidity position value.
        
        Args:
            pool_address: Address of the pool contract
            chain: Chain identifier
            position: Position data dictionary
            token0_address: Address of token0
            token1_address: Address of token1
            position_manager_address: Address of position manager contract
            contract_id: Contract identifier
            get_position_callable: Name of the position getter function
            position_data_key: Key for position data in response
            slot0_contract_id: Optional contract ID for slot0 calls
        
        Returns:
            Dictionary mapping token addresses to their quantities
        """
        # Early validation of required parameters
        if not all([pool_address, chain, position, token0_address, token1_address, position_manager_address]):
            self.context.logger.error("Missing required parameters")
            return {}

        # Get slot0 data in a single call
        slot0_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=slot0_contract_id or contract_id,
            contract_callable="slot0",
            data_key="slot0",
            chain_id=chain,
        )
        
        # Early return if slot0 data is invalid
        if not slot0_data:
            self.context.logger.error(f"Failed to get slot0 data for pool {pool_address}")
            return {}

        # Extract and validate slot0 data
        sqrt_price_x96 = slot0_data.get("sqrt_price_x96")
        current_tick = slot0_data.get("tick")
        if not sqrt_price_x96 or current_tick is None:
            self.context.logger.error(f"Invalid slot0 data: {slot0_data}")
            return {}

        # Get token decimals in parallel using a list comprehension
        token_decimals = yield from self._get_token_decimals_pair(chain, token0_address, token1_address)
        if None in token_decimals:
            return {}
        token0_decimals, token1_decimals = token_decimals

        # Initialize quantities
        total_token0_qty = Decimal(0)
        total_token1_qty = Decimal(0)

        # Handle position(s)
        positions_to_process = (
            position.get("positions", []) if isinstance(position.get("positions", []), list)
            else [position]
        )

        # Process all positions
        for pos in positions_to_process:
            token_id = pos.get("token_id")
            if not token_id:
                continue

            position_details = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=position_manager_address,
                contract_public_id=contract_id,
                contract_callable=get_position_callable,
                data_key=position_data_key,
                token_id=token_id,
                chain_id=chain,
            )
            
            if not position_details:
                self.context.logger.error(f"Failed to get position details for token ID {token_id}")
                continue

            # Calculate amounts for this position
            amount0, amount1 = yield from self._calculate_position_amounts(
                position_details, current_tick, sqrt_price_x96, pos
            )
            
            total_token0_qty += Decimal(amount0)
            total_token1_qty += Decimal(amount1)

        # Calculate final adjusted quantities
        result = {
            token0_address: self._adjust_for_decimals(total_token0_qty, token0_decimals),
            token1_address: self._adjust_for_decimals(total_token1_qty, token1_decimals),
        }

        # Log results
        self.context.logger.info(
            f"CL Pool Total Position Balances - "
            f"Token0: {result[token0_address]} {position.get('token0_symbol')}, "
            f"Token1: {result[token1_address]} {position.get('token1_symbol')}"
        )
        
        return result
    
    def _calculate_position_amounts(
        self, 
        position_details: Dict[str, Any], 
        current_tick: int, 
        sqrt_price_x96: int, 
        position: Dict[str, Any],
    ) -> Generator[None, None, Tuple[int, int]]:
        """
        Calculate token amounts for a position based on whether it's in range or not.
        
        Args:
            position_details: Position details from the contract
            current_tick: Current tick from the pool
            sqrt_price_x96: Current sqrt price from the pool
            position: Position data from our system
            
        Returns:
            Tuple of (amount0, amount1)
        """
        # Extract position details
        tick_lower = int(position_details.get("tickLower"))
        tick_upper = int(position_details.get("tickUpper"))
        liquidity = int(position_details.get("liquidity"))
        tokens_owed0 = int(position_details.get("tokensOwed0", 0))
        tokens_owed1 = int(position_details.get("tokensOwed1", 0))
        
        # Log position details
        self.context.logger.info(
            f"For position, liquidity range is [{tick_lower}, {tick_upper}] "
            f"and current tick is {current_tick}"
        )
        
        # Check if current tick is within the provided tick range
        if tick_lower <= current_tick <= tick_upper:
            # In range, use getAmountsForLiquidity
            sqrtA = TickMath.getSqrtRatioAtTick(tick_lower)
            sqrtB = TickMath.getSqrtRatioAtTick(tick_upper)
            
            # Calculate amounts using getAmountsForLiquidity
            amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
                sqrt_price_x96,
                sqrtA,
                sqrtB,
                liquidity
            )
            
            self.context.logger.info(
                f"Position is in range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper})"
            )
        else:
            # Out of range, return invested amounts + tokensOwed
            amount0 = position.get("amount0", 0)
            amount1 = position.get("amount1", 0)
            
            # Add uncollected fees (tokensOwed)
            amount0 += tokens_owed0
            amount1 += tokens_owed1
            
            self.context.logger.info(
                f"Position is out of range. Current tick: {current_tick}, "
                f"Range: [{tick_lower}, {tick_upper})"
            )
            
        return amount0, amount1

    def _get_user_share_value_velodrome_cl(
        self,
        pool_address,
        chain,
        position,
        token0_address,
        token1_address,
    ):
        """Calculate the user's share value and token balances in a Velodrome CL pool."""
        position_manager_address = (
            self.params.velodrome_non_fungible_position_manager_contract_addresses.get(
                chain
            )
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return {}
            
        return (yield from self._calculate_cl_position_value(
            pool_address=pool_address,
            chain=chain,
            position=position,
            token0_address=token0_address,
            token1_address=token1_address,
            position_manager_address=position_manager_address,
            contract_id=VelodromeNonFungiblePositionManagerContract.contract_id,
            get_position_callable="get_position",
            position_data_key="data",
            slot0_contract_id=VelodromeCLPoolContract.contract_id,
        ))

    def _get_user_share_value_velodrome_non_cl(
        self,
        user_address,
        pool_address,
        chain,
        position,
        token0_address,
        token1_address,
    ):
        """Calculate the user's share value and token balances in a Velodrome non-CL pool."""
        user_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="check_balance",
            data_key="token",
            account=user_address,
            chain_id=chain,
        )
        if user_balance is None:
            self.context.logger.error(
                f"Failed to get user balance for pool: {pool_address}"
            )
            return {}

        total_supply = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_total_supply",
            data_key="data",
            chain_id=chain,
        )
        if not total_supply:
            self.context.logger.error(
                f"Failed to get total supply for pool: {pool_address}"
            )
            return {}

        reserves = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromePoolContract.contract_id,
            contract_callable="get_reserves",
            data_key="data",
            chain_id=chain,
        )
        if not reserves:
            self.context.logger.error(
                f"Failed to get reserves for pool: {pool_address}"
            )
            return {}

        getcontext().prec = 50
        user_share = Decimal(str(user_balance)) / Decimal(str(total_supply))
        token0_decimals, token1_decimals = yield from self._get_token_decimals_pair(
            chain, token0_address, token1_address
        )
        if token0_decimals is None or token1_decimals is None:
            return {}

        token0_balance = self._adjust_for_decimals(reserves[0], token0_decimals)
        token1_balance = self._adjust_for_decimals(reserves[1], token1_decimals)
        user_token0_balance = user_share * token0_balance
        user_token1_balance = user_share * token1_balance

        self.context.logger.info(
            f"Velodrome Non-CL Pool Balances - "
            f"User share: {user_share}, "
            f"Token0: {user_token0_balance} {position.get('token0_symbol')}, "
            f"Token1: {user_token1_balance} {position.get('token1_symbol')}"
        )
        return {
            token0_address: user_token0_balance,
            token1_address: user_token1_balance,
        }

    def get_user_share_value_uniswap(
        self, pool_address: str, token_id: int, chain: str, position
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Uniswap V3 position."""
        token0_address = position.get("token0")
        token1_address = position.get("token1")

        if not token0_address or not token1_address or not token_id:
            self.context.logger.error("Token addresses or token_id not found")
            return {}

        # Get the position manager address for the chain
        position_manager_address = (
            self.params.uniswap_position_manager_contract_addresses.get(chain)
        )
        if not position_manager_address:
            self.context.logger.error(
                f"No position manager address found for chain {chain}"
            )
            return {}
            
        # Use the common helper function for CL position calculation
        return (yield from self._calculate_cl_position_value(
            pool_address=pool_address,
            chain=chain,
            position=position,
            token0_address=token0_address,
            token1_address=token1_address,
            position_manager_address=position_manager_address,
            contract_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            get_position_callable="get_position",
            position_data_key="data",
            slot0_contract_id=UniswapV3PoolContract.contract_id,
        ))

    def get_user_share_value_balancer(
        self, user_address: str, pool_id: str, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Decimal]]]:
        """Calculate the user's share value and token balances in a Balancer pool using direct contract calls."""

        # Step 1: Get the pool tokens and balances from the Vault contract
        vault_address = self.params.balancer_vault_contract_addresses.get(chain)
        if not vault_address:
            self.context.logger.error(f"Vault address not found for chain: {chain}")
            return {}

        pool_tokens_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=vault_address,
            contract_public_id=VaultContract.contract_id,
            contract_callable="get_pool_tokens",
            data_key="tokens",
            pool_id=pool_id,
            chain_id=chain,
        )

        if not pool_tokens_data:
            self.context.logger.error(
                f"Failed to get pool tokens for pool ID: {pool_id}"
            )
            return {}

        tokens = pool_tokens_data[0]  # Array of token addresses
        balances = pool_tokens_data[1]  # Array of token balances

        # Step 2: Get the user's balance of pool tokens (BPT)
        user_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="check_balance",
            data_key="token",
            account=user_address,
            chain_id=chain,
        )

        if user_balance is None:
            self.context.logger.error(
                f"Failed to get user balance for pool: {pool_address}"
            )
            return {}

        # Step 3: Get the total supply of pool tokens
        total_supply = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=ERC20.contract_id,
            contract_callable="get_total_supply",
            data_key="data",
            chain_id=chain,
        )

        if total_supply is None or total_supply == 0:
            self.context.logger.error(
                f"Failed to get total supply for pool: {pool_address}"
            )
            return {}

        # Step 4: Calculate the user's share of the pool
        getcontext().prec = 50  # Increase decimal precision
        ctx = Context(prec=50)  # Use higher-precision data type

        user_balance_decimal = Decimal(str(user_balance))
        total_supply_decimal = Decimal(str(total_supply))

        if total_supply_decimal == 0:
            self.context.logger.error(f"Total supply is zero for pool: {pool_address}")
            return {}

        user_share = ctx.divide(user_balance_decimal, total_supply_decimal)

        # Step 5: Calculate user's token balances
        user_token_balances = {}
        for i, token_address in enumerate(tokens):
            token_address = to_checksum_address(token_address)
            token_balance = Decimal(str(balances[i]))

            # Get token decimals
            token_decimals = yield from self._get_token_decimals(chain, token_address)
            if token_decimals is None:
                self.context.logger.error(
                    f"Failed to get decimals for token: {token_address}"
                )
                continue

            # Adjust token balance based on decimals
            adjusted_token_balance = token_balance / Decimal(10**token_decimals)

            # Calculate user's token balance
            user_token_balance = user_share * adjusted_token_balance
            user_token_balances[token_address] = user_token_balance

        return user_token_balances

    def get_user_share_value_sturdy(
        self, user_address: str, aggregator_address: str, asset_address: str, chain: str
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Calculate the user's share value and token balance in a Sturdy vault."""
        # Get user's underlying asset balance in the vault
        user_asset_balance = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="balance_of",
            data_key="amount",
            owner=user_address,
            chain_id=chain,
        )
        if user_asset_balance is None:
            self.context.logger.error("Failed to get user's asset balance.")
            return {}

        user_asset_balance = Decimal(user_asset_balance)

        # Get decimals for proper scaling
        decimals = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="decimals",
            data_key="decimals",
            chain_id=chain,
        )
        if decimals is None:
            self.context.logger.error("Failed to get decimals.")
            return {}

        scaling_factor = Decimal(10 ** int(decimals))

        # Adjust decimals for assets
        user_asset_balance /= scaling_factor

        return {asset_address: user_asset_balance}

    def _get_aggregator_name(
        self, aggregator_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get the name of the Sturdy Aggregator."""
        aggreator_name = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=aggregator_address,
            contract_public_id=YearnV3VaultContract.contract_id,
            contract_callable="name",
            data_key="name",
            chain_id=chain,
        )
        return aggreator_name

    def _get_balancer_pool_name(
        self, pool_address: str, chain: str
    ) -> Generator[None, None, Optional[str]]:
        """Get the name of the Balancer Pool."""
        pool_name = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=WeightedPoolContract.contract_id,
            contract_callable="get_name",
            data_key="name",
            chain_id=chain,
        )
        return pool_name
