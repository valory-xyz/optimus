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

"""This package contains round behaviours of LiquidityTraderAbciApp."""

import json
from decimal import Context, Decimal, getcontext
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    Type,
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
                    if dex_type == DexType.BALANCER.value
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
                        user_address, pool_address, token_id, chain, position
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
                user_address,
                pool_address,
                token_id,
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

    def _get_user_share_value_velodrome_cl(
        self,
        user_address,
        pool_address,
        token_id,
        chain,
        position,
        token0_address,
        token1_address,
    ):
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

        slot0_data = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=VelodromeCLPoolContract.contract_id,
            contract_callable="slot0",
            data_key="slot0",
            chain_id=chain,
        )
        if not slot0_data:
            self.context.logger.error(
                f"Failed to get slot0 data for pool {pool_address}"
            )
            return {}

        sqrt_price_x96 = slot0_data.get("sqrt_price_x96")
        current_tick = slot0_data.get("tick")
        self.context.logger.info(
            f"Fetched pool data once - Current Tick: {current_tick}, Sqrt Price X96: {sqrt_price_x96}"
        )

        # Handle multiple or single positions
        positions_data = (
            position.get("positions", [])
            if (
                isinstance(token_id, (list, tuple))
                or (
                    position.get("positions") and len(position.get("positions", [])) > 1
                )
            )
            else [dict(token_id=token_id)]
        )

        total_token0_qty = Decimal(0)
        total_token1_qty = Decimal(0)

        for pos in positions_data:
            pos_token_id = pos.get("token_id")
            position_details = yield from self.contract_interact(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                contract_address=position_manager_address,
                contract_public_id=VelodromeNonFungiblePositionManagerContract.contract_id,
                contract_callable="get_position",
                data_key="data",
                token_id=pos_token_id,
                chain_id=chain,
            )
            if not position_details:
                self.context.logger.error(
                    f"Failed to get position details for token ID {pos_token_id}"
                )
                continue
            tick_lower = int(position_details.get("tickLower"))
            tick_upper = int(position_details.get("tickUpper"))
            liquidity = int(position_details.get("liquidity"))
            tokens_owed0 = int(position_details.get("tokensOwed0", 0))
            tokens_owed1 = int(position_details.get("tokensOwed1", 0))

            if current_tick < tick_lower:
                # All in token0
                sqrtA = TickMath.getSqrtRatioAtTick(tick_lower)
                sqrtB = TickMath.getSqrtRatioAtTick(tick_upper)
                amount0 = LiquidityAmounts._getAmount0ForLiquidity(
                    sqrtA, sqrtB, liquidity
                )
                amount1 = 0
            elif current_tick >= tick_upper:
                sqrtA = TickMath.getSqrtRatioAtTick(tick_lower)
                sqrtB = TickMath.getSqrtRatioAtTick(tick_upper)
                amount0 = 0
                amount1 = LiquidityAmounts._getAmount1ForLiquidity(
                    sqrtA, sqrtB, liquidity
                )
            else:
                # In range, use band math
                reserves_and_balances = self.get_reserves_and_balances(
                    position=position_details,
                    sqrt_price_x96=sqrt_price_x96,
                    current_tick=current_tick,
                    tick_lower=tick_lower,
                    tick_upper=tick_upper,
                    liquidity=liquidity,
                    tokens_owed0=tokens_owed0,
                    tokens_owed1=tokens_owed1,
                )
                amount0 = reserves_and_balances.get("current_token0_qty", 0)
                amount1 = reserves_and_balances.get("current_token1_qty", 0)

            if amount0 < 1e-8 and amount1 < 1e-8:
                # Value is negligible due to narrow band
                self.context.logger.warning(
                    "User position is in a very narrow band and out of range. "
                    "Current claimable value is extremely small. "
                    "If the price moves into the tick range, the value will be claimable."
                )
                # Optionally, show initial deposit as fallback
                amount0 = pos.get("amount0", 0)
                amount1 = pos.get("amount1", 0)

            total_token0_qty += Decimal(amount0)
            total_token1_qty += Decimal(amount1)

        token0_decimals, token1_decimals = yield from self._get_token_decimals_pair(
            chain, token0_address, token1_address
        )
        if token0_decimals is None or token1_decimals is None:
            return {}

        adjusted_token0_qty = self._adjust_for_decimals(
            total_token0_qty, token0_decimals
        )
        adjusted_token1_qty = self._adjust_for_decimals(
            total_token1_qty, token1_decimals
        )

        self.context.logger.info(
            f"Velodrome CL Pool Total Position Balances - "
            f"Token0: {adjusted_token0_qty} {position.get('token0_symbol')}, "
            f"Token1: {adjusted_token1_qty} {position.get('token1_symbol')}"
        )
        return {
            token0_address: adjusted_token0_qty,
            token1_address: adjusted_token1_qty,
        }

    def _get_user_share_value_velodrome_non_cl(
        self,
        user_address,
        pool_address,
        chain,
        position,
        token0_address,
        token1_address,
    ):
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

    def get_reserves_and_balances(
        self,
        position,
        sqrt_price_x96=None,
        current_tick=None,
        tick_lower=None,
        tick_upper=None,
        liquidity=None,
        tokens_owed0=0,
        tokens_owed1=0,
    ) -> Dict:
        """Calculate token amounts for a concentrated liquidity position based on Uniswap V3 methodology."""
        # Extract position details if passed via position object
        if position is not None:
            tick_lower = position.get("tickLower", tick_lower)
            tick_upper = position.get("tickUpper", tick_upper)
            liquidity = position.get("liquidity", liquidity)
            tokens_owed0 = position.get("tokensOwed0", tokens_owed0)
            tokens_owed1 = position.get("tokensOwed1", tokens_owed1)

        # Calculate sqrtRatioA and sqrtRatioB
        sqrt_ratio_a_x96 = TickMath.getSqrtRatioAtTick(tick_lower)
        sqrt_ratio_b_x96 = TickMath.getSqrtRatioAtTick(tick_upper)

        # Ensure sqrtRatioA <= sqrtRatioB
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        # Calculate amounts
        amount0, amount1 = LiquidityAmounts.getAmountsForLiquidity(
            sqrt_price_x96 or TickMath.getSqrtRatioAtTick(current_tick),
            sqrt_ratio_a_x96,
            sqrt_ratio_b_x96,
            liquidity,
        )

        # Add uncollected fees
        amount0 += tokens_owed0
        amount1 += tokens_owed1

        return {
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

    def get_user_share_value_uniswap(
        self, user_address: str, pool_address: str, token_id: int, chain: str, position
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

        # First get position details directly from the position manager
        position_details = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=position_manager_address,
            contract_public_id=UniswapV3NonfungiblePositionManagerContract.contract_id,
            contract_callable="get_position_details",
            data_key="data",
            token_id=token_id,
            chain_id=chain,
        )

        if not position_details:
            self.context.logger.error(
                f"Failed to get position details for token ID {token_id}"
            )
            return {}

        # Get reserves and balances using the position details
        reserves_and_balances = yield from self.contract_interact(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=pool_address,
            contract_public_id=UniswapV3PoolContract.contract_id,
            contract_callable="get_reserves_and_balances",
            data_key="data",
            position=position_details,
            chain_id=chain,
        )

        if not reserves_and_balances:
            self.context.logger.error(
                f"Failed to get reserves and balances for pool {pool_address} with token_id {token_id}"
            )
            return {}

        # Get token decimals
        token0_decimals = yield from self._get_token_decimals(chain, token0_address)
        token1_decimals = yield from self._get_token_decimals(chain, token1_address)

        if token0_decimals is None or token1_decimals is None:
            self.context.logger.error("Failed to get token decimals")
            return {}

        # Get the current amounts in the position
        current_token0_qty = Decimal(
            str(reserves_and_balances.get("current_token0_qty", 0))
        )
        current_token1_qty = Decimal(
            str(reserves_and_balances.get("current_token1_qty", 0))
        )

        # Log position details from the position manager
        self.context.logger.info(
            f"Uniswap V3 Position Details from Manager - Token ID: {token_id}, "
            f"Token0: {position_details.get('token0')}, "
            f"Token1: {position_details.get('token1')}, "
            f"Fee: {position_details.get('fee')}, "
            f"Tick Lower: {position_details.get('tickLower')}, "
            f"Tick Upper: {position_details.get('tickUpper')}, "
            f"Liquidity: {position_details.get('liquidity')}"
        )

        # Log additional information from the pool calculation
        self.context.logger.info(
            f"Uniswap V3 Position Details from Pool - "
            f"Current Tick: {reserves_and_balances.get('current_tick')}, "
            f"Uncollected Fees Token0: {reserves_and_balances.get('tokens_owed0')}, "
            f"Uncollected Fees Token1: {reserves_and_balances.get('tokens_owed1')}"
        )

        # Adjust for decimals
        adjusted_token0_qty = current_token0_qty / Decimal(10**token0_decimals)
        adjusted_token1_qty = current_token1_qty / Decimal(10**token1_decimals)

        self.context.logger.info(
            f"Uniswap V3 Position Balances - "
            f"Token0: {adjusted_token0_qty} {position.get('token0_symbol')}, "
            f"Token1: {adjusted_token1_qty} {position.get('token1_symbol')}"
        )

        # Return the balances
        return {
            token0_address: adjusted_token0_qty,
            token1_address: adjusted_token1_qty,
        }
