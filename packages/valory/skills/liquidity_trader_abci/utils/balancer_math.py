"""Balancer Math"""
from decimal import Decimal, getcontext
from typing import List, Union


getcontext().prec = 50


class BalancerProportionalMath:
    """Simplified Balancer math for proportional operations only"""

    @staticmethod
    def query_proportional_exit(
        token_balances: List[Union[int, str]],
        total_bpt_supply: Union[int, str],
        bpt_amount_in: Union[int, str],
    ) -> List[int]:
        """Simulate proportional exit (EXACT_BPT_IN_FOR_TOKENS_OUT).Works for all pool types since proportional math is universal"""
        # Convert to Decimal for precise calculations
        token_balances_decimal = [Decimal(str(balance)) for balance in token_balances]
        total_supply_decimal = Decimal(str(total_bpt_supply))
        bpt_amount_decimal = Decimal(str(bpt_amount_in))

        # Validation
        if total_supply_decimal <= 0:
            raise ValueError("Total BPT supply must be positive")
        if bpt_amount_decimal <= 0:
            raise ValueError("BPT amount must be positive")
        if bpt_amount_decimal > total_supply_decimal:
            raise ValueError("Cannot burn more BPT than total supply")

        # Calculate proportional exit ratio
        exit_ratio = bpt_amount_decimal / total_supply_decimal

        # Calculate amounts out for each token
        amounts_out = [int(balance * exit_ratio) for balance in token_balances_decimal]

        return amounts_out

    @staticmethod
    def query_proportional_join(
        token_balances: List[Union[int, str]],
        total_bpt_supply: Union[int, str],
        amounts_in: List[Union[int, str]],
    ) -> int:
        """Simulate proportional join (EXACT_TOKENS_IN_FOR_BPT_OUT). Works for all pool types since proportional math is universal"""
        # Convert to Decimal for precise calculations
        token_balances_decimal = [Decimal(str(balance)) for balance in token_balances]
        total_supply_decimal = Decimal(str(total_bpt_supply))
        amounts_in_decimal = [Decimal(str(amount)) for amount in amounts_in]

        # Validation
        if len(amounts_in_decimal) != len(token_balances_decimal):
            raise ValueError("Number of amounts_in must match number of tokens")
        if total_supply_decimal <= 0:
            raise ValueError("Total BPT supply must be positive")

        # Calculate join ratios for each token
        join_ratios = []
        for i, (balance, amount_in) in enumerate(
            zip(token_balances_decimal, amounts_in_decimal)
        ):
            if amount_in > 0 and balance > 0:
                ratio = amount_in / balance
                join_ratios.append(ratio)
            elif amount_in == 0:
                join_ratios.append(Decimal("0"))
            else:
                raise ValueError(f"Invalid amount for token {i}: {amount_in}")

        # Use minimum ratio for proportional join (limiting factor)
        if not join_ratios or all(ratio == 0 for ratio in join_ratios):
            raise ValueError("At least one token amount must be positive")

        min_ratio = min(ratio for ratio in join_ratios if ratio > 0)

        # Calculate BPT out based on minimum ratio
        bpt_amount_out = int(min_ratio * total_supply_decimal)

        return bpt_amount_out
