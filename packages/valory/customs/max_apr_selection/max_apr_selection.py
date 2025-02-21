import math
from typing import Any, Dict, List, Union

# Constants
REQUIRED_FIELDS = ("trading_opportunities", "current_positions", "max_pools", "check_sharpe_ratio")
SHARPE_RATIO_THRESHOLD = 1
DEPTH_SCORE_THRESHOLD = 50
IL_RISK_SCORE_THRESHOLD = -0.2

# Weights for each metric
SHARPE_RATIO_WEIGHT = 0.4
DEPTH_SCORE_WEIGHT = 0.3
IL_RISK_SCORE_WEIGHT = 0.3
MIN_COMPOSITE_SCORE_RATIO = 0.5

logs = []

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    return missing


def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}


def calculate_composite_score(pool, max_values, check_sharpe_ratio):
    """Calculate the composite score for a given pool."""
    sharpe_ratio = pool.get("sharpe_ratio", math.nan)
    depth_score = pool.get("depth_score", math.nan)
    il_risk_score = pool.get("il_risk_score", math.nan)

    if math.isnan(sharpe_ratio):
        return 0

    if check_sharpe_ratio:
        return sharpe_ratio / max_values["sharpe_ratio"]
    else:
        if math.isnan(depth_score) or math.isnan(il_risk_score):
            return 0

        # Normalize metrics
        normalized_sharpe_ratio = sharpe_ratio / max_values["sharpe_ratio"]
        normalized_depth_score = depth_score / max_values["depth_score"]
        normalized_il_risk_score = (abs(il_risk_score)) / abs(max_values["il_risk_score"])

        # Calculate composite score
        return (
            SHARPE_RATIO_WEIGHT * normalized_sharpe_ratio
            + DEPTH_SCORE_WEIGHT * normalized_depth_score
            + IL_RISK_SCORE_WEIGHT * normalized_il_risk_score
        )


def get_max_values(pools):
    """Get maximum values for normalization."""
    return {
        "sharpe_ratio": max((pool.get("sharpe_ratio", 0) for pool in pools)),
        "depth_score": max((pool.get("depth_score", 0) for pool in pools)),
        "il_risk_score": max((pool.get("il_risk_score", 0) for pool in pools)),
    }


def calculate_relative_percentages(percentages):
    """
    Calculate the relative percentages of a list of percentages.
    This function takes a list of percentages and calculates the relative
    percentage of each element with respect to the running total of the
    percentages encountered so far.
    Args:
        percentages (list of float): A list of percentages.
    Returns:
        list of float: A list of relative percentages.
    """
    total_percentage = sum(percentages)
    dynamic_percentages = []

    if total_percentage == 0:
        logs.append("ERROR: Total percentage cannot be zero.")
        return []

    for percentage in percentages:
        dynamic_percentage = percentage / total_percentage
        dynamic_percentages.append(dynamic_percentage)
        total_percentage -= percentage

    return dynamic_percentages

def get_max_values(pools):
    """Get maximum values for normalization."""
    return {
        "sharpe_ratio": max((pool.get("sharpe_ratio", 0) for pool in pools)),
        "depth_score": max((pool.get("depth_score", 0) for pool in pools)),
        "il_risk_score": max((pool.get("il_risk_score", 0) for pool in pools)),
    }

def apply_risk_thresholds_and_select_optimal_strategy(
    trading_opportunities,
    current_positions=None,
    improvement_threshold=0.1,
    max_pools=1,
    check_sharpe_ratio=False,
):
    """Apply risk thresholds and select the optimal strategy based on combined metrics."""
    reasoning = []
    base_description = "The agent analyzes opportunities using a balanced score of risk-adjusted returns (Sharpe ratio), market liquidity (depth score), and impermanent loss risk. A better overall score might not mean better values in all metrics. "

    # Filter opportunities based on risk thresholds
    filtered_opportunities = []
    for opportunity in trading_opportunities:
        sharpe_ratio = opportunity.get("sharpe_ratio", 0)
        depth_score = opportunity.get("depth_score", 0)
        il_risk_score = opportunity.get("il_risk_score", float("inf"))

        logs.append(f"Evaluating opportunity: {opportunity}")
        if (
            not isinstance(sharpe_ratio, (int, float))
            or not isinstance(depth_score, (int, float))
            or not isinstance(il_risk_score, (int, float))
        ):
            logs.append("WARNING: Invalid values for risk metrics")
            continue

        if check_sharpe_ratio:
            if sharpe_ratio <= SHARPE_RATIO_THRESHOLD:
                logs.append(f"Opportunity does not meet the {SHARPE_RATIO_THRESHOLD=}")
                continue
        else:
            if sharpe_ratio <= SHARPE_RATIO_THRESHOLD:
                logs.append(f"Opportunity does not meet the {SHARPE_RATIO_THRESHOLD=}")
                continue

            if depth_score <= DEPTH_SCORE_THRESHOLD:
                logs.append(f"Opportunity does not meet the {DEPTH_SCORE_THRESHOLD=}")
                continue

            if il_risk_score <= IL_RISK_SCORE_THRESHOLD:
                logs.append(f"Opportunity does not meet the {IL_RISK_SCORE_THRESHOLD=}")
                continue

        logs.append("Opportunity meets all risk thresholds")
        filtered_opportunities.append(opportunity)

    if not filtered_opportunities:
        logs.append("No opportunities meet the risk thresholds.")
        reasoning.append("No opportunities currently meet our minimum requirements for safe and profitable trading.")
        return {
            "optimal_strategies": [], 
            "position_to_exit": {}, 
            "reasoning": base_description + " | ".join(reasoning)
        }

    # Calculate max values for normalization
    max_values = get_max_values(filtered_opportunities)

    # Calculate composite scores for filtered opportunities
    for opportunity in filtered_opportunities:
        opportunity["composite_score"] = calculate_composite_score(
            opportunity, max_values, check_sharpe_ratio
        )

    position_to_exit = {}
    optimal_opportunities = []

    if current_positions:
        # Calculate composite score for the current pool
        current_composite_scores = [
            calculate_composite_score(pool, max_values, check_sharpe_ratio) for pool in current_positions
        ]

        # Identify the least performing current pool
        least_performing_index = current_composite_scores.index(min(current_composite_scores))
        least_performing_score = current_composite_scores[least_performing_index]
        position_to_exit = current_positions[least_performing_index]

        # Compare each opportunity with the least performing current pool
        better_opportunities = [
            opportunity
            for opportunity in filtered_opportunities
            if opportunity["composite_score"]
            > least_performing_score * (1 + improvement_threshold)
        ]

        if better_opportunities:
            better_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
            optimal_opportunities = [better_opportunities[0]]
            optimal_opportunities[0]["relative_funds_percentage"] = 1.0
            
            # Add user-friendly reasoning with metric comparisons
            if position_to_exit:
                exit_pool = position_to_exit
                reasoning.append(
                    f"Currently invested in {exit_pool.get('dex_type', 'unknown')} pool {exit_pool.get('pool_address', 'unknown')} "
                    f"with metrics: risk-adjusted returns {exit_pool.get('sharpe_ratio', 0):.2f}, "
                    f"market liquidity {exit_pool.get('depth_score', 0):.2f}, "
                    f"impermanent loss risk {exit_pool.get('il_risk_score', 0):.2f}"
                )
            
            top_opp = optimal_opportunities[0]
            
            # Calculate metric changes and prepare comparison text
            metrics_comparison = []
            current_sharpe = position_to_exit.get('sharpe_ratio', 0)
            current_depth = position_to_exit.get('depth_score', 0)
            current_il = position_to_exit.get('il_risk_score', 0)
            
            new_sharpe = top_opp.get('sharpe_ratio', 0)
            new_depth = top_opp.get('depth_score', 0)
            new_il = top_opp.get('il_risk_score', 0)

            if new_sharpe > current_sharpe:
                metrics_comparison.append(f"better risk-adjusted returns ({new_sharpe:.2f} vs {current_sharpe:.2f})")
            else:
                metrics_comparison.append(f"slightly lower risk-adjusted returns ({new_sharpe:.2f} vs {current_sharpe:.2f})")

            if new_depth > current_depth:
                metrics_comparison.append(f"improved market liquidity ({new_depth:.2f} vs {current_depth:.2f})")
            else:
                metrics_comparison.append(f"slightly lower market liquidity ({new_depth:.2f} vs {current_depth:.2f})")

            if new_il > current_il:
                metrics_comparison.append(f"better impermanent loss protection ({new_il:.2f} vs {current_il:.2f})")
            else:
                metrics_comparison.append(f"slightly higher impermanent loss risk ({new_il:.2f} vs {current_il:.2f})")
            
            composite_improvement = ((top_opp['composite_score'] - least_performing_score) / least_performing_score) * 100

            reasoning.append(
                f"Found opportunity in {top_opp.get('dex_type', 'unknown')} pool {top_opp.get('pool_address', 'unknown')} "
                f"with {composite_improvement:.1f}% better overall performance and expected APR of {top_opp.get('apr', 0):.2f}%. "
                f"The new pool shows: {', '.join(metrics_comparison)}. "
                f"While not all metrics are improved, the combined performance score indicates a better opportunity."
            )
        else:
            current_pool = position_to_exit
            reasoning.append(
                "Current position remains optimal. Current metrics: "
                f"risk-adjusted returns {current_pool.get('sharpe_ratio', 0):.2f}, "
                f"market liquidity {current_pool.get('depth_score', 0):.2f}, "
                f"impermanent loss risk {current_pool.get('il_risk_score', 0):.2f}. "
                "No better opportunities found at this time."
            )
            return {
                "optimal_strategies": [], 
                "position_to_exit": {}, 
                "reasoning": base_description + " | ".join(reasoning)
            }
    else:
        # For new entries without current positions
        filtered_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        top_composite_score = filtered_opportunities[0]["composite_score"]

        optimal_opportunities = [
            opp for opp in filtered_opportunities[:max_pools]
            if opp["composite_score"] >= MIN_COMPOSITE_SCORE_RATIO * top_composite_score
        ]

        if not optimal_opportunities:
            reasoning.append("No opportunities currently meet our minimum requirements for safe and profitable trading.")
            return {
                "optimal_strategies": [], 
                "position_to_exit": {}, 
                "reasoning": base_description + " | ".join(reasoning)
            }
        
        # Add user-friendly reasoning for new entries
        for opp in optimal_opportunities:
            reasoning.append(
                f"Identified promising opportunity in {opp.get('dex_type', 'unknown')} pool "
                f"{opp.get('pool_address', 'unknown')} with expected APR of {opp.get('apr', 0):.2f}%. "
                f"The pool shows solid overall performance with metrics: "
                f"risk-adjusted returns {opp.get('sharpe_ratio', 0):.2f}, "
                f"market liquidity {opp.get('depth_score', 0):.2f}, "
                f"impermanent loss risk {opp.get('il_risk_score', 0):.2f}"
            )
    
    return {
        "optimal_strategies": optimal_opportunities, 
        "position_to_exit": position_to_exit,
        "reasoning": base_description + " | ".join(reasoning)
    }    

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    optimal_strategies = apply_risk_thresholds_and_select_optimal_strategy(**kwargs)
    optimal_strategies['logs'] = logs
    return optimal_strategies